"""Multi-agent workflow built with LangGraph.

Planner -> Retriever -> Writer -> Fact-Checker -> (loop back to Writer or finish)
"""
from functools import lru_cache
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .config import settings
from .llm import get_llm
from .vectorstore import get_store


class Plan(BaseModel):
    queries: List[str] = Field(description="1 to 3 focused search queries")


class Verdict(BaseModel):
    grounded: bool = Field(description="True if every claim is supported by the context")
    issues: str = Field(default="", description="Unsupported or wrong claims, if any")


class State(TypedDict, total=False):
    question: str
    queries: List[str]
    docs: List[Document]
    answer: str
    grounded: bool
    feedback: str
    attempts: int
    trace: List[str]


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', '?')}, p.{d.metadata.get('page', '-')})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


# ---------- Agents (graph nodes) ----------
def planner(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Break the user question into 1-3 short, focused search queries for a document index."),
        ("human", "{question}"),
    ])
    plan = (prompt | get_llm().with_structured_output(Plan)).invoke({"question": state["question"]})
    queries = plan.queries or [state["question"]]
    return {"queries": queries, "attempts": 0, "trace": [f"planner: {queries}"]}


def retriever(state: State) -> State:
    store = get_store()
    seen, docs = set(), []
    for q in state["queries"]:
        # MMR keeps results relevant but diverse
        for d in store.max_marginal_relevance_search(q, k=settings.top_k, fetch_k=settings.top_k * 3):
            key = d.page_content[:120]
            if key not in seen:
                seen.add(key)
                docs.append(d)
    return {"docs": docs, "trace": state["trace"] + [f"retriever: {len(docs)} chunks"]}


def writer(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer ONLY from the context. Cite sources as [1], [2]. "
         "If the context is insufficient, say you don't know.\n"
         "Reviewer feedback to fix (if any): {feedback}\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    out = (prompt | get_llm()).invoke({
        "question": state["question"],
        "context": _format_docs(state["docs"]),
        "feedback": state.get("feedback", "none"),
    })
    attempts = state.get("attempts", 0) + 1
    return {"answer": out.content, "attempts": attempts,
            "trace": state["trace"] + [f"writer: attempt {attempts}"]}


def fact_checker(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict fact-checker. Decide whether the answer is fully supported by the context."),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ])
    v = (prompt | get_llm().with_structured_output(Verdict)).invoke({
        "context": _format_docs(state["docs"]), "answer": state["answer"],
    })
    return {"grounded": v.grounded, "feedback": v.issues,
            "trace": state["trace"] + [f"fact_checker: grounded={v.grounded}"]}


def route_after_check(state: State) -> str:
    if state["grounded"] or state["attempts"] > settings.max_retries:
        return END
    return "writer"


@lru_cache
def get_graph():
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("retriever", retriever)
    g.add_node("writer", writer)
    g.add_node("fact_checker", fact_checker)
    g.set_entry_point("planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "writer")
    g.add_edge("writer", "fact_checker")
    g.add_conditional_edges("fact_checker", route_after_check, {"writer": "writer", END: END})
    return g.compile()


def sources_from(docs: List[Document]) -> list:
    return [{
        "id": i,
        "source": d.metadata.get("source"),
        "page": d.metadata.get("page"),
        "snippet": d.page_content[:200],
    } for i, d in enumerate(docs, 1)]
