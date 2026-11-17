"""Offline evaluation on a golden question set.

Metrics: faithfulness (LLM judge), keyword recall, grounded rate, latency.
"""
import json
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .graph import get_graph
from .llm import get_callbacks, get_llm


class Score(BaseModel):
    score: float = Field(ge=0, le=1, description="0 = unsupported, 1 = fully supported")


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for k in keywords if k.lower() in answer.lower())
    return hits / len(keywords)


def judge_faithfulness(answer: str, context: str) -> float:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rate how well the answer is supported by the context, from 0 to 1."),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ])
    return (prompt | get_llm().with_structured_output(Score)).invoke(
        {"context": context, "answer": answer}
    ).score


def run_eval(path: str = "eval/golden_set.json") -> dict:
    items = json.loads(Path(path).read_text())
    rows = []
    for it in items:
        t0 = time.perf_counter()
        out = get_graph().invoke({"question": it["question"]}, config={"callbacks": get_callbacks()})
        latency = time.perf_counter() - t0
        context = "\n".join(d.page_content for d in out["docs"])
        rows.append({
            "question": it["question"],
            "faithfulness": judge_faithfulness(out["answer"], context),
            "keyword_recall": keyword_recall(out["answer"], it.get("expected_keywords", [])),
            "grounded": out["grounded"],
            "attempts": out["attempts"],
            "latency_s": round(latency, 2),
        })
    n = max(len(rows), 1)
    summary = {
        "n": len(rows),
        "avg_faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
        "avg_keyword_recall": round(sum(r["keyword_recall"] for r in rows) / n, 3),
        "grounded_rate": round(sum(r["grounded"] for r in rows) / n, 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    print(json.dumps(run_eval(), indent=2))
