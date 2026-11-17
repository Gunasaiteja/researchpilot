import logging
import os
import tempfile
import time
from collections import defaultdict

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("researchpilot")

app = FastAPI(title="ResearchPilot", version="1.0.0",
              description="Multi-agent RAG assistant on Vertex AI")

METRICS = {"requests": 0, "errors": 0, "latency_sum": 0.0, "by_path": defaultdict(int)}


def require_key(x_api_key: str = Header(default="")):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.middleware("http")
async def track(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS["errors"] += 1
        raise
    elapsed = time.perf_counter() - start
    METRICS["requests"] += 1
    METRICS["latency_sum"] += elapsed
    METRICS["by_path"][request.url.path] += 1
    if response.status_code >= 500:
        METRICS["errors"] += 1
    log.info("%s %s -> %s in %.0fms", request.method, request.url.path,
             response.status_code, elapsed * 1000)
    return response


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(require_key)])
def metrics():
    n = max(METRICS["requests"], 1)
    return {
        "requests": METRICS["requests"],
        "errors": METRICS["errors"],
        "avg_latency_ms": round(METRICS["latency_sum"] / n * 1000, 1),
        "by_path": dict(METRICS["by_path"]),
    }


@app.post("/upload", dependencies=[Depends(require_key)])
def upload(file: UploadFile = File(...)):
    from .ingest import ingest_file  # lazy: keeps startup and tests light

    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        return ingest_file(path, file.filename or "upload")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(path)


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    from .graph import get_graph, sources_from
    from .llm import get_callbacks

    t0 = time.perf_counter()
    out = get_graph().invoke({"question": req.question}, config={"callbacks": get_callbacks()})
    return {
        "answer": out["answer"],
        "grounded": out["grounded"],
        "attempts": out["attempts"],
        "sources": sources_from(out["docs"]),
        "trace": out["trace"],
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }


@app.post("/evaluate", dependencies=[Depends(require_key)])
def evaluate():
    from .evaluation import run_eval

    return run_eval()
