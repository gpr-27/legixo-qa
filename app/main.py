"""FastAPI service exposing the grounded Q&A endpoint.

The clients and the compiled graph are built once at startup (lifespan) and reused
for every request.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import groq
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from google.genai import errors as genai_errors
from langgraph.errors import GraphRecursionError
from pinecone.exceptions import PineconeException

from . import clients
from .graph import build_graph
from .llm import REFUSAL
from .schemas import AskRequest, AskResponse
from .settings import get_settings

_log = logging.getLogger(__name__)
_FALLBACK_RECURSION_LIMIT = 12
_UPSTREAM_ERRORS = (groq.APIError, genai_errors.APIError, PineconeException)
_UI_FILE = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pc = clients.get_pinecone(settings)
    index = clients.get_index(pc, settings)
    embedder = clients.Embedder(settings)
    groq = clients.get_groq(settings)

    app.state.graph = build_graph(index, embedder, groq, settings)
    # recursion_limit counts super-steps; keep it well above the expand + per-loop nodes.
    app.state.recursion_limit = settings.max_loops * 3 + 7
    yield


app = FastAPI(title="Legixo Q&A", version="1.0.0", lifespan=lifespan)


def get_graph(request: Request):
    return request.app.state.graph


@app.get("/", include_in_schema=False)
def home():
    """Serve the minimal chat UI (it just calls POST /ask)."""
    return FileResponse(_UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, trace: bool = False, graph=Depends(get_graph)):
    # Sync handler on purpose: the Groq/Gemini/Pinecone clients are blocking, so
    # FastAPI runs this in its threadpool and the event loop is never blocked.
    initial = {
        "question": req.question,
        "queries": [],
        "documents": [],
        "answer": "",
        "citations": [],
        "retries": 0,
        "steps": [],
    }
    limit = getattr(app.state, "recursion_limit", _FALLBACK_RECURSION_LIMIT)
    try:
        final = graph.invoke(initial, {"recursion_limit": limit})
    except GraphRecursionError:
        # The loop guard tripped — abstain cleanly rather than 500.
        final = {"answer": REFUSAL, "citations": [], "steps": ["recursion_limit_hit"]}
    except _UPSTREAM_ERRORS as exc:
        # A provider/network failure — distinct from a bug, which we let surface as 500.
        _log.warning("upstream error on /ask: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": f"upstream service error: {type(exc).__name__}"},
        )

    return AskResponse(
        answer=final["answer"],
        citations=final["citations"],
        trace=final["steps"] if trace else None,
    )
