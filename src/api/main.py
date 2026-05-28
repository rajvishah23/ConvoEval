"""
src/api/main.py
────────────────
FastAPI REST API for ConvoEval.

Endpoints:
  GET  /health          → Ollama + registry status
  GET  /facets          → List all facets (with optional domain filter)
  POST /score           → Score a conversation
  POST /score/turn      → Score a single turn
  GET  /facets/reload   → Hot-reload facets CSV
"""

from __future__ import annotations

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.ollama_client import OllamaClient, DEFAULT_OLLAMA_URL, DEFAULT_MODEL
from src.pipeline.facet_registry import FacetRegistry, get_registry
from src.pipeline.turn_splitter import split_conversation
from src.scoring.orchestrator import ScoringOrchestrator
from src.scoring.aggregator import results_to_dict, compute_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ConvoEval API",
    description="Production-ready conversation turn evaluation across 300–5000 facets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────

FACETS_PATH = Path(os.getenv("FACETS_PATH", "data/facets_processed.csv"))
OLLAMA_URL = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

_registry: Optional[FacetRegistry] = None
_client: Optional[OllamaClient] = None


@app.on_event("startup")
async def startup():
    global _registry, _client
    _registry = get_registry()
    if FACETS_PATH.exists():
        _registry.load_csv(FACETS_PATH)
    else:
        logger.warning(f"Facets CSV not found at {FACETS_PATH}. Registry is empty.")

    _client = OllamaClient(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    logger.info(f"ConvoEval API ready. {len(_registry)} facets loaded.")


@app.on_event("shutdown")
async def shutdown():
    if _client:
        await _client.close()


# ── Request/Response models ───────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str


class ScoreRequest(BaseModel):
    conversation: list[Message] = Field(..., min_items=1)
    facet_ids: Optional[list[str]] = Field(None, description="Subset of facet IDs to score. Defaults to all.")
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    score_user_turns: bool = False


class SingleTurnRequest(BaseModel):
    content: str
    role: str = "assistant"
    context: list[Message] = []
    facet_ids: Optional[list[str]] = None
    model: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    registry_summary = _registry.summary() if _registry else {}
    ollama_status = await _client.health_check() if _client else {"status": "not_initialized"}
    return {
        "api": "ok",
        "registry": registry_summary,
        "ollama": ollama_status,
    }


@app.get("/facets")
async def list_facets(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    evaluable_only: bool = Query(True, description="Only return text-evaluable facets"),
    limit: int = Query(100, le=5000),
):
    if _registry is None:
        raise HTTPException(503, "Registry not initialized")

    facets = _registry.by_domain(domain) if domain else _registry.all()

    result = []
    for f in facets[:limit]:
        result.append({
            "facet_id": f.facet_id,
            "facet_name": f.facet_name,
            "domain": f.domain,
            "description": f.description,
            "polarity": f.polarity,
            "requires_context": f.requires_context,
            "chunk_group": f.chunk_group,
        })
    return {"facets": result, "total": len(facets)}


@app.get("/facets/domains")
async def list_domains():
    if _registry is None:
        raise HTTPException(503, "Registry not initialized")
    return {"domains": _registry.domains(), "summary": _registry.summary()}


@app.get("/facets/reload")
async def reload_facets():
    if _registry is None:
        raise HTTPException(503, "Registry not initialized")
    count = _registry.reload()
    return {"status": "reloaded", "facets_loaded": count}


@app.post("/score")
async def score_conversation(req: ScoreRequest):
    if _registry is None or _client is None:
        raise HTTPException(503, "Service not ready")

    # Use custom model if provided
    client = _client
    if req.model and req.model != _client.model:
        client = OllamaClient(base_url=OLLAMA_URL, model=req.model)

    messages = [{"role": m.role, "content": m.content} for m in req.conversation]
    turns = split_conversation(
        messages,
        conversation_id=req.conversation_id,
        score_user_turns=req.score_user_turns,
    )

    if not turns:
        return {"error": "No scoreable turns found. Make sure there are assistant turns.", "turns": []}

    orchestrator = ScoringOrchestrator(client=client, registry=_registry)
    results = await orchestrator.score_conversation(turns, facet_ids=req.facet_ids)

    output = results_to_dict(results)
    output["summary"] = compute_summary(results)
    return output


@app.post("/score/turn")
async def score_single_turn(req: SingleTurnRequest):
    if _registry is None or _client is None:
        raise HTTPException(503, "Service not ready")

    client = _client
    if req.model and req.model != _client.model:
        client = OllamaClient(base_url=OLLAMA_URL, model=req.model)

    # Build a synthetic conversation with the single turn
    messages = [{"role": m.role, "content": m.content} for m in req.context]
    messages.append({"role": req.role, "content": req.content})

    turns = split_conversation(messages, score_user_turns=(req.role == "user"))
    if not turns:
        raise HTTPException(400, "No scoreable turn found.")

    orchestrator = ScoringOrchestrator(client=client, registry=_registry)
    result = await orchestrator.score_turn(turns[-1], facet_ids=req.facet_ids)

    return {
        "turn_id": result.turn_id,
        "role": result.role,
        "scores": {
            fid: {
                "score": fs.score,
                "confidence": fs.confidence,
                "rationale": fs.rationale,
            }
            for fid, fs in result.facet_scores.items()
        },
    }
