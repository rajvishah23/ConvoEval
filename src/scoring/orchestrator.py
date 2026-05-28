"""
src/scoring/orchestrator.py
────────────────────────────
Async chunked scoring orchestrator.
Chunk size is small (5) to stay within Groq/HF token limits.
"""
from __future__ import annotations

import asyncio, json, logging, os, re
from dataclasses import dataclass, field
from typing import Optional

from src.models.ollama_client import OllamaClient
from src.pipeline.facet_registry import Facet, FacetRegistry
from src.pipeline.turn_splitter import Turn

logger = logging.getLogger(__name__)

# Smaller chunk = shorter prompt = fits within free API token limits
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5"))

SYSTEM_PROMPT = """You are a conversation quality evaluator. Score conversation turns on specific facets.
Score scale: 1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent.
Always return valid JSON only."""


@dataclass
class FacetScore:
    facet_id: str
    score: int              # 1–5
    rationale: str
    confidence: float = 0.5
    error: Optional[str] = None


@dataclass
class TurnResult:
    turn_id: str
    turn_index: int
    role: str
    content_preview: str
    facet_scores: dict[str, FacetScore] = field(default_factory=dict)
    model_used: str = ""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_score_json(raw: str, expected_ids: list[str]) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        fixed = re.sub(r',\s*([}\]])', r'\1', match.group()).replace("'", '"')
        try:
            parsed = json.loads(fixed)
        except:
            return {}

    scores_block = parsed.get("scores", parsed)
    result = {}
    for fid in expected_ids:
        if fid in scores_block:
            entry = scores_block[fid]
            if isinstance(entry, dict):
                score     = entry.get("score", 3)
                rationale = entry.get("rationale", "")
            elif isinstance(entry, (int, float)):
                score, rationale = int(entry), ""
            else:
                continue
            try:
                score = max(1, min(5, int(score)))
            except:
                score = 3
            result[fid] = {"score": score, "rationale": str(rationale)}
    return result


def _parse_confidence_json(raw: str, expected_ids: list[str]) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
        conf   = parsed.get("confidence", parsed)
        return {
            fid: max(0.0, min(1.0, float(conf[fid])))
            for fid in expected_ids if fid in conf
        }
    except:
        return {}


# ── Per-chunk scoring (3-turn to keep prompts short) ─────────────────────────

async def _score_chunk(client: OllamaClient, turn: Turn, facets: list[Facet]) -> dict[str, FacetScore]:
    facet_ids  = [f.facet_id for f in facets]
    turn_block = turn.to_prompt_block(include_context=True)

    # ── Turn 1+2: brief analysis ──────────────────────────────────────────────
    analysis_prompt = (
        f"Analyse this conversation turn briefly (2-3 sentences):\n\n{turn_block}"
    )
    try:
        analysis = await client.chat(
            messages=[{"role": "user", "content": analysis_prompt}],
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=150,
        )
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
        analysis = "Analysis unavailable."

    # ── Turn 3+4: scoring ─────────────────────────────────────────────────────
    # Keep facet descriptions short to stay within token limits
    facet_lines = []
    for f in facets:
        facet_lines.append(f"- {f.facet_id}: {f.prompt_hint[:80]}")
    facets_block = "\n".join(facet_lines)

    example = json.dumps(
        {"scores": {fid: {"score": 3, "rationale": "reason"} for fid in facet_ids[:2]}},
        indent=2
    )

    scoring_prompt = (
        f"Score these facets for the turn above.\n\n"
        f"Facets:\n{facets_block}\n\n"
        f"Return ONLY JSON like:\n{example}\n\n"
        f"Include all {len(facet_ids)} facets. Scores must be integers 1-5."
    )

    try:
        score_raw = await client.chat(
            messages=[
                {"role": "user",      "content": analysis_prompt},
                {"role": "assistant", "content": analysis},
                {"role": "user",      "content": scoring_prompt},
            ],
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return {
            fid: FacetScore(fid, 3, "Scoring failed.", 0.0, str(e))
            for fid in facet_ids
        }

    parsed = _parse_score_json(score_raw, facet_ids)

    # ── Turn 5+6: confidence ──────────────────────────────────────────────────
    scores_summary = {fid: parsed.get(fid, {}).get("score", 3) for fid in facet_ids}
    conf_prompt = (
        f"Confidence (0.0-1.0) for each score: {json.dumps(scores_summary)}\n"
        f'Return ONLY JSON: {{"confidence": {{"{facet_ids[0]}": 0.85, ...}}}}'
    )
    try:
        conf_raw = await client.chat(
            messages=[
                {"role": "user",      "content": analysis_prompt},
                {"role": "assistant", "content": analysis},
                {"role": "user",      "content": scoring_prompt},
                {"role": "assistant", "content": score_raw},
                {"role": "user",      "content": conf_prompt},
            ],
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=150,
        )
        confidences = _parse_confidence_json(conf_raw, facet_ids)
    except Exception as e:
        logger.warning(f"Confidence failed: {e}")
        confidences = {}

    # ── Assemble ──────────────────────────────────────────────────────────────
    results = {}
    for fid in facet_ids:
        if fid in parsed:
            s = parsed[fid]
            results[fid] = FacetScore(
                facet_id=fid,
                score=s["score"],
                rationale=s.get("rationale", ""),
                confidence=confidences.get(fid, 0.75),
            )
        else:
            results[fid] = FacetScore(fid, 3, "Could not parse model output.", 0.1, "parse_failure")
    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ScoringOrchestrator:
    def __init__(self, client: OllamaClient, registry: FacetRegistry,
                 chunk_size: int = CHUNK_SIZE, max_concurrent_chunks: int = 2):
        self.client     = client
        self.registry   = registry
        self.chunk_size = chunk_size
        self.semaphore  = asyncio.Semaphore(max_concurrent_chunks)

    async def score_turn(self, turn: Turn, facet_ids: Optional[list[str]] = None) -> TurnResult:
        if facet_ids:
            facets = self.registry.get_many(facet_ids)
        else:
            facets = [f for f in self.registry.all() if getattr(f, 'evaluable_from_text', True)]

        chunks = [facets[i:i+self.chunk_size] for i in range(0, len(facets), self.chunk_size)]

        result = TurnResult(
            turn_id=turn.turn_id, turn_index=turn.index,
            role=turn.role, content_preview=turn.content[:120],
            model_used=self.client.model,
        )

        async def _bounded(chunk):
            async with self.semaphore:
                return await _score_chunk(self.client, turn, chunk)

        chunk_results = await asyncio.gather(*[_bounded(c) for c in chunks])
        for cr in chunk_results:
            result.facet_scores.update(cr)
        return result

    async def score_conversation(self, turns: list[Turn],
                                  facet_ids: Optional[list[str]] = None) -> list[TurnResult]:
        results = []
        for turn in turns:
            logger.info(f"Scoring turn {turn.index} ({turn.role})...")
            results.append(await self.score_turn(turn, facet_ids))
        return results
