"""
src/scoring/orchestrator.py
────────────────────────────
Core scoring orchestrator. For each conversation turn:

  1. Load relevant facets from the registry (chunked)
  2. For each chunk: multi-turn chat (analysis → scoring → confidence)
  3. Parse JSON scores + confidence
  4. Return structured results

No one-shot prompts — every scoring call uses a 3-turn chain:
  Turn 1 [user]     : Present turn for analysis
  Turn 2 [assistant]: CoT analysis (model generates)
  Turn 3 [user]     : Request JSON scores
  Turn 4 [assistant]: JSON output (model generates)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.models.ollama_client import OllamaClient
from src.pipeline.facet_registry import Facet, FacetRegistry
from src.pipeline.turn_splitter import Turn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert conversation quality evaluator with deep expertise in linguistics, pragmatics, safety analysis, and affective computing.

Your job is to evaluate a single conversation turn against specific quality facets using a two-step process:

STEP 1 — ANALYSIS: Think carefully about the turn and its context. Be thorough but concise (3-5 sentences).
STEP 2 — SCORING: Assign integer scores 1-5 per facet.

Score meaning:
  1 = Poor        (fails the facet entirely)
  2 = Below Avg   (partial failure, noticeable issues)
  3 = Average     (meets minimum expectations)
  4 = Good        (clearly above average)
  5 = Excellent   (exemplary, near-perfect)

Always return valid JSON when asked. Never refuse to score."""


@dataclass
class FacetScore:
    facet_id: str
    score: int                      # 1–5
    rationale: str
    confidence: float = 0.5         # 0.0–1.0
    error: Optional[str] = None


@dataclass
class TurnResult:
    turn_id: str
    turn_index: int
    role: str
    content_preview: str            # first 120 chars
    facet_scores: dict[str, FacetScore] = field(default_factory=dict)
    analysis: str = ""
    model_used: str = ""


# ── JSON Parsing ──────────────────────────────────────────────────────────────

def _parse_score_json(raw: str, expected_ids: list[str]) -> dict[str, dict]:
    """Extract score dict from model output. Robust to markdown fences and prose."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Try to find a JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        # Try to fix common issues: trailing commas, single quotes
        fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
        fixed = fixed.replace("'", '"')
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            return {}

    scores_block = parsed.get("scores", parsed)

    result = {}
    for fid in expected_ids:
        if fid in scores_block:
            entry = scores_block[fid]
            if isinstance(entry, dict):
                score = entry.get("score", 3)
                rationale = entry.get("rationale", "")
            elif isinstance(entry, (int, float)):
                score = int(entry)
                rationale = ""
            else:
                continue
            try:
                score = max(1, min(5, int(score)))
            except (TypeError, ValueError):
                score = 3
            result[fid] = {"score": score, "rationale": str(rationale)}

    return result


def _parse_confidence_json(raw: str, expected_ids: list[str]) -> dict[str, float]:
    """Extract confidence values from model output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
        conf_block = parsed.get("confidence", parsed)
        return {
            fid: max(0.0, min(1.0, float(conf_block[fid])))
            for fid in expected_ids
            if fid in conf_block
        }
    except Exception:
        return {}


# ── Per-chunk scoring ─────────────────────────────────────────────────────────

async def _score_chunk(
    client: OllamaClient,
    turn: Turn,
    facets: list[Facet],
    temperature: float = 0.1,
) -> dict[str, FacetScore]:
    """
    Score one chunk of facets for one turn.
    Uses 4-turn multi-turn conversation (no one-shot).
    """
    facet_ids = [f.facet_id for f in facets]
    turn_block = turn.to_prompt_block(include_context=True)

    # ── Turn 1+2: Analysis ────────────────────────────────────────────────────
    analysis_prompt = (
        f"Please carefully analyse the following conversation turn:\n\n"
        f"{turn_block}\n\n"
        f"Think step by step:\n"
        f"1. Role and purpose of this turn\n"
        f"2. Linguistic quality\n"
        f"3. Pragmatic alignment with context\n"
        f"4. Emotional tone\n"
        f"5. Any safety concerns\n\n"
        f"Provide a concise analysis (3–5 sentences)."
    )

    try:
        analysis = await client.chat(
            messages=[{"role": "user", "content": analysis_prompt}],
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=300,
        )
    except Exception as e:
        logger.warning(f"Analysis call failed for turn {turn.turn_id}: {e}")
        analysis = "Analysis unavailable due to model error."

    # ── Turn 3+4: Scoring ─────────────────────────────────────────────────────
    facet_lines = []
    for f in facets:
        rubric_text = "\n".join(f"    {k}: {v}" for k, v in sorted(f.rubric.items()))
        facet_lines.append(
            f"### {f.facet_id}\n"
            f"  Name: {f.facet_name} | Domain: {f.domain}\n"
            f"  Instruction: {f.prompt_hint}\n"
            f"  Rubric:\n{rubric_text}"
        )
    facets_block = "\n\n".join(facet_lines)

    # Build example JSON format with first 2 facet IDs
    example_entries = "\n".join(
        f'    "{fid}": {{"score": 3, "rationale": "one sentence reason"}}'
        for fid in facet_ids[:2]
    )

    scoring_prompt = (
        f"Based on your analysis above, now score the turn on these {len(facets)} facets.\n\n"
        f"{facets_block}\n\n"
        f"Return ONLY valid JSON — no extra text, no markdown fences:\n"
        f'{{\n  "scores": {{\n{example_entries},\n    ... (one per facet)\n  }}\n}}\n\n'
        f"Every facet_id listed above must appear. score must be an integer 1–5."
    )

    try:
        score_raw = await client.chat(
            messages=[
                {"role": "user", "content": analysis_prompt},
                {"role": "assistant", "content": analysis},
                {"role": "user", "content": scoring_prompt},
            ],
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=800,
        )
    except Exception as e:
        logger.error(f"Scoring call failed for turn {turn.turn_id}: {e}")
        return {
            fid: FacetScore(fid, 3, "Scoring failed due to model error.", 0.0, str(e))
            for fid in facet_ids
        }

    parsed_scores = _parse_score_json(score_raw, facet_ids)

    # ── Turn 5+6: Confidence ──────────────────────────────────────────────────
    scores_summary = json.dumps(
        {fid: parsed_scores.get(fid, {}).get("score", 3) for fid in facet_ids},
        indent=2,
    )
    confidence_prompt = (
        f"You just assigned these scores:\n{scores_summary}\n\n"
        f"For each score, how confident are you (0.0–1.0)?\n"
        f"Consider: Is the rubric clearly applicable? Is there ambiguity?\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{\n  "confidence": {{\n    "facet_id": 0.85,\n    ...\n  }}\n}}'
    )

    try:
        conf_raw = await client.chat(
            messages=[
                {"role": "user", "content": analysis_prompt},
                {"role": "assistant", "content": analysis},
                {"role": "user", "content": scoring_prompt},
                {"role": "assistant", "content": score_raw},
                {"role": "user", "content": confidence_prompt},
            ],
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=400,
        )
        confidences = _parse_confidence_json(conf_raw, facet_ids)
    except Exception as e:
        logger.warning(f"Confidence call failed: {e}")
        confidences = {}

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {}
    for fid in facet_ids:
        if fid in parsed_scores:
            s = parsed_scores[fid]
            results[fid] = FacetScore(
                facet_id=fid,
                score=s["score"],
                rationale=s.get("rationale", ""),
                confidence=confidences.get(fid, 0.5),
            )
        else:
            results[fid] = FacetScore(
                facet_id=fid,
                score=3,
                rationale="Could not parse model output.",
                confidence=0.1,
                error="parse_failure",
            )

    return results


# ── Main orchestrator ─────────────────────────────────────────────────────────

class ScoringOrchestrator:
    def __init__(
        self,
        client: OllamaClient,
        registry: FacetRegistry,
        chunk_size: int = 20,
        max_concurrent_chunks: int = 3,
    ) -> None:
        self.client = client
        self.registry = registry
        self.chunk_size = chunk_size
        self.semaphore = asyncio.Semaphore(max_concurrent_chunks)

    async def score_turn(
        self,
        turn: Turn,
        facet_ids: Optional[list[str]] = None,
    ) -> TurnResult:
        """Score a single turn against the given facet IDs (or all facets)."""
        if facet_ids:
            facets = self.registry.get_many(facet_ids)
        else:
            facets = [f for f in self.registry.all() if getattr(f, 'evaluable_from_text', True)]

        # Split into chunks
        chunks = [
            facets[i: i + self.chunk_size]
            for i in range(0, len(facets), self.chunk_size)
        ]

        result = TurnResult(
            turn_id=turn.turn_id,
            turn_index=turn.index,
            role=turn.role,
            content_preview=turn.content[:120],
            model_used=self.client.model,
        )

        async def _bounded_chunk(chunk):
            async with self.semaphore:
                return await _score_chunk(self.client, turn, chunk)

        chunk_results = await asyncio.gather(*[_bounded_chunk(c) for c in chunks])

        for chunk_result in chunk_results:
            result.facet_scores.update(chunk_result)

        return result

    async def score_conversation(
        self,
        turns: list[Turn],
        facet_ids: Optional[list[str]] = None,
    ) -> list[TurnResult]:
        """Score all turns in a conversation sequentially."""
        results = []
        for turn in turns:
            logger.info(f"Scoring turn {turn.index} ({turn.role})...")
            result = await self.score_turn(turn, facet_ids)
            results.append(result)
        return results
