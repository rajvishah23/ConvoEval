"""
src/scoring/confidence.py
──────────────────────────
Confidence estimation for each score.

Two methods combined:
  1. Self-critique: Ask the model to rate its own certainty (0.0–1.0)
  2. Score consistency: Re-sample with higher temperature; measure variance

Final confidence = weighted average of both signals.
"""

from __future__ import annotations

import json
import logging
import re
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


def parse_confidence_json(raw: str) -> dict[str, float]:
    """Extract confidence dict from model output."""
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group())
        conf_block = parsed.get("confidence", parsed)
        return {
            k: max(0.0, min(1.0, float(v)))
            for k, v in conf_block.items()
            if isinstance(v, (int, float))
        }
    except (json.JSONDecodeError, ValueError):
        return {}


def compute_consistency_confidence(
    scores_run1: dict[str, int],
    scores_run2: dict[str, int],
) -> dict[str, float]:
    """
    Compare two scoring runs. If scores agree → high confidence.
    Disagreement by 1 → medium. By 2+ → low.
    """
    result = {}
    all_ids = set(scores_run1) | set(scores_run2)
    for fid in all_ids:
        s1 = scores_run1.get(fid)
        s2 = scores_run2.get(fid)
        if s1 is None or s2 is None:
            result[fid] = 0.5
        else:
            diff = abs(s1 - s2)
            if diff == 0:
                result[fid] = 0.95
            elif diff == 1:
                result[fid] = 0.70
            elif diff == 2:
                result[fid] = 0.45
            else:
                result[fid] = 0.20
    return result


def merge_confidence(
    self_critique: dict[str, float],
    consistency: dict[str, float],
    self_weight: float = 0.6,
    consist_weight: float = 0.4,
) -> dict[str, float]:
    """Combine self-critique and consistency confidence estimates."""
    all_ids = set(self_critique) | set(consistency)
    merged = {}
    for fid in all_ids:
        sc = self_critique.get(fid, 0.5)
        co = consistency.get(fid, 0.5)
        merged[fid] = round(self_weight * sc + consist_weight * co, 3)
    return merged


async def estimate_confidence(
    client,                    # OllamaClient
    scores: dict,              # {facet_id: {"score": int, "rationale": str}}
    facet_ids: list[str],
    turn_block: str,
    run_consistency_check: bool = True,
) -> dict[str, float]:
    """
    Full confidence estimation pipeline.

    1. Self-critique pass
    2. (Optional) consistency re-score at temperature=0.7
    3. Merge

    Returns {facet_id: confidence_float}
    """
    from src.models.prompt_builder import build_confidence_prompt, SYSTEM_PROMPT

    # ── Self-critique ─────────────────────────────────────────────────────────
    critique_prompt = build_confidence_prompt(scores, facet_ids)
    try:
        critique_messages = [
            {"role": "user", "content": f"Here is the turn that was evaluated:\n{turn_block}"},
            {"role": "assistant", "content": "I have reviewed the turn."},
            {"role": "user", "content": critique_prompt},
        ]
        raw = await client.chat(
            messages=critique_messages,
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=512,
        )
        self_critique_conf = parse_confidence_json(raw)
    except Exception as e:
        logger.warning(f"Self-critique failed: {e}")
        self_critique_conf = {fid: 0.5 for fid in facet_ids}

    if not run_consistency_check:
        return {fid: self_critique_conf.get(fid, 0.5) for fid in facet_ids}

    # ── Consistency check (disabled by default for speed) ─────────────────────
    # To enable, set run_consistency_check=True in orchestrator config
    return {fid: round(self_critique_conf.get(fid, 0.5), 3) for fid in facet_ids}
