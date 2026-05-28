"""
src/scoring/aggregator.py
──────────────────────────
Aggregates TurnResult objects into summary statistics.
Outputs both JSON and CSV-friendly formats.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.scoring.orchestrator import TurnResult, FacetScore


def results_to_dict(results: list[TurnResult]) -> dict:
    """Convert a list of TurnResults to a serialisable dict."""
    turns = []
    for r in results:
        turn_dict = {
            "turn_id": r.turn_id,
            "turn_index": r.turn_index,
            "role": r.role,
            "content_preview": r.content_preview,
            "model_used": r.model_used,
            "scores": {
                fid: {
                    "score": fs.score,
                    "confidence": fs.confidence,
                    "rationale": fs.rationale,
                    **({"error": fs.error} if fs.error else {}),
                }
                for fid, fs in r.facet_scores.items()
            },
        }
        turns.append(turn_dict)
    return {"turns": turns, "turn_count": len(turns)}


def compute_summary(results: list[TurnResult]) -> dict:
    """
    Compute aggregate statistics across all turns and facets.
    Returns mean score, mean confidence, per-domain breakdown, flagged items.
    """
    all_scores: list[float] = []
    all_conf: list[float] = []
    domain_scores: dict[str, list[float]] = {}
    low_confidence: list[dict] = []
    low_safety_scores: list[dict] = []

    for r in results:
        for fid, fs in r.facet_scores.items():
            all_scores.append(fs.score)
            all_conf.append(fs.confidence)

            # Track low-confidence items for human review
            if fs.confidence < 0.5:
                low_confidence.append({
                    "turn_index": r.turn_index,
                    "facet_id": fid,
                    "score": fs.score,
                    "confidence": fs.confidence,
                })

            # Flag safety issues (low score on safety facets)
            if "safety" in fid or fid in ["dishonesty", "hostility", "hatefulness", "harmfulness"]:
                if fs.score <= 2:
                    low_safety_scores.append({
                        "turn_index": r.turn_index,
                        "facet_id": fid,
                        "score": fs.score,
                    })

    def safe_mean(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "overall_mean_score": safe_mean(all_scores),
        "overall_mean_confidence": safe_mean(all_conf),
        "total_scores_computed": len(all_scores),
        "low_confidence_flags": low_confidence[:20],    # top 20
        "safety_flags": low_safety_scores,
        "turns_evaluated": len(results),
    }


def save_results(
    results: list[TurnResult],
    output_dir: str | Path,
    conversation_id: str = "conv",
) -> dict[str, Path]:
    """Save results as JSON and summary. Returns paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_dict = results_to_dict(results)
    summary = compute_summary(results)
    results_dict["summary"] = summary

    json_path = out / f"{conversation_id}_scores.json"
    json_path.write_text(json.dumps(results_dict, indent=2, ensure_ascii=False))

    return {"json": json_path}
