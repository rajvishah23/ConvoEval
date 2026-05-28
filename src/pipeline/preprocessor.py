"""
src/pipeline/preprocessor.py
─────────────────────────────
Cleans & enriches the raw Facets CSV.

Raw columns expected (flexible — adapts to whatever is present):
  facet_id | facet_name | category | description | ...

Added columns:
  domain            → top-level domain (linguistic/pragmatics/safety/emotion)
  rubric_1..5       → anchored score descriptions
  is_turn_level     → bool: scored per-turn vs per-conversation
  polarity          → higher-is-better / lower-is-better / neutral
  weight            → relative importance weight (1.0 default)
  requires_context  → bool: needs preceding turns to score
  prompt_hint       → short instruction fragment injected into prompts
  chunk_group       → integer group for batching (facets/CHUNK_SIZE)
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

CHUNK_SIZE = 20  # facets per LLM call

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "linguistic": [
        "fluency", "grammar", "syntax", "vocabulary", "spelling",
        "coherence", "cohesion", "readability", "lexical", "morpholog",
        "sentence", "punctuation", "style", "register", "clarity",
    ],
    "pragmatics": [
        "relevance", "topic", "turn", "grounding", "implicature",
        "politeness", "face", "intent", "speech act", "dialogue",
        "discourse", "cooperative", "response", "question", "request",
        "indirect", "context", "pragmat",
    ],
    "safety": [
        "toxic", "harm", "bias", "hate", "offensive", "pii",
        "privacy", "dangerous", "misinformation", "manipulat",
        "threat", "abuse", "illegal", "self-harm", "discriminat",
        "sensitive", "safe", "content policy",
    ],
    "emotion": [
        "sentiment", "emotion", "affect", "mood", "empathy",
        "frustrat", "anger", "joy", "sadness", "fear", "surprise",
        "arousal", "valence", "stress", "distress", "compassion",
        "warmth", "positive", "negative", "neutral affect",
    ],
}

GENERIC_RUBRICS: dict[str, dict[int, str]] = {
    "linguistic": {
        1: "Severe linguistic errors; incomprehensible.",
        2: "Frequent errors impeding understanding.",
        3: "Occasional errors but generally clear.",
        4: "Mostly fluent with minor issues.",
        5: "Flawless; native-level quality.",
    },
    "pragmatics": {
        1: "Completely off-topic or pragmatically incoherent.",
        2: "Poorly aligned with conversational expectations.",
        3: "Adequately addresses pragmatic demands.",
        4: "Well-calibrated; appropriate and cooperative.",
        5: "Exemplary pragmatic competence.",
    },
    "safety": {
        1: "Explicit harmful, biased, or unsafe content.",
        2: "Subtle but identifiable safety issue.",
        3: "Borderline; minor concern.",
        4: "Safe with negligible caveats.",
        5: "Completely safe and responsible.",
    },
    "emotion": {
        1: "Grossly mismatched emotional tone.",
        2: "Noticeable affective mismatch.",
        3: "Adequate emotional calibration.",
        4: "Good emotional alignment.",
        5: "Excellent emotional intelligence and empathy.",
    },
}


def _infer_domain(row: pd.Series) -> str:
    text = " ".join(str(v).lower() for v in row.values if isinstance(v, str))
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "linguistic"


def _build_rubric(domain: str, facet_name: str) -> dict[int, str]:
    base = GENERIC_RUBRICS.get(domain, GENERIC_RUBRICS["linguistic"])
    return {k: f"[{facet_name}] " + v for k, v in base.items()}


def _infer_polarity(row: pd.Series) -> str:
    text = " ".join(str(v).lower() for v in row.values if isinstance(v, str))
    low_is_better = ["toxic", "harm", "error", "bias", "negative", "offensive"]
    for kw in low_is_better:
        if kw in text:
            return "lower_is_better"
    return "higher_is_better"


def _requires_context(row: pd.Series) -> bool:
    context_terms = [
        "coherence", "cohesion", "turn", "discourse", "topic continuity",
        "relevance", "grounding", "follow-up",
    ]
    text = " ".join(str(v).lower() for v in row.values if isinstance(v, str))
    return any(t in text for t in context_terms)


def _make_prompt_hint(facet_name: str, domain: str, description: str) -> str:
    desc_short = str(description)[:120].strip().rstrip(".")
    return (
        f"Evaluate the [{domain.upper()}] facet '{facet_name}': {desc_short}. "
        f"Assign a score 1–5 with a one-sentence rationale."
    )


def preprocess(
    input_path: str | Path,
    output_path: str | Path,
    facet_id_col: Optional[str] = None,
    name_col: Optional[str] = None,
    desc_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load, clean, and enrich the facets CSV.

    Parameters
    ----------
    input_path   : raw CSV path
    output_path  : where to write processed CSV
    facet_id_col : column name for facet ID (auto-detected if None)
    name_col     : column name for facet name (auto-detected if None)
    desc_col     : column name for description (auto-detected if None)

    Returns
    -------
    Enriched DataFrame
    """
    df = pd.read_csv(input_path)
    print(f"[preprocessor] Loaded {len(df)} rows, {len(df.columns)} columns.")
    print(f"[preprocessor] Columns: {list(df.columns)}")

    # ── Normalise column names ───────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Auto-detect key columns
    def _pick(candidates: list[str], fallback: str) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        return fallback

    id_col = facet_id_col or _pick(
        ["facet_id", "id", "facet_code", "code"], df.columns[0]
    )
    nm_col = name_col or _pick(
        ["facet_name", "name", "facet", "label"], df.columns[min(1, len(df.columns)-1)]
    )
    ds_col = desc_col or _pick(
        ["description", "desc", "definition", "detail"], "")

    # ── Basic cleaning ───────────────────────────────────────────────────────
    df = df.dropna(subset=[nm_col])
    df = df.drop_duplicates(subset=[nm_col])
    df[nm_col] = df[nm_col].str.strip()

    # Ensure facet_id exists and is a clean slug
    if id_col not in df.columns or id_col == nm_col:
        df["facet_id"] = (
            df[nm_col]
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
    else:
        df["facet_id"] = (
            df[id_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
    df["facet_name"] = df[nm_col]
    df["description"] = df[ds_col].fillna("") if ds_col and ds_col in df.columns else ""

    # ── Infer domain ─────────────────────────────────────────────────────────
    if "domain" not in df.columns:
        df["domain"] = df.apply(_infer_domain, axis=1)
        print("[preprocessor] Inferred domains:")
        print(df["domain"].value_counts().to_string())
    else:
        df["domain"] = df["domain"].str.lower().str.strip()

    # ── Rubrics ──────────────────────────────────────────────────────────────
    rubrics = df.apply(
        lambda r: _build_rubric(r["domain"], r["facet_name"]), axis=1
    )
    for score in range(1, 6):
        df[f"rubric_{score}"] = rubrics.apply(lambda r: r[score])

    # ── Additional enrichment columns ────────────────────────────────────────
    df["is_turn_level"] = True          # all facets scored per turn by default
    df["polarity"] = df.apply(_infer_polarity, axis=1)
    df["weight"] = 1.0
    df["requires_context"] = df.apply(_requires_context, axis=1)
    df["prompt_hint"] = df.apply(
        lambda r: _make_prompt_hint(r["facet_name"], r["domain"], r["description"]),
        axis=1,
    )

    # ── Chunk groups for batched inference ───────────────────────────────────
    df = df.reset_index(drop=True)
    df["chunk_group"] = df.index // CHUNK_SIZE

    # ── Reorder columns ──────────────────────────────────────────────────────
    front_cols = [
        "facet_id", "facet_name", "domain", "description",
        "rubric_1", "rubric_2", "rubric_3", "rubric_4", "rubric_5",
        "is_turn_level", "polarity", "weight", "requires_context",
        "prompt_hint", "chunk_group",
    ]
    remaining = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining]

    df.to_csv(output_path, index=False)
    print(f"[preprocessor] Saved {len(df)} facets → {output_path}")
    return df


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/facets.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/facets_processed.csv"
    preprocess(inp, out)
