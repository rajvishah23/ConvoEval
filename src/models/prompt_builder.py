"""
src/models/prompt_builder.py
──────────────────────────────
Builds multi-turn (chain-of-thought) prompts for facet scoring.

Flow per facet chunk:
  Turn 0 [system]   : Role definition + scoring protocol
  Turn 1 [user]     : Present the conversation turn
  Turn 2 [assistant]: CoT reasoning (model fills in)
  Turn 3 [user]     : "Now assign scores for facets X, Y, Z..."
  Turn 4 [assistant]: JSON scores (model fills in)

This satisfies the "No one-shot prompt" hard constraint.
"""

from __future__ import annotations

import json
from typing import Optional

from src.pipeline.facet_registry import Facet
from src.pipeline.turn_splitter import Turn

SYSTEM_PROMPT = """You are an expert conversation quality evaluator with deep expertise in linguistics, pragmatics, safety analysis, and affective computing.

Your task is to evaluate a single conversation turn on specific quality facets. You MUST follow this two-step process:

STEP 1 — ANALYSIS: Reason carefully about the turn. Consider the full context provided.
STEP 2 — SCORING: Assign a score (1–5) for each facet with a one-sentence rationale.

Scoring scale:
  1 = Poor       (fails the facet entirely)
  2 = Below Avg  (partial failure, noticeable issues)
  3 = Average    (meets minimum expectations)
  4 = Good       (clearly above average)
  5 = Excellent  (exemplary, near-perfect)

Always output valid JSON in your final response. Never refuse to score."""


def build_analysis_prompt(turn: Turn) -> str:
    """
    Turn 1 (user): Present the conversation turn for analysis.
    The model's Turn 2 response will be CoT reasoning.
    """
    context_block = turn.to_prompt_block(include_context=True)
    return f"""Please analyse the following conversation turn carefully before scoring.

{context_block}

Think step by step:
1. What is the role of this turn in the conversation?
2. Note any linguistic strengths or issues
3. Assess pragmatic alignment with the context
4. Flag any safety concerns
5. Identify the emotional tone

Provide your analysis in 3–5 sentences."""


def build_scoring_prompt(facets: list[Facet], analysis_context: str) -> str:
    """
    Turn 3 (user): Ask for scores after CoT analysis.
    The model's Turn 4 response will be JSON scores.
    """
    facet_lines = []
    for f in facets:
        rubric = f.rubric_text()
        facet_lines.append(
            f"### {f.facet_id}\n"
            f"Name: {f.facet_name}\n"
            f"Domain: {f.domain}\n"
            f"Instruction: {f.prompt_hint}\n"
            f"Rubric:\n{rubric}"
        )

    facets_block = "\n\n".join(facet_lines)
    facet_ids = [f.facet_id for f in facets]

    return f"""Based on your analysis above, now score the turn on these {len(facets)} facets:

{facets_block}

Respond with ONLY valid JSON in this exact format (no extra text):
{{
  "scores": {{
    {', '.join(f'"{fid}": {{"score": <1-5>, "rationale": "<one sentence>"}}' for fid in facet_ids[:3])}
    ... (one entry per facet)
  }}
}}

Important:
- Every facet_id listed above must appear in your response
- score must be an integer 1, 2, 3, 4, or 5
- rationale must be a single concise sentence"""


def build_confidence_prompt(scores: dict, facet_ids: list[str]) -> str:
    """
    Optional Turn 5/6: Self-critique for confidence estimation.
    """
    scores_str = json.dumps(
        {fid: scores[fid]["score"] for fid in facet_ids if fid in scores},
        indent=2,
    )
    return f"""You previously assigned these scores:
{scores_str}

For each score, how confident are you on a scale 0.0–1.0?
Consider: Is the rubric clearly applicable? Is there ambiguity? Would another evaluator likely agree?

Respond with ONLY valid JSON:
{{
  "confidence": {{
    {', '.join(f'"{fid}": <0.0-1.0>' for fid in facet_ids[:3])}
    ...
  }}
}}"""


def build_chat_messages(
    turn: Turn,
    facets: list[Facet],
    prior_analysis: Optional[str] = None,
) -> list[dict]:
    """
    Build the full multi-turn chat message list for one chunk of facets.

    Returns list of {role, content} dicts for OllamaClient.chat()
    """
    messages = []

    # Turn 1: Present the turn for analysis
    messages.append({
        "role": "user",
        "content": build_analysis_prompt(turn),
    })

    # Turn 2: Analysis (pre-filled if we have it, else model generates)
    if prior_analysis:
        messages.append({
            "role": "assistant",
            "content": prior_analysis,
        })
        # Turn 3: Scoring request
        messages.append({
            "role": "user",
            "content": build_scoring_prompt(facets, prior_analysis),
        })
    # If no prior_analysis: we'll do two separate API calls (analysis, then scoring)

    return messages
