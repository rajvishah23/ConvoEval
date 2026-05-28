"""
src/pipeline/turn_splitter.py
──────────────────────────────
Splits raw conversation dicts into Turn objects ready for scoring.

Input format (flexible):
  [{"role": "user"|"assistant"|"system", "content": "..."}]

Handles:
  - System prompts (stripped or kept as context)
  - Multi-turn context windows (for facets that require_context)
  - Metadata propagation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json


@dataclass
class Turn:
    index: int                          # 0-based position in conversation
    role: str                           # user | assistant | system
    content: str
    context_window: list[dict]          # preceding turns (for context-aware facets)
    conversation_id: str
    turn_id: str                        # stable hash for caching
    metadata: dict = field(default_factory=dict)

    def to_prompt_block(self, include_context: bool = True) -> str:
        """Render turn + optional context as a text block for the LLM prompt."""
        lines = []
        if include_context and self.context_window:
            lines.append("=== CONVERSATION CONTEXT ===")
            for msg in self.context_window:
                lines.append(f"[{msg['role'].upper()}]: {msg['content']}")
            lines.append("")
        lines.append("=== TURN TO EVALUATE ===")
        lines.append(f"[{self.role.upper()}]: {self.content}")
        return "\n".join(lines)


def split_conversation(
    messages: list[dict],
    conversation_id: Optional[str] = None,
    score_user_turns: bool = False,
    context_window_size: int = 4,
) -> list[Turn]:
    """
    Convert a list of messages into Turn objects.

    Parameters
    ----------
    messages            : raw message list
    conversation_id     : optional stable ID (auto-generated if None)
    score_user_turns    : if True, also create Turn objects for user messages
    context_window_size : how many preceding messages to include as context

    Returns
    -------
    List of Turn objects (default: assistant turns only)
    """
    if not conversation_id:
        payload = json.dumps(messages, sort_keys=True).encode()
        conversation_id = hashlib.md5(payload).hexdigest()[:12]

    turns: list[Turn] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "user").lower()

        # Skip system messages as scoreable turns
        if role == "system":
            continue

        # By default only score assistant turns
        if not score_user_turns and role != "assistant":
            continue

        # Build context window (preceding non-system messages)
        context = [
            m for m in messages[max(0, i - context_window_size): i]
            if m.get("role", "").lower() != "system"
        ]

        content = msg.get("content", "")
        turn_hash = hashlib.md5(
            f"{conversation_id}:{i}:{content}".encode()
        ).hexdigest()[:16]

        turns.append(Turn(
            index=i,
            role=role,
            content=content,
            context_window=context,
            conversation_id=conversation_id,
            turn_id=turn_hash,
            metadata=msg.get("metadata", {}),
        ))

    return turns


def load_conversation_file(path: str) -> dict:
    """Load a conversation JSON file. Returns dict with 'id' and 'messages'."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    # Support both {"messages": [...]} and bare list formats
    if isinstance(data, list):
        return {"id": None, "messages": data}
    return data
