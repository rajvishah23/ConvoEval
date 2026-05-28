"""
src/pipeline/facet_registry.py
────────────────────────────────
In-memory registry for all facets. Supports 5000+ facets with:
  - O(1) lookup by facet_id
  - Domain/chunk group filtering
  - Hot-reload from CSV without service restart
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd


@dataclass
class Facet:
    facet_id: str
    facet_name: str
    domain: str
    description: str
    rubric: dict[int, str]          # {1: "...", 2: "...", ..., 5: "..."}
    is_turn_level: bool
    polarity: str                   # higher_is_better | lower_is_better | neutral
    weight: float
    requires_context: bool
    prompt_hint: str
    chunk_group: int

    def rubric_text(self) -> str:
        lines = [f"  {k}: {v}" for k, v in sorted(self.rubric.items())]
        return "\n".join(lines)


class FacetRegistry:
    """
    Thread-safe registry. Singleton per process via `get_registry()`.
    Supports ≥5000 facets — just load a bigger CSV.
    """

    def __init__(self) -> None:
        self._facets: dict[str, Facet] = {}
        self._lock = threading.RLock()
        self._source_path: Optional[Path] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_csv(self, path: str | Path) -> int:
        """Load processed facets CSV. Returns count loaded."""
        path = Path(path)
        df = pd.read_csv(path)
        self._source_path = path

        rubric_cols = {
            int(c.split("_")[1]): c
            for c in df.columns
            if c.startswith("rubric_") and c.split("_")[1].isdigit()
        }

        new_facets: dict[str, Facet] = {}
        for _, row in df.iterrows():
            rubric = {
                score: str(row.get(col, ""))
                for score, col in rubric_cols.items()
            }
            f = Facet(
                facet_id=str(row["facet_id"]),
                facet_name=str(row["facet_name"]),
                domain=str(row.get("domain", "linguistic")),
                description=str(row.get("description", "")),
                rubric=rubric,
                is_turn_level=bool(row.get("is_turn_level", True)),
                polarity=str(row.get("polarity", "higher_is_better")),
                weight=float(row.get("weight", 1.0)),
                requires_context=bool(row.get("requires_context", False)),
                prompt_hint=str(row.get("prompt_hint", "")),
                chunk_group=int(row.get("chunk_group", 0)),
            )
            new_facets[f.facet_id] = f

        with self._lock:
            self._facets = new_facets

        print(f"[FacetRegistry] Loaded {len(self._facets)} facets from {path}")
        return len(self._facets)

    def reload(self) -> int:
        """Hot-reload from the same source path."""
        if self._source_path is None:
            raise RuntimeError("No source path set. Call load_csv() first.")
        return self.load_csv(self._source_path)

    # ── Querying ──────────────────────────────────────────────────────────────

    def get(self, facet_id: str) -> Optional[Facet]:
        with self._lock:
            return self._facets.get(facet_id)

    def get_many(self, facet_ids: list[str]) -> list[Facet]:
        with self._lock:
            return [self._facets[fid] for fid in facet_ids if fid in self._facets]

    def all(self) -> list[Facet]:
        with self._lock:
            return list(self._facets.values())

    def by_domain(self, domain: str) -> list[Facet]:
        with self._lock:
            return [f for f in self._facets.values() if f.domain == domain]

    def by_chunk_group(self, group: int) -> list[Facet]:
        with self._lock:
            return [f for f in self._facets.values() if f.chunk_group == group]

    def chunk_groups(self) -> list[int]:
        with self._lock:
            return sorted(set(f.chunk_group for f in self._facets.values()))

    def domains(self) -> list[str]:
        with self._lock:
            return sorted(set(f.domain for f in self._facets.values()))

    def __len__(self) -> int:
        return len(self._facets)

    def summary(self) -> dict:
        with self._lock:
            domain_counts = {}
            for f in self._facets.values():
                domain_counts[f.domain] = domain_counts.get(f.domain, 0) + 1
            return {
                "total_facets": len(self._facets),
                "domains": domain_counts,
                "chunk_groups": len(self.chunk_groups()),
            }


# ── Module-level singleton ────────────────────────────────────────────────────

_registry: Optional[FacetRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> FacetRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = FacetRegistry()
    return _registry
