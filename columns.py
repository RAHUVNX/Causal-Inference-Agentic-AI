"""Lightweight column configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_FEATURE_COLUMN = "features"


@dataclass(frozen=True)
class Column:
    """Minimal column descriptor used across the repository."""

    name: str
    dtype: Optional[str] = None
