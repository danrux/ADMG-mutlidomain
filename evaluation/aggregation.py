"""Utilities for storing heterogeneous metric dictionaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def flatten_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    """Flatten nested evaluator output without imposing one metric schema."""
    flat: dict[str, float | int] = {}
    for name, value in metrics.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, prefix=key))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flat[key] = value.item() if isinstance(value, np.generic) else value
    return flat


def summarize_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Compute means for numeric metric keys shared by every run."""
    if not rows:
        return {}
    flattened = [flatten_metrics(row) for row in rows]
    shared = set.intersection(*(set(row) for row in flattened))
    return {
        f"mean.{key}": float(np.nanmean([row[key] for row in flattened]))
        for key in sorted(shared)
    }
