"""Deterministic 512-D unit-normalized embedding fixtures."""

from __future__ import annotations

import math
from collections.abc import Sequence

DIMENSION = 512


def _unit_vector(first_one_at: int) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[first_one_at] = 1.0
    return vector


def vector_a() -> list[float]:
    return _unit_vector(0)


def vector_b() -> list[float]:
    return _unit_vector(1)


def assert_unit_norm(embedding: Sequence[float]) -> None:
    assert len(embedding) == DIMENSION
    assert all(math.isfinite(v) for v in embedding)
    norm = math.sqrt(sum(v * v for v in embedding))
    assert norm > 0.0
    assert abs(norm - 1.0) < 1e-9, f"Expected unit norm, got {norm}"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
