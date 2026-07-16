"""Rational PTS / timestamp utilities."""

from __future__ import annotations

from fractions import Fraction


def pts_to_ns(pts: int, time_base_num: int, time_base_den: int) -> int:
    """Convert integer PTS to integer nanoseconds using exact rationals.

    pts_ns = round(Fraction(pts) * Fraction(time_base_num, time_base_den) * 1_000_000_000)
    """
    if time_base_den == 0:
        raise ValueError("time_base_den must be positive")
    return int(round(Fraction(pts) * Fraction(time_base_num, time_base_den) * 1_000_000_000))


def ns_to_pts(ns: int, time_base_num: int, time_base_den: int) -> int:
    """Convert integer nanoseconds back to the nearest integer PTS."""
    if time_base_num == 0:
        raise ValueError("time_base_num must be non-zero")
    return int(round(Fraction(ns) * Fraction(time_base_den, time_base_num) / 1_000_000_000))


def time_base_to_fraction(time_base_num: int, time_base_den: int) -> Fraction:
    """Return the time base as a Fraction."""
    return Fraction(time_base_num, time_base_den)
