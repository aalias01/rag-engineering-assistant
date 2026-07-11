"""
src/stats.py — Small-sample statistics helpers for evaluation reporting.

Wilson score intervals instead of normal-approximation intervals: our eval
sets are small (n = 31 retrieval queries, 33 holdout intents, ~60 factual
queries), and the Wilson interval stays honest near p = 0 or 1 where the
normal approximation collapses. Report the interval, not just the point
estimate — "31/33 = 93.9% (Wilson 95% CI 80.4–98.3%)" is the claim an
interviewer can't puncture.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054  # two-sided 95%


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (low, high) in [0, 1]. For n = 0 returns (0.0, 1.0) — no data,
    no claim.
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} outside [0, n={n}]")
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def format_proportion(successes: int, n: int) -> str:
    """'31/33 = 93.9% (95% CI 80.4–98.3%)' — the reporting house style."""
    if n == 0:
        return "0/0 (no data)"
    low, high = wilson_ci(successes, n)
    return (
        f"{successes}/{n} = {successes / n:.1%} "
        f"(95% CI {low:.1%}–{high:.1%})"
    )
