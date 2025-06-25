"""Utility functions to measure balance between treatment and control groups.

These metrics are deliberately lightweight so that they can be used in greedy
search loops.  All functions operate on sequences of ``GeoUnit`` objects and
return **lower-is-better** scores (0 = perfect balance).
"""
from __future__ import annotations

from typing import List, Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Basic aggregate helpers
# ---------------------------------------------------------------------------

def _aggregate(geos: List[GeoUnit]) -> Tuple[float, float, Dict[str, float]]:
    """Return (response_sum, spend_sum, covariate_sums).

    If some GeoUnits lack a covariate key it will be treated as zero.
    """
    resp_sum = sum(g.response for g in geos)
    spend_sum = sum(g.spend for g in geos)
    cov_sums: Dict[str, float] = {}
    for g in geos:
        if g.covariates:
            for k, v in g.covariates.items():
                cov_sums[k] = cov_sums.get(k, 0.0) + v
    return resp_sum, spend_sum, cov_sums

# ---------------------------------------------------------------------------
# Balance metrics
# ---------------------------------------------------------------------------

def absolute_difference(treatment: List[GeoUnit], control: List[GeoUnit]) -> float:
    """Absolute difference in *response* sums between treatment & control."""
    resp_t, _, _ = _aggregate(treatment)
    resp_c, _, _ = _aggregate(control)
    return abs(resp_t - resp_c)


def relative_difference(treatment: List[GeoUnit], control: List[GeoUnit]) -> float:
    """Absolute diff divided by max(|treatment|, 1e-9)."""
    resp_t, _, _ = _aggregate(treatment)
    resp_c, _, _ = _aggregate(control)
    return abs(resp_t - resp_c) / max(abs(resp_t), 1e-9)


def covariate_diffs(treatment: List[GeoUnit], control: List[GeoUnit]) -> Dict[str, float]:
    """Return absolute differences for each covariate key."""
    _, _, cov_t = _aggregate(treatment)
    _, _, cov_c = _aggregate(control)
    keys = set(cov_t) | set(cov_c)
    return {k: abs(cov_t.get(k, 0.0) - cov_c.get(k, 0.0)) for k in keys}

__all__ = [
    "absolute_difference",
    "relative_difference",
    "covariate_diffs",
]
