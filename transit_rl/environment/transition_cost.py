"""
Transition cost between the COMPLETE previous plan P_{t-1} and the COMPLETE
current plan P_t.

Definitions (all components lie in [0, 1]):

route distance  d(a, b) = Levenshtein(a, b) / max(|a|, |b|)
                (minimum over b and reversed(b) when routes are bidirectional)

route matching  previous and current routes are paired by minimum-total-cost
                bipartite matching on d (Hungarian algorithm; greedy fallback).
                Pairs with d > ROUTE_MATCH_MAX_DISTANCE are discarded, i.e.
                treated as one removed route plus one new route.

C_route = (sum_matched d + U_prev + U_curr) / (M + U_prev + U_curr)
          M = matched pairs, U_prev/U_curr = unmatched previous/current routes
          (each removed or added route counts as a full change of 1).

C_freq  = (sum_matched |f_t - f_{t-1}| + sum_unmatched f) / (sum f_{t-1} + sum f_t)

C_transition = C_route. C_freq is retained as a diagnostic only.

Alternative definitions can be plugged in by subclassing
``TransitionCostCalculator`` and overriding ``route_distance`` / ``compute``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..config import ExperimentConfig
from .service_plan import ServicePlan

try:  # optional dependency
    from scipy.optimize import linear_sum_assignment as _hungarian
except Exception:  # pragma: no cover - depends on environment
    _hungarian = None


@dataclass(frozen=True)
class TransitionResult:
    route_change: float
    frequency_change: float
    total: float  # route_change only; frequency_change is diagnostic
    matches: Tuple[Tuple[int, int, float], ...]  # (prev index, curr index, distance)
    unmatched_previous: Tuple[int, ...]
    unmatched_current: Tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "transition_route_change": self.route_change,
            "transition_frequency_change": self.frequency_change,
            "transition_cost": self.total,
            "matched_routes": [list(m) for m in self.matches],
            "removed_previous_routes": list(self.unmatched_previous),
            "new_current_routes": list(self.unmatched_current),
        }


ZERO_TRANSITION = TransitionResult(0.0, 0.0, 0.0, (), (), ())


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Classic sequence edit distance (insert / delete / substitute = 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class TransitionCostCalculator:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.method = cfg.route_matching_method if _hungarian is not None else "greedy"
        self._distance_cache: dict = {}  # exact memoisation of route_distance

    def route_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        cached = self._distance_cache.get((a, b))
        if cached is None:
            if len(self._distance_cache) > 500000:
                self._distance_cache.clear()
            cached = self._distance_cache[(a, b)] = self._route_distance(a, b)
        return cached

    def _route_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        longest = max(len(a), len(b))
        if longest == 0:
            return 0.0
        d = levenshtein(a, b)
        if self.cfg.consider_route_reversal:
            d = min(d, levenshtein(a, tuple(reversed(b))))
        return d / longest

    def _match(self, cost: List[List[float]]) -> List[Tuple[int, int]]:
        if not cost or not cost[0]:
            return []
        if self.method == "hungarian":
            rows, cols = _hungarian(cost)
            return list(zip(rows.tolist(), cols.tolist()))
        # Greedy: repeatedly take the globally cheapest remaining pair.
        pairs = sorted((c, i, j) for i, row in enumerate(cost) for j, c in enumerate(row))
        used_i, used_j, out = set(), set(), []
        for _, i, j in pairs:
            if i not in used_i and j not in used_j:
                used_i.add(i)
                used_j.add(j)
                out.append((i, j))
        return out

    def compute(self, previous: ServicePlan, current: ServicePlan) -> TransitionResult:
        n_p, n_c = len(previous), len(current)
        if n_p == 0 and n_c == 0:
            return ZERO_TRANSITION
        cost = [[self.route_distance(p.route, c.route) for c in current] for p in previous]
        matches = [(i, j, cost[i][j]) for i, j in self._match(cost)
                   if cost[i][j] <= self.cfg.route_match_max_distance]
        mi = {i for i, _, _ in matches}
        mj = {j for _, j, _ in matches}
        un_p = tuple(i for i in range(n_p) if i not in mi)
        un_c = tuple(j for j in range(n_c) if j not in mj)

        route_change = (sum(d for _, _, d in matches) + len(un_p) + len(un_c)) / (
            len(matches) + len(un_p) + len(un_c))

        freq_total = sum(r.frequency for r in previous) + sum(r.frequency for r in current)
        freq_diff = (sum(abs(current[j].frequency - previous[i].frequency) for i, j, _ in matches)
                     + sum(previous[i].frequency for i in un_p)
                     + sum(current[j].frequency for j in un_c))
        freq_change = freq_diff / freq_total if freq_total > 0 else 0.0

        return TransitionResult(route_change, freq_change, route_change, tuple(matches), un_p, un_c)
