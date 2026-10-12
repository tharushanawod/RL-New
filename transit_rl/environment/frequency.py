"""
Frequency and fleet calculation (independent of RL logic).

For route k:   v_k = f_k * T_k / 60
    f_k : frequency (buses/hour)
    T_k : round-trip time (minutes) = forward time + backward time + 2*layover

ROUNDING RULE: v_k = ceil(f_k * T_k / 60).  A fractional bus cannot run, so the
requirement is always rounded up (with a tiny tolerance for float noise).
Fleet constraint: sum_k v_k <= MAX_FLEET.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import ExperimentConfig
from ..data.network import TransitNetwork
from .service_plan import ServicePlan


class FleetCalculator:
    def __init__(self, network: TransitNetwork, cfg: ExperimentConfig) -> None:
        self.network = network
        self.cfg = cfg
        self._round_trip_cache: Dict[Tuple[int, ...], float] = {}

    # ------------------------------------------------------------- route times
    def one_way_time(self, route: Sequence[int]) -> float:
        """Forward in-vehicle time from first to last stop."""
        return sum(self.network.link_time(a, b) for a, b in zip(route[:-1], route[1:]))

    def reverse_time(self, route: Sequence[int]) -> float:
        """Backward in-vehicle time from last to first stop."""
        return sum(self.network.link_time(b, a) for a, b in zip(route[:-1], route[1:]))

    def round_trip_time(self, route: Tuple[int, ...]) -> float:
        cached = self._round_trip_cache.get(route)
        if cached is None:
            # Bidirectional routes serve both directions.  ASSUMPTION for
            # one-directional routes: the bus deadheads back along the reverse
            # path, so the cycle time is the same formula.
            cached = (self.one_way_time(route) + self.reverse_time(route)
                      + 2.0 * self.cfg.terminal_layover_minutes)
            self._round_trip_cache[route] = cached
        return cached

    # ------------------------------------------------------------------ buses
    def buses_required(self, route: Tuple[int, ...], frequency: float) -> int:
        exact = frequency * self.round_trip_time(route) / 60.0
        return int(math.ceil(exact - self.cfg.fleet_rounding_tolerance))

    def buses_per_route(self, plan: ServicePlan) -> List[int]:
        return [self.buses_required(rs.route, rs.frequency) for rs in plan]

    def fleet_used(self, plan: ServicePlan) -> int:
        return sum(self.buses_per_route(plan))

    def fleet_remaining(self, plan: ServicePlan) -> int:
        """F_remaining = F_max - sum_k v_k (buses released by a route return here)."""
        return self.cfg.max_fleet - self.fleet_used(plan)

    # -------------------------------------------------------------- frequency
    def frequency_grid(self) -> List[float]:
        grid, f = [], self.cfg.min_frequency
        while f <= self.cfg.max_frequency + 1e-9:
            grid.append(round(f, 10))
            f += self.cfg.frequency_step
        return grid

    def initial_frequency_for_new_route(self, route: Tuple[int, ...], fleet_remaining: int) -> Optional[float]:
        """Frequency assigned by END_ROUTE, or None if no feasible frequency."""
        f0 = self.cfg.initial_route_frequency
        if self.buses_required(route, f0) <= fleet_remaining:
            return f0
        if not self.cfg.end_route_frequency_fallback:
            return None
        feasible = [f for f in self.frequency_grid()
                    if f <= f0 and self.buses_required(route, f) <= fleet_remaining]
        return max(feasible) if feasible else None
