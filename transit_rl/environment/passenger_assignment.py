"""
Passenger assignment (transport simulation component, no RL logic).

The assignment ALWAYS starts from the original, unmodified OD matrix of the
hour and the complete current service plan.  Served/unserved passengers are
written to separate output arrays; the OD matrix itself is never altered, so
when a better route appears passengers are simply re-assigned on the next run.

Algorithm for every ordered OD pair (i, j) with demand d > 0:
    1. direct paths: routes serving both i and j in the travel direction.
       Direct routes whose in-vehicle time (IVT) is within
       DIRECT_PATH_TOLERANCE_MINUTES of the fastest one form a common-lines
       set: combined frequency F = sum f_k, wait = factor * 60 / F, demand
       split in proportion f_k / F.
    2. one-transfer paths: route k1 from i to transfer stop s, route k2 from s
       to j. Choose the best path with space on every leg.
    3. two-transfer paths (optional): k1: i->s1, k2: s1->s2, k3: s2->j.
    4. generalised time = IVT + waiting time(s) + transfer penalty per transfer.
    5. path category chosen by PATH_CHOICE_RULE (see config).
    6. board only up to the residual capacity of every traversed segment.
       Retry leftover demand on other feasible paths; otherwise it is unserved.

Hourly capacity per segment and direction is frequency * BUS_CAPACITY *
BUS_LOAD_FACTOR (50 passengers per departure with 40 seats and factor 1.25).
Seats are reusable after alighting. OD pairs are processed in deterministic
origin/destination order; this is a greedy feasible assignment, not a global
maximum-throughput optimisation. Fractional flows and expected waits are hourly
averages; no individual departure timetable is simulated.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ExperimentConfig
from ..data.network import TransitNetwork
from .service_plan import ServicePlan

Leg = Tuple[int, int, int, float]  # (route index, board stop, alight stop, passengers)


@dataclass(frozen=True)
class AssignmentResult:
    """Passenger-assignment outputs (kept separate from the OD matrix).

    Times are passenger-minutes unless stated otherwise.  Instances may be
    shared through the cache and must be treated as read-only.
    """

    total_demand: float
    served_passengers: float
    total_in_vehicle_time: float
    total_waiting_time: float
    total_transfer_penalty_time: float
    total_passenger_travel_time: float      # in-vehicle + waiting
    direct_passengers: float
    one_transfer_passengers: float
    two_transfer_passengers: float
    unserved_passengers: float
    overload_passengers: float              # sum over route-directions of peak load above capacity
    average_waiting_time: float             # minutes per served passenger
    average_travel_time: float              # minutes per served passenger (IVT + wait)
    assigned_demand: np.ndarray = field(repr=False)
    unserved_demand: np.ndarray = field(repr=False)
    direct_demand: np.ndarray = field(repr=False)
    transfer_demand: np.ndarray = field(repr=False)
    route_peak_load: Tuple[float, ...] = ()  # max directional link load per route

    @property
    def transfer_passengers(self) -> float:
        return self.one_transfer_passengers + self.two_transfer_passengers

    def metrics(self) -> Dict[str, float]:
        """Scalar metrics for logging."""
        return {
            "total_demand": self.total_demand,
            "served_passengers": self.served_passengers,
            "total_passenger_travel_time": self.total_passenger_travel_time,
            "total_in_vehicle_time": self.total_in_vehicle_time,
            "total_waiting_time": self.total_waiting_time,
            "total_transfer_penalty_time": self.total_transfer_penalty_time,
            "direct_passengers": self.direct_passengers,
            "one_transfer_passengers": self.one_transfer_passengers,
            "two_transfer_passengers": self.two_transfer_passengers,
            "transfer_passengers": self.transfer_passengers,
            "unserved_passengers": self.unserved_passengers,
            "overload_passengers": self.overload_passengers,
            "average_waiting_time": self.average_waiting_time,
            "average_travel_time": self.average_travel_time,
        }


class _RouteIndex:
    """Pre-computed lookup data for one route of the plan.

    ``ivt[(a, b)]`` = in-vehicle time from stop a to stop b along the route
    (only for pairs that can be travelled in that direction).
    """

    __slots__ = ("stops", "frequency", "pos", "ivt", "wait")

    def __init__(self, stops: Tuple[int, ...], frequency: float, table: Tuple[Dict[int, int], Dict],
                 waiting_factor: float) -> None:
        self.stops = stops
        self.frequency = frequency
        self.pos, self.ivt = table
        self.wait = waiting_factor * 60.0 / frequency


def _route_table(stops: Tuple[int, ...], network: TransitNetwork, bidirectional: bool
                 ) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
    """Stop positions and all stop-to-stop in-vehicle times of one route."""
    pos: Dict[int, int] = {}
    for idx, s in enumerate(stops):
        pos.setdefault(s, idx)  # first occurrence if repeats are allowed
    fwd, bwd = [0.0], [0.0]
    for a, b in zip(stops[:-1], stops[1:]):
        fwd.append(fwd[-1] + network.link_time(a, b))
        bwd.append(bwd[-1] + network.link_time(b, a))
    ivt: Dict[Tuple[int, int], float] = {}
    for a, pa in pos.items():
        for b, pb in pos.items():
            if pa < pb:
                ivt[(a, b)] = fwd[pb] - fwd[pa]
            elif pa > pb and bidirectional:
                ivt[(a, b)] = bwd[pa] - bwd[pb]
    return pos, ivt


class PassengerAssignment:
    """Assigns the hourly OD demand to a service plan."""

    def __init__(self, network: TransitNetwork, cfg: ExperimentConfig,
                 od_matrices: Sequence[np.ndarray]) -> None:
        self.network = network
        self.cfg = cfg
        self.od_matrices = od_matrices
        n = network.num_stops
        # Non-zero OD pairs per hour, read directly from the original matrices.
        self._pairs: List[List[Tuple[int, int, float]]] = []
        for od in od_matrices:
            pairs = []
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i == j and cfg.ignore_od_diagonal:
                        continue
                    d = float(od[i - 1, j - 1])
                    if d > 0:
                        pairs.append((i, j, d))
            self._pairs.append(pairs)
        self._cache: "OrderedDict[tuple, AssignmentResult]" = OrderedDict()
        self._route_tables: Dict[Tuple[int, ...], tuple] = {}  # exact per-route lookup cache
        self.cache_hits = 0
        self.cache_misses = 0

    # ---------------------------------------------------------------- public
    def assign(self, hour: int, plan: ServicePlan) -> AssignmentResult:
        """Assign OD_hour (1-based hour) to ``plan``.  Results are cached on
        (hour, routes, frequencies), which is exact because assignment is a
        deterministic function of exactly these inputs."""
        key = (hour, tuple((rs.route, rs.frequency) for rs in plan))
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return cached
        self.cache_misses += 1
        result = self._assign_uncached(hour, plan)
        if self.cfg.assignment_cache_size > 0:
            self._cache[key] = result
            if len(self._cache) > self.cfg.assignment_cache_size:
                self._cache.popitem(last=False)
        return result

    # --------------------------------------------------------------- helpers
    def _table(self, route: Tuple[int, ...]) -> tuple:
        t = self._route_tables.get(route)
        if t is None:
            if len(self._route_tables) > 100000:
                self._route_tables.clear()
            t = _route_table(route, self.network, self.cfg.routes_are_bidirectional)
            self._route_tables[route] = t
        return t

    def _choose_path(self, i: int, j: int, routes: List[_RouteIndex],
                     routes_at: Dict[int, List[int]], common: list,
                     available: Callable[[int, int, int], float]):
        """Choose a path with residual capacity on every leg.

        Recomputed after a bottleneck fills so leftover demand can use other
        direct lines or transfer paths. Ties follow route/stop order.
        """
        cfg, R = self.cfg, len(routes)
        penalty = cfg.transfer_penalty_minutes
        times = {}

        def travel_time(k: int, a: int, b: int) -> Optional[float]:
            key = (k, a, b)
            if key not in times:
                t = routes[k].ivt.get((a, b))
                times[key] = t if t is not None and available(k, a, b) > 1e-9 else None
            return times[key]

        def acceptable(gen: float) -> bool:
            return cfg.max_acceptable_trip_minutes is None or gen <= cfg.max_acceptable_trip_minutes

        options = []  # (generalised time, n_transfers, ivt, wait, legs_with_shares)

        # ---- 1. direct ------------------------------------------------
        cands = []
        for k in routes_at[i]:
            t = travel_time(k, i, j)
            if t is not None:
                cands.append((k, t))
        if cands:
            best = min(t for _, t in cands)
            group = [(k, t) for k, t in cands if t <= best + cfg.direct_path_tolerance_minutes]
            F = sum(routes[k].frequency for k, _ in group)
            wait = cfg.waiting_time_factor * 60.0 / F
            ivt = sum(routes[k].frequency * t for k, t in group) / F
            legs = [((k, i, j), routes[k].frequency / F) for k, _ in group]
            if acceptable(ivt + wait):
                options.append((ivt + wait, 0, ivt, wait, legs))

        need_more = cfg.path_choice_rule == "min_generalised_time" or not options
        # ---- 2. one transfer -------------------------------------------
        if need_more:
            best1 = None
            for k1 in routes_at[i]:
                r1 = routes[k1]
                for k2 in routes_at[j]:
                    if k1 == k2:
                        continue
                    r2 = routes[k2]
                    for s in common[k1][k2]:
                        if s == i or s == j:
                            continue
                        t1 = travel_time(k1, i, s)
                        t2 = travel_time(k2, s, j)
                        if t1 is None or t2 is None:
                            continue
                        gen = t1 + t2 + r1.wait + r2.wait + penalty
                        if acceptable(gen) and (best1 is None or gen < best1[0]):
                            best1 = (gen, 1, t1 + t2, r1.wait + r2.wait,
                                     [((k1, i, s), 1.0), ((k2, s, j), 1.0)])
            if best1 is not None:
                options.append(best1)
            need_more = cfg.path_choice_rule == "min_generalised_time" or not options

        # ---- 3. two transfers -----------------------------------------
        if need_more and cfg.enable_two_transfer_paths:
            best2 = None
            for k1 in routes_at[i]:
                r1 = routes[k1]
                for k3 in routes_at[j]:
                    if k3 == k1:
                        continue
                    r3 = routes[k3]
                    for k2 in range(R):
                        if k2 == k1 or k2 == k3:
                            continue
                        r2 = routes[k2]
                        for s1 in common[k1][k2]:
                            if s1 == i or s1 == j:
                                continue
                            t1 = travel_time(k1, i, s1)
                            if t1 is None:
                                continue
                            for s2 in common[k2][k3]:
                                if s2 == j or s2 == s1 or s2 == i:
                                    continue
                                t2 = travel_time(k2, s1, s2)
                                t3 = travel_time(k3, s2, j)
                                if t2 is None or t3 is None:
                                    continue
                                w = r1.wait + r2.wait + r3.wait
                                gen = t1 + t2 + t3 + w + 2 * penalty
                                if acceptable(gen) and (best2 is None or gen < best2[0]):
                                    best2 = (gen, 2, t1 + t2 + t3, w,
                                             [((k1, i, s1), 1.0), ((k2, s1, s2), 1.0), ((k3, s2, j), 1.0)])
            if best2 is not None:
                options.append(best2)

        # ---- choose ----------------------------------------------------
        chosen = None
        if options:
            chosen = options[0] if cfg.path_choice_rule == "min_transfers_first" else min(options, key=lambda o: o[0])
        return chosen

    def _assign_uncached(self, hour: int, plan: ServicePlan) -> AssignmentResult:
        cfg, n = self.cfg, self.network.num_stops
        routes = [_RouteIndex(rs.route, rs.frequency, self._table(rs.route), cfg.waiting_time_factor)
                  for rs in plan]
        R = len(routes)
        capacities = [r.frequency * cfg.bus_capacity * cfg.bus_load_factor for r in routes]
        routes_at: Dict[int, List[int]] = {s: [] for s in self.network.stops}
        for k, r in enumerate(routes):
            for s in r.pos:
                routes_at[s].append(k)
        common = [[tuple(sorted(set(routes[a].pos) & set(routes[b].pos))) if a != b else () for b in range(R)]
                  for a in range(R)]
        penalty = cfg.transfer_penalty_minutes

        assigned = np.zeros((n, n))
        unserved = np.zeros((n, n))
        direct_m = np.zeros((n, n))
        transfer_m = np.zeros((n, n))
        fwd_load = [np.zeros(max(len(r.stops) - 1, 0)) for r in routes]
        bwd_load = [np.zeros(max(len(r.stops) - 1, 0)) for r in routes]

        def leg_load(k: int, a: int, b: int) -> np.ndarray:
            pa, pb = routes[k].pos[a], routes[k].pos[b]
            return fwd_load[k][pa:pb] if pa < pb else bwd_load[k][pb:pa]

        def available(k: int, a: int, b: int) -> float:
            return max(0.0, capacities[k] - float(leg_load(k, a, b).max()))

        tot_ivt = tot_wait = tot_pen = 0.0
        n_direct = n_one = n_two = n_unserved = 0.0
        total_demand = 0.0

        for i, j, d in self._pairs[hour - 1]:
            total_demand += d
            remaining = d
            while remaining > 0:
                chosen = self._choose_path(i, j, routes, routes_at, common, available)
                if chosen is None:
                    break
                legs = chosen[4]
                amount = min(remaining, min(available(k, a, b) / share
                                           for (k, a, b), share in legs))
                _, n_tr, ivt, wait, legs = chosen
                tot_ivt += amount * ivt
                tot_wait += amount * wait
                tot_pen += amount * n_tr * penalty
                assigned[i - 1, j - 1] += amount
                if n_tr == 0:
                    n_direct += amount
                    direct_m[i - 1, j - 1] += amount
                else:
                    transfer_m[i - 1, j - 1] += amount
                    if n_tr == 1:
                        n_one += amount
                    else:
                        n_two += amount
                for (k, a, b), share in legs:
                    leg_load(k, a, b)[:] += amount * share
                remaining = max(0.0, remaining - amount)
            n_unserved += remaining
            unserved[i - 1, j - 1] = remaining

        # ---- actual boarded loads (overload should be zero within rounding) -
        overload = 0.0
        peaks = []
        for k, r in enumerate(routes):
            capacity = capacities[k]
            pf = float(fwd_load[k].max()) if fwd_load[k].size else 0.0
            pb = float(bwd_load[k].max()) if bwd_load[k].size else 0.0
            overload += max(0.0, pf - capacity) + max(0.0, pb - capacity)
            peaks.append(max(pf, pb))

        served = total_demand - n_unserved
        for arr in (assigned, unserved, direct_m, transfer_m):
            arr.setflags(write=False)
        return AssignmentResult(
            total_demand=total_demand,
            served_passengers=served,
            total_in_vehicle_time=tot_ivt,
            total_waiting_time=tot_wait,
            total_transfer_penalty_time=tot_pen,
            total_passenger_travel_time=tot_ivt + tot_wait,
            direct_passengers=n_direct,
            one_transfer_passengers=n_one,
            two_transfer_passengers=n_two,
            unserved_passengers=n_unserved,
            overload_passengers=overload,
            average_waiting_time=tot_wait / served if served > 0 else 0.0,
            average_travel_time=(tot_ivt + tot_wait) / served if served > 0 else 0.0,
            assigned_demand=assigned,
            unserved_demand=unserved,
            direct_demand=direct_m,
            transfer_demand=transfer_m,
            route_peak_load=tuple(peaks),
        )
