"""
State-space explosion diagnostics.

* visit statistics of the sparse Q-table
* a rough order-of-magnitude estimate of the theoretical state space.  Routes
  are only COUNTED here (depth-first search) to report the size of the space;
  they are never stored or used to pre-populate the Q-table.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from ..agent.q_learning_agent import QTable, VisitTable
from ..config import ExperimentConfig
from ..data.network import TransitNetwork


def visit_statistics(q: QTable, visits: VisitTable) -> Dict[str, Any]:
    pair_counts: List[int] = []
    state_counts: List[int] = []
    for key, row in visits.items():
        counts = list(row.values())
        pair_counts.extend(counts)
        state_counts.append(sum(counts))
    n_states = len(q)
    n_pairs = len(pair_counts)
    return {
        "unique_states_encountered": n_states,
        "state_action_pairs_visited": n_pairs,
        "states_visited_once": sum(1 for c in state_counts if c == 1),
        "states_visited_more_than_once": sum(1 for c in state_counts if c > 1),
        "state_action_pairs_visited_once": sum(1 for c in pair_counts if c == 1),
        "state_action_pairs_visited_more_than_once": sum(1 for c in pair_counts if c > 1),
        "max_visit_count": max(pair_counts) if pair_counts else 0,
        "average_visit_count": (sum(pair_counts) / n_pairs) if n_pairs else 0.0,
        "max_state_visits": max(state_counts) if state_counts else 0,
        "average_state_visits": (sum(state_counts) / n_states) if n_states else 0.0,
        "average_actions_tried_per_state": (n_pairs / n_states) if n_states else 0.0,
        "fraction_states_revisited": (sum(1 for c in state_counts if c > 1) / n_states) if n_states else 0.0,
        "total_updates": sum(pair_counts),
    }


def count_possible_routes(network: TransitNetwork, cfg: ExperimentConfig) -> int:
    """Number of directed simple paths with MIN..MAX_ROUTE_LENGTH stops."""
    if cfg.allow_repeated_stops:
        # Walks: dynamic programming on path length.
        ways = {s: 1 for s in network.stops}
        total = 0
        for length in range(2, cfg.max_route_length + 1):
            ways = {v: sum(ways[u] for u in network.stops if network.has_link(u, v)) for v in network.stops}
            if length >= cfg.min_route_length:
                total += sum(ways.values())
        return total

    total = 0

    def dfs(path: List[int], visited: set) -> None:
        nonlocal total
        if len(path) >= cfg.min_route_length:
            total += 1
        if len(path) == cfg.max_route_length:
            return
        for j in network.neighbours[path[-1] - 1]:
            if j not in visited:
                visited.add(j)
                path.append(j)
                dfs(path, visited)
                path.pop()
                visited.remove(j)

    for s in network.stops:
        dfs([s], {s})
    return total


def _log10_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(10)


def state_space_estimate(network: TransitNetwork, cfg: ExperimentConfig) -> Dict[str, Any]:
    """Rough, fleet-unconstrained order-of-magnitude estimate.

    completed plans per hour  ~ sum_{k=0..MAX_ROUTES} C(routes, k) * n_freq^k
    states (completed plans only, ignoring partial routes and kept flags)
                              ~ hours * plans(current) * plans(previous)
    The fleet limit removes some plans, partial routes / kept flags add many
    more, so this is an indication of scale, not an exact count.
    """
    n_routes = count_possible_routes(network, cfg)
    if cfg.routes_are_bidirectional:
        n_routes_services = n_routes // 2  # a route and its reverse are one service
    else:
        n_routes_services = n_routes
    n_freq = int(round((cfg.max_frequency - cfg.min_frequency) / cfg.frequency_step)) + 1
    terms = [_log10_comb(n_routes_services, k) + k * math.log10(n_freq)
             for k in range(0, min(cfg.max_routes, n_routes_services) + 1)]
    m = max(terms)
    log_plans = m + math.log10(sum(10 ** (t - m) for t in terms))
    return {
        "possible_directed_routes": n_routes,
        "possible_route_services": n_routes_services,
        "frequency_levels": n_freq,
        "log10_possible_plans_per_hour": log_plans,
        "log10_possible_states_rough": math.log10(cfg.num_hours) + 2 * log_plans,
    }


def process_memory_mb() -> float:
    """Resident memory of this process in MB (NaN if it cannot be measured)."""
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        pass
    try:
        import resource  # Unix only
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")
