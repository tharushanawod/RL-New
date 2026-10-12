"""
Service-plan data structures.

A service plan is an immutable tuple of ``RouteService`` records.  Because all
parts are tuples, plans are hashable and can be embedded in Q-table state keys
without any conversion, and copying a plan can never alias mutable data.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple, Tuple


class RouteService(NamedTuple):
    """One route of a service plan.

    route:     ordered stop sequence, e.g. (1, 3, 6, 8)
    frequency: buses/hour
    kept:      intra-hour decision marker set by KEEP_ROUTE.  A kept route is
               locked for the rest of the hour (no further frequency changes,
               keep or removal).  It is not a service attribute and is reset
               to False when the plan becomes the previous plan P_{t-1}.
    """

    route: Tuple[int, ...]
    frequency: float
    kept: bool = False


ServicePlan = Tuple[RouteService, ...]
EMPTY_PLAN: ServicePlan = ()


def make_plan(records: Iterable[Dict[str, Any]]) -> ServicePlan:
    """Build a plan from ``[{"route": [...], "frequency": f}, ...]``."""
    return tuple(
        RouteService(tuple(int(s) for s in r["route"]), float(r["frequency"]), bool(r.get("kept", False)))
        for r in records
    )


def reset_kept_flags(plan: ServicePlan) -> ServicePlan:
    return tuple(rs._replace(kept=False) for rs in plan)


def plan_to_records(plan: ServicePlan, include_kept: bool = True) -> List[Dict[str, Any]]:
    """Complete JSON-friendly representation (lossless)."""
    out = []
    for rs in plan:
        rec: Dict[str, Any] = {"route": list(rs.route), "frequency": rs.frequency}
        if include_kept:
            rec["kept"] = rs.kept
        out.append(rec)
    return out


def routes_equivalent(a: Tuple[int, ...], b: Tuple[int, ...], bidirectional: bool) -> bool:
    return a == b or (bidirectional and a == tuple(reversed(b)))
