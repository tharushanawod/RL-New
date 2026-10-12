"""
Raw state representation and LOSSLESS conversion helpers.

    S_t = (t, OD_t, R_t_current, P_t_current, P_{t-1}, F_t_remaining)

    hour             1..24
    od_matrix        complete raw 15x15 OD matrix (tuple of tuples)
    current_route    complete stop sequence under construction, e.g. (1, 3, 6)
    current_plan     complete current-hour plan: RouteService(route, frequency, kept)
    previous_plan    complete previous-hour plan P_{t-1}: RouteService(route, frequency)
    fleet_remaining  F_max - sum_k v_k

All parts are tuples, so the state itself is hashable.  Nothing is
compressed, aggregated, discretised or hashed to a shorter code.
"""

from __future__ import annotations

import json
from typing import Any, Dict, NamedTuple, Tuple

from .service_plan import RouteService, ServicePlan, plan_to_records

StateKey = Tuple[Any, ...]


class EnvironmentState(NamedTuple):
    hour: int
    od_matrix: Tuple[Tuple[Any, ...], ...]
    current_route: Tuple[int, ...]
    current_plan: ServicePlan
    previous_plan: ServicePlan
    fleet_remaining: int


def state_to_key(state: EnvironmentState) -> StateKey:
    """Return a hashable Q-table key for ``state``.

    This conversion exists only to make the raw state hashable.
    No state information is removed.

    The key is a plain tuple containing every element of the state.  The plan
    elements are kept as RouteService named tuples, which compare and hash
    exactly like plain (route, frequency, kept) tuples, so keys rebuilt from
    CSV are equal to training keys.  The OD matrix, previous plan and
    unchanged route records are shared by reference between keys (Python
    stores a reference, not a copy), which saves memory without losing
    information.
    """
    return (
        state.hour,
        state.od_matrix,
        state.current_route,
        state.current_plan,
        state.previous_plan,
        state.fleet_remaining,
    )


def key_to_state(key: StateKey) -> EnvironmentState:
    """Exact inverse of ``state_to_key``."""
    hour, od, route, plan, prev, fleet = key
    return EnvironmentState(hour, od, route, tuple(RouteService(*r) for r in plan),
                            tuple(RouteService(*r) for r in prev), fleet)


# ----------------------------------------------------------------- JSON I/O
# JSON is used only as a storage format for CSV cells.  Python's json module
# writes floats with repr(), which round-trips exactly, and ints stay ints.

def key_to_json_fields(key: StateKey) -> Dict[str, Any]:
    hour, od, route, plan, prev, fleet = key
    return {
        "hour": hour,
        "od_matrix": json.dumps(od, separators=(",", ":")),
        "current_route": json.dumps(list(route)),
        "current_service_plan": json.dumps(plan_to_records(tuple(RouteService(*r) for r in plan), True)),
        "previous_service_plan": json.dumps(plan_to_records(tuple(RouteService(*r) for r in prev), False)),
        "fleet_remaining": fleet,
    }


def _plan_key_from_json(text: str) -> Tuple[Tuple[Any, ...], ...]:
    return tuple((tuple(r["route"]), r["frequency"], bool(r.get("kept", False))) for r in json.loads(text))


def _tupleize(obj: Any) -> Any:
    return tuple(_tupleize(x) for x in obj) if isinstance(obj, list) else obj


def json_fields_to_key(fields: Dict[str, str], od_cache: Dict[str, Any] | None = None) -> StateKey:
    """Rebuild the exact state key from CSV/JSON fields."""
    od_text = fields["od_matrix"]
    if od_cache is not None and od_text in od_cache:
        od = od_cache[od_text]
    else:
        od = _tupleize(json.loads(od_text))
        if od_cache is not None:
            od_cache[od_text] = od
    return (
        int(fields["hour"]),
        od,
        tuple(json.loads(fields["current_route"])),
        _plan_key_from_json(fields["current_service_plan"]),
        _plan_key_from_json(fields["previous_service_plan"]),
        int(fields["fleet_remaining"]),
    )
