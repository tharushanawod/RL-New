"""
Discrete action space and dynamic action masking.

An action is a small immutable tuple ``Action(type, arg)``:

    ADD_STOP(j)               j = stop id (1..15)
    REMOVE_LAST_STOP
    END_ROUTE
    INCREASE_FREQUENCY(k)     k = 0-based index of the route in the current plan
    DECREASE_FREQUENCY(k)
    KEEP_ROUTE(k)
    REMOVE_ROUTE(k)
    FINISH_PLAN

Route indices refer to positions in the current plan, which is part of the
state, so an action is always unambiguous given its state.

Masking rules (``ActionMasker.valid_actions``):

  While a route is under construction (current_route non-empty) only route
  construction actions are offered:
    ADD_STOP(j)       j is a physical neighbour of the terminal stop, j not
                      already in the route (unless repeats allowed) and the
                      route is shorter than MAX_ROUTE_LENGTH
    END_ROUTE         route has >= MIN_ROUTE_LENGTH stops, is not a duplicate
                      of a planned route and a feasible frequency fits the
                      remaining fleet
    REMOVE_LAST_STOP  route has > MIN_ROUTE_LENGTH stops.
                      ASSUMPTION (dead-end fallback): if no other construction
                      action is valid, REMOVE_LAST_STOP is offered regardless
                      of length, so the agent can never get stuck.

  When no route is under construction:
    ADD_STOP(j)       starts a new route at any stop j, if fewer than
                      MAX_ROUTES routes exist and at least one bus is free
    INCREASE_FREQUENCY(k)  not kept, f+step <= MAX_FREQUENCY, extra buses fit
    DECREASE_FREQUENCY(k)  not kept, f-step >= MIN_FREQUENCY
    KEEP_ROUTE(k)     route not already kept
    REMOVE_ROUTE(k)   route not kept
    FINISH_PLAN       always
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

from ..config import ExperimentConfig
from ..data.network import TransitNetwork
from .frequency import FleetCalculator
from .service_plan import routes_equivalent
from .state import EnvironmentState

ADD_STOP = "ADD_STOP"
REMOVE_LAST_STOP = "REMOVE_LAST_STOP"
END_ROUTE = "END_ROUTE"
INCREASE_FREQUENCY = "INCREASE_FREQUENCY"
DECREASE_FREQUENCY = "DECREASE_FREQUENCY"
KEEP_ROUTE = "KEEP_ROUTE"
REMOVE_ROUTE = "REMOVE_ROUTE"
FINISH_PLAN = "FINISH_PLAN"

ACTION_TYPES = (ADD_STOP, REMOVE_LAST_STOP, END_ROUTE, INCREASE_FREQUENCY,
                DECREASE_FREQUENCY, KEEP_ROUTE, REMOVE_ROUTE, FINISH_PLAN)
_TYPES_WITH_ARG = {ADD_STOP, INCREASE_FREQUENCY, DECREASE_FREQUENCY, KEEP_ROUTE, REMOVE_ROUTE}


class Action(NamedTuple):
    type: str
    arg: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.type}({self.arg})" if self.arg is not None else self.type


def action_from_string(text: str) -> Action:
    """Inverse of ``str(action)`` - used when loading Q-tables from CSV."""
    text = text.strip()
    if "(" in text:
        name, arg = text[:-1].split("(", 1)
        action = Action(name, int(arg))
    else:
        action = Action(text, None)
    if action.type not in ACTION_TYPES or ((action.arg is not None) != (action.type in _TYPES_WITH_ARG)):
        raise ValueError(f"Unknown action string: {text!r}")
    return action


class ActionMasker:
    """Computes the set of valid actions for a state."""

    def __init__(self, network: TransitNetwork, fleet: FleetCalculator, cfg: ExperimentConfig) -> None:
        self.network = network
        self.fleet = fleet
        self.cfg = cfg

    def end_route_frequency(self, state: EnvironmentState) -> Optional[float]:
        """Frequency END_ROUTE would assign, or None if END_ROUTE is invalid."""
        route, cfg = state.current_route, self.cfg
        if len(route) < cfg.min_route_length:
            return None
        if not cfg.allow_duplicate_routes and any(
                routes_equivalent(route, rs.route, cfg.routes_are_bidirectional) for rs in state.current_plan):
            return None
        return self.fleet.initial_frequency_for_new_route(route, state.fleet_remaining)

    def valid_actions(self, state: EnvironmentState) -> List[Action]:
        cfg, route, plan = self.cfg, state.current_route, state.current_plan
        actions: List[Action] = []

        if route:  # ---------------- route under construction ----------------
            if len(route) < cfg.max_route_length:
                for j in self.network.neighbours[route[-1] - 1]:
                    if cfg.allow_repeated_stops or j not in route:
                        actions.append(Action(ADD_STOP, j))
            if self.end_route_frequency(state) is not None:
                actions.append(Action(END_ROUTE))
            if len(route) > cfg.min_route_length or not actions:
                actions.append(Action(REMOVE_LAST_STOP))
            return actions

        # ---------------------- plan-level actions -----------------------------
        if len(plan) < cfg.max_routes and state.fleet_remaining > 0:
            actions.extend(Action(ADD_STOP, j) for j in self.network.stops)
        for k, rs in enumerate(plan):
            if rs.kept:
                continue
            up = rs.frequency + cfg.frequency_step
            if up <= cfg.max_frequency + 1e-9:
                extra = self.fleet.buses_required(rs.route, up) - self.fleet.buses_required(rs.route, rs.frequency)
                if extra <= state.fleet_remaining:
                    actions.append(Action(INCREASE_FREQUENCY, k))
            if rs.frequency - cfg.frequency_step >= cfg.min_frequency - 1e-9:
                actions.append(Action(DECREASE_FREQUENCY, k))
            actions.append(Action(KEEP_ROUTE, k))
            actions.append(Action(REMOVE_ROUTE, k))
        actions.append(Action(FINISH_PLAN))
        return actions
