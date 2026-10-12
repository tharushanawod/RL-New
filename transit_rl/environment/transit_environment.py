"""
Time-dependent transit network design environment.

    1 episode = 24 hourly steps t = 1..24 processed in sequence.  Step t uses
                the OD file given by ExperimentConfig.od_file_number(t); by
                default the day starts at 08:00 (od_hour_08 ... 23, 00 ... 07).
    1 RL step = one action of the agent

Hour lifecycle
--------------
start of hour t:
    previous_plan = P[t-1]                       (never modified afterwards)
    current_plan  = copy of P[t-1] with kept flags cleared ("copy_previous")
                    or the empty plan ("empty")
during hour t:
    the agent builds / adapts current_plan with route and frequency actions
FINISH_PLAN (or the per-hour step limit):
    P[t] = current_plan                          (immutable tuple = deep copy)
    hour t+1 starts with previous_plan = P[t]

Plans are immutable tuples of NamedTuples, so assigning them can never alias
or mutate an earlier plan; this is equivalent to ``deepcopy`` but cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..data.network import TransitNetwork, route_is_physical_path
from ..data.od_loader import od_to_key
from . import actions as A
from .actions import Action, ActionMasker
from .frequency import FleetCalculator
from .passenger_assignment import AssignmentResult, PassengerAssignment
from .reward import RewardCalculator
from .service_plan import RouteService, ServicePlan, make_plan, plan_to_records, reset_kept_flags
from .state import EnvironmentState
from .transition_cost import ZERO_TRANSITION, TransitionCostCalculator, TransitionResult


class InvalidActionError(ValueError):
    pass


@dataclass(frozen=True)
class PlanEvaluation:
    """Complete evaluation of a (partial or final) hourly plan."""

    assignment: AssignmentResult
    transition: TransitionResult
    buses_per_route: List[int]
    fleet_used: int
    reward: Dict[str, float]


@dataclass
class StepResult:
    next_state: Optional[EnvironmentState]  # None after hour 24 (terminal)
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class TransitEnvironment:
    """Transport simulation wrapped as an RL environment (no learning logic)."""

    def __init__(self, network: TransitNetwork, od_matrices: Sequence[np.ndarray],
                 cfg: ExperimentConfig) -> None:
        if len(od_matrices) != cfg.num_hours:
            raise ValueError(f"Expected {cfg.num_hours} OD matrices, got {len(od_matrices)}")
        for h, m in enumerate(od_matrices, 1):
            if m.shape != (network.num_stops, network.num_stops):
                raise ValueError(f"OD_{h} has shape {m.shape}, expected {(network.num_stops,) * 2}")
        self.network = network
        self.cfg = cfg
        self.od_matrices = list(od_matrices)
        # One shared, lossless, hashable copy of each raw OD matrix.
        self._od_keys = [od_to_key(m) for m in self.od_matrices]

        self.fleet = FleetCalculator(network, cfg)
        self.assigner = PassengerAssignment(network, cfg, self.od_matrices)
        self.transition_calc = TransitionCostCalculator(cfg)
        self.reward_calc = RewardCalculator(cfg)
        self.masker = ActionMasker(network, self.fleet, cfg)
        self.initial_plan: ServicePlan = self._validated_initial_plan()
        self.reset()

    # ------------------------------------------------------------ validation
    def _validated_initial_plan(self) -> ServicePlan:
        cfg = self.cfg
        plan = make_plan(cfg.initial_service_plan)
        for rs in plan:
            if not (cfg.min_route_length <= len(rs.route) <= cfg.max_route_length):
                raise ValueError(f"INITIAL_SERVICE_PLAN route {rs.route} violates route-length limits")
            if not route_is_physical_path(self.network, rs.route):
                raise ValueError(f"INITIAL_SERVICE_PLAN route {rs.route} is not a physical path")
            if not (cfg.min_frequency <= rs.frequency <= cfg.max_frequency):
                raise ValueError(f"INITIAL_SERVICE_PLAN frequency {rs.frequency} outside limits")
        if len(plan) > cfg.max_routes or self.fleet.fleet_used(plan) > cfg.max_fleet:
            raise ValueError("INITIAL_SERVICE_PLAN exceeds MAX_ROUTES or MAX_FLEET")
        return plan

    # --------------------------------------------------------------- episode
    def reset(self) -> EnvironmentState:
        """Start a new episode at hour 1."""
        self.hour = 1
        self.hourly_plans: Dict[int, ServicePlan] = {0: self.initial_plan}  # P[0] = P_0
        self.hourly_records: List[Dict[str, Any]] = []
        self.episode_rl_steps = 0
        self.done = False
        self._start_hour()
        return self.state

    def _start_hour(self) -> None:
        self.previous_plan: ServicePlan = self.hourly_plans[self.hour - 1]
        self.current_plan: ServicePlan = (reset_kept_flags(self.previous_plan)
                                          if self.cfg.current_plan_start_mode == "copy_previous" else ())
        self.current_route: tuple = ()
        self.steps_in_hour = 0
        self.hour_rl_reward = 0.0
        self._current_eval: Optional[PlanEvaluation] = (
            self.evaluate_plan(self.current_plan)
            if self.cfg.reward_mode in ("delta", "dense") else None)
        self._hour_start_reward = (
            self._current_eval.reward["total_reward"] if self._current_eval is not None else 0.0)

    @property
    def state(self) -> EnvironmentState:
        return EnvironmentState(
            hour=self.hour,
            od_matrix=self._od_keys[self.hour - 1],
            current_route=self.current_route,
            current_plan=self.current_plan,
            previous_plan=self.previous_plan,
            fleet_remaining=self.fleet.fleet_remaining(self.current_plan),
        )

    def valid_actions(self) -> List[Action]:
        if self.done:
            return []
        return self.masker.valid_actions(self.state)

    # ------------------------------------------------------------ evaluation
    def evaluate_plan(self, plan: ServicePlan) -> PlanEvaluation:
        """Passenger assignment + transition cost + reward for ``plan`` in the current hour."""
        buses = self.fleet.buses_per_route(plan)
        assignment = self.assigner.assign(self.hour, plan)
        if self.hour == 1 and not self.cfg.apply_transition_cost_first_hour:
            transition = ZERO_TRANSITION
        else:
            transition = self.transition_calc.compute(self.previous_plan, plan)
        reward = self.reward_calc.compute(assignment, sum(buses), transition)
        return PlanEvaluation(assignment, transition, buses, sum(buses), reward)

    # ------------------------------------------------------------------ step
    def _evaluate_action_plan(self) -> PlanEvaluation:
        """Score the committed plan plus a feasible route under construction.

        Tentative routes affect reward feedback only: they reserve no buses
        and are not stored in the service plan until END_ROUTE. A duplicate,
        too-short, or fleet-infeasible route contributes no tentative service.
        """
        frequency = self.masker.end_route_frequency(self.state)
        plan = self.current_plan
        if frequency is not None:
            plan = plan + (RouteService(self.current_route, frequency, False),)
        return self.evaluate_plan(plan)

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode finished; call reset()")
        state = self.state
        valid = self.masker.valid_actions(state)
        if action not in valid:
            if self.cfg.raise_on_invalid_action:
                raise InvalidActionError(f"Action {action} is not valid in hour {self.hour} "
                                         f"(route={self.current_route}, valid={[str(a) for a in valid]})")
            return StepResult(state, 0.0, False, {"invalid_action": str(action)})

        self.episode_rl_steps += 1
        self.steps_in_hour += 1
        info: Dict[str, Any] = {"hour": self.hour, "action": str(action)}
        plan_changed = False
        t = action.type

        if t == A.ADD_STOP:
            self.current_route = self.current_route + (action.arg,)
        elif t == A.REMOVE_LAST_STOP:
            self.current_route = self.current_route[:-1]
        elif t == A.END_ROUTE:
            freq = self.masker.end_route_frequency(state)
            self.current_plan = self.current_plan + (RouteService(self.current_route, freq, False),)
            self.current_route = ()
            plan_changed = True
        elif t in (A.INCREASE_FREQUENCY, A.DECREASE_FREQUENCY):
            k = action.arg
            step = self.cfg.frequency_step if t == A.INCREASE_FREQUENCY else -self.cfg.frequency_step
            rs = self.current_plan[k]
            new_rs = rs._replace(frequency=round(rs.frequency + step, 10))
            self.current_plan = self.current_plan[:k] + (new_rs,) + self.current_plan[k + 1:]
            plan_changed = True
        elif t == A.KEEP_ROUTE:
            k = action.arg
            self.current_plan = (self.current_plan[:k] + (self.current_plan[k]._replace(kept=True),)
                                 + self.current_plan[k + 1:])
        elif t == A.REMOVE_ROUTE:
            k = action.arg  # its buses return to F_remaining automatically
            self.current_plan = self.current_plan[:k] + self.current_plan[k + 1:]
            plan_changed = True

        reward = 0.0
        # --- passenger assignment after meaningful operational changes -------
        if self.cfg.reward_mode == "dense":
            new_eval = self._evaluate_action_plan()
            reward = new_eval.reward["total_reward"] - self._current_eval.reward["total_reward"]
            self._current_eval = new_eval
            info["reward_breakdown"] = new_eval.reward
            info["tentative_route_scored"] = bool(
                self.current_route and self.masker.end_route_frequency(self.state) is not None)
        elif plan_changed and (self.cfg.evaluate_after_each_change or self.cfg.reward_mode == "delta"):
            new_eval = self.evaluate_plan(self.current_plan)
            info["reward_breakdown"] = new_eval.reward
            if self.cfg.reward_mode == "delta":
                reward += new_eval.reward["total_reward"] - self._current_eval.reward["total_reward"]
                self._current_eval = new_eval
        elif (self.cfg.assign_after_route_construction_steps and t in (A.ADD_STOP, A.REMOVE_LAST_STOP)
              and len(self.current_route) >= 2):
            # Optional, information only: evaluate the plan as if the partial
            # route were committed at its END_ROUTE frequency.
            f = self.fleet.initial_frequency_for_new_route(self.current_route,
                                                          self.fleet.fleet_remaining(self.current_plan))
            if f is not None:
                tentative = self.current_plan + (RouteService(self.current_route, f, False),)
                info["tentative_reward_breakdown"] = self.evaluate_plan(tentative).reward

        # --- end of hour --------------------------------------------------------
        self.hour_rl_reward += reward
        forced = t != A.FINISH_PLAN and self.steps_in_hour >= self.cfg.max_steps_per_hour
        if t == A.FINISH_PLAN or forced:
            if forced and self.current_route:
                info["discarded_unfinished_route"] = list(self.current_route)
                self.current_route = ()
            reward += self._finish_hour(forced, info)

        next_state = None if self.done else self.state
        return StepResult(next_state, reward, self.done, info)

    def _finish_hour(self, forced: bool, info: Dict[str, Any]) -> float:
        final = self.evaluate_plan(self.current_plan)
        reward = final.reward["total_reward"] if self.cfg.reward_mode == "terminal" else 0.0
        if self.cfg.reward_mode == "delta":
            # Any change since the last evaluation was already rewarded.
            reward = final.reward["total_reward"] - self._current_eval.reward["total_reward"]
        elif self.cfg.reward_mode == "dense":
            # Undo any tentative service discarded at the deadline, and restore
            # the hour's starting score. Intermediate deltas + this correction
            # sum to the committed plan's score, even with an inherited plan.
            reward = (final.reward["total_reward"] - self._current_eval.reward["total_reward"]
                      + self._hour_start_reward)

        p_t = reset_kept_flags(self.current_plan)
        self.hourly_plans[self.hour] = p_t   # P[t] (immutable copy)
        clock = self.cfg.od_file_number(self.hour)
        record = {
            "hour": self.hour,                      # episode step t (1..24)
            "clock_hour": clock,                    # e.g. 8 for 08:00-09:00
            "od_file": self.cfg.od_filename_pattern.format(hour=clock),
            "routes": [list(rs.route) for rs in p_t],
            "frequencies": [rs.frequency for rs in p_t],
            "buses_per_route": final.buses_per_route,
            "round_trip_minutes": [self.fleet.round_trip_time(rs.route) for rs in p_t],
            "service_plan": plan_to_records(p_t, include_kept=False),
            "previous_service_plan": plan_to_records(self.previous_plan, include_kept=False),
            "total_fleet_used": final.fleet_used,
            **final.assignment.metrics(),
            **{k: v for k, v in final.transition.as_dict().items()},
            **final.reward,
            "forced_finish": forced,
            "rl_steps_in_hour": self.steps_in_hour,
            "rl_reward_in_hour": self.hour_rl_reward + reward,
        }
        self.hourly_records.append(record)
        info["hour_finished"] = self.hour
        info["final_reward_breakdown"] = final.reward

        if self.hour == self.cfg.num_hours:
            self.done = True
        else:
            self.hour += 1
            self._start_hour()
        return reward
