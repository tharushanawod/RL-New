"""Hourly plan reward: R_t = S_t - alpha * O_t - lambda * C_t.

S_t = served passengers / total demand, using actual assignment counts.
Zero demand gives S_t = 0. Travel time remains a diagnostic only;
no transfer penalties, imputed unserved time, or overload penalties are added.
O_t = buses used / max fleet, using the existing rounded fleet calculation.
C_t = normalised route change from the existing route matching.

Raw components and weighted penalties are logged separately. No reward
normalisation or automatic coefficient rescaling is applied.
"""

from __future__ import annotations

from typing import Dict

from ..config import ExperimentConfig
from .passenger_assignment import AssignmentResult
from .transition_cost import TransitionResult


class RewardCalculator:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def service_terms(self, a: AssignmentResult) -> Dict[str, float]:
        served_ratio = a.served_passengers / a.total_demand if a.total_demand > 0 else 0.0
        return {
            "served_demand_ratio": served_ratio,
            "total_passenger_travel_time": a.total_passenger_travel_time,
            "service_component": served_ratio,
        }

    def operator_terms(self, fleet_used: int) -> Dict[str, float]:
        operator = fleet_used / self.cfg.max_fleet
        return {
            "fleet_used": fleet_used,
            "operator_component": operator,
            "operator_penalty": self.cfg.operator_cost_alpha * operator,
        }

    def compute(self, assignment: AssignmentResult, fleet_used: int,
                transition: TransitionResult) -> Dict[str, float]:
        s = self.service_terms(assignment)
        o = self.operator_terms(fleet_used)
        # Use route_change explicitly: frequency changes are diagnostic only.
        transition_component = self.cfg.transition_penalty_lambda * transition.route_change
        total = s["service_component"] - o["operator_penalty"] - transition_component
        return {
            **s,
            **o,
            "transition_cost": transition.route_change,
            "transition_component": transition_component,
            "total_reward": total,
        }
