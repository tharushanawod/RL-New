"""Numerical reward checks and integration with assignment, matching and outputs."""

import csv
import dataclasses
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transit_rl.config import ExperimentConfig
from transit_rl.data.network import build_network
from transit_rl.environment.actions import Action
from transit_rl.environment.service_plan import RouteService
from transit_rl.environment.transit_environment import TransitEnvironment
from transit_rl.environment import transition_cost
from transit_rl.environment.transition_cost import TransitionCostCalculator, ZERO_TRANSITION
from transit_rl.experiment.trainer import save_hourly_plans, summarise_day, train_replication
from transit_rl.main import build_config, parse_args


class RewardTests(unittest.TestCase):
    def make_env(self, **overrides):
        cfg = dataclasses.replace(
            ExperimentConfig(), num_stops=3, num_hours=2, od_start_file_hour=0,
            max_fleet=100, bus_capacity=20, initial_route_frequency=2,
            max_route_length=3, make_plots=False, **overrides)
        cfg.validate()
        network = build_network([(1, 2, 5), (2, 3, 7)], 3, True)
        od = np.zeros((3, 3))
        od[0, 1], od[0, 2] = 80, 20
        od.setflags(write=False)
        return TransitEnvironment(network, [od, od * 2], cfg)

    def test_hand_calculated_reward_and_component_scales(self):
        env = self.make_env()
        assignment = env.assigner.assign(1, (RouteService((1, 2), 2),))
        # 80 served of 100; each spends 5 minutes travelling + 15 waiting.
        self.assertEqual(assignment.served_passengers, 80)
        self.assertEqual(assignment.total_passenger_travel_time, 1600)
        transition = dataclasses.replace(ZERO_TRANSITION, route_change=0.2, total=0.2)
        reward = env.reward_calc.compute(assignment, 10, transition)
        self.assertEqual(reward["served_demand_ratio"], 0.8)
        self.assertAlmostEqual(reward["service_component"], 0.8)
        self.assertAlmostEqual(reward["operator_component"], 0.1)
        self.assertAlmostEqual(reward["operator_penalty"], 0.03)
        self.assertAlmostEqual(reward["transition_component"], 0.06)
        self.assertAlmostEqual(reward["total_reward"], 0.71)
        self.assertAlmostEqual(reward["operator_penalty"] / reward["service_component"], 0.0375)
        self.assertAlmostEqual(reward["transition_component"] / reward["service_component"], 0.075)
        # RewardCalculator uses route_change, even if a caller supplies another total.
        self.assertEqual(reward, env.reward_calc.compute(
            assignment, 10, dataclasses.replace(transition, total=0.9, frequency_change=0.7)))

    def test_existing_fleet_rounding_and_overload_accounting(self):
        env = self.make_env()
        plan = (RouteService((1, 2), 2),)
        evaluation = env.evaluate_plan(plan)
        # ceil(2 * (5 + 5) / 60) = 1; route capacity = 2 * 20 = 40.
        self.assertEqual(evaluation.buses_per_route, [1])
        self.assertEqual(evaluation.assignment.overload_passengers, 40)
        self.assertAlmostEqual(evaluation.reward["operator_component"], 0.01)
        self.assertAlmostEqual(evaluation.reward["total_reward"], 0.8 - 0.3 * 0.01)

    def test_fleet_masking_and_bus_release_are_preserved(self):
        env = self.make_env()
        env = TransitEnvironment(env.network, env.od_matrices, dataclasses.replace(env.cfg, max_fleet=1))
        for action in (Action("ADD_STOP", 1), Action("ADD_STOP", 2), Action("END_ROUTE")):
            env.step(action)
        self.assertEqual(env.state.fleet_remaining, 0)
        self.assertFalse(any(a.type == "ADD_STOP" for a in env.valid_actions()))
        for _ in range(4):
            env.step(Action("INCREASE_FREQUENCY", 0))
        self.assertEqual(env.current_plan[0].frequency, 6)
        self.assertNotIn(Action("INCREASE_FREQUENCY", 0), env.valid_actions())
        env.step(Action("REMOVE_ROUTE", 0))
        self.assertEqual(env.state.fleet_remaining, 1)

    def test_service_depends_on_coverage_and_is_independent_of_travel_time(self):
        env = self.make_env()
        assignment = env.assigner.assign(1, (RouteService((1, 2), 2),))
        # Halving the served count halves quality, even if travel time also halves.
        half_served = dataclasses.replace(assignment, served_passengers=40, total_passenger_travel_time=800)
        self.assertAlmostEqual(env.reward_calc.service_terms(half_served)["service_component"], 0.4)
        double_demand = dataclasses.replace(assignment, total_demand=200, served_passengers=160,
                                            total_passenger_travel_time=3200)
        self.assertAlmostEqual(env.reward_calc.service_terms(double_demand)["service_component"], 0.8)
        for travel_time in (0, 1e-12, 1600, 3200):
            with self.subTest(travel_time=travel_time):
                altered = dataclasses.replace(assignment, total_passenger_travel_time=travel_time)
                reward = env.reward_calc.compute(altered, 1, ZERO_TRANSITION)
                self.assertAlmostEqual(reward["service_component"], 0.8)
                self.assertAlmostEqual(reward["total_reward"], 0.797)

    def test_zero_demand_and_unserved_are_safe(self):
        env = self.make_env()
        unserved = env.assigner.assign(1, ())
        self.assertEqual(env.reward_calc.compute(unserved, 0, ZERO_TRANSITION)["total_reward"], 0)
        served = env.assigner.assign(1, (RouteService((1, 2), 2),))
        for assignment in (
            dataclasses.replace(served, total_demand=0, served_passengers=0),
            dataclasses.replace(unserved, total_demand=0),
        ):
            with self.subTest(assignment=assignment):
                reward = env.reward_calc.compute(assignment, 10, ZERO_TRANSITION)
                self.assertEqual(reward["service_component"], 0)
                self.assertAlmostEqual(reward["total_reward"], -0.03)
                self.assertTrue(math.isfinite(reward["total_reward"]))

    def test_transfer_service_uses_actual_served_count(self):
        env = self.make_env(operator_cost_alpha=0, transition_penalty_lambda=0)
        assignment = env.assigner.assign(
            1, (RouteService((1, 2), 2), RouteService((2, 3), 2)))
        # Direct 80*(5+15)=1600; transfer 20*(5+7+15+15)=840.
        self.assertEqual(assignment.total_passenger_travel_time, 2440)
        self.assertEqual(assignment.total_transfer_penalty_time, 100)
        self.assertEqual(assignment.one_transfer_passengers, 20)
        self.assertAlmostEqual(env.reward_calc.compute(assignment, 2, ZERO_TRANSITION)["total_reward"],
                               1)

    def test_diagnostic_metrics_do_not_add_reward_terms(self):
        env = self.make_env()
        assignment = env.assigner.assign(1, (RouteService((1, 2), 2),))
        altered = dataclasses.replace(
            assignment, total_transfer_penalty_time=10000, overload_passengers=10000,
            direct_passengers=0, one_transfer_passengers=80, two_transfer_passengers=0)
        self.assertEqual(env.reward_calc.compute(assignment, 1, ZERO_TRANSITION),
                         env.reward_calc.compute(altered, 1, ZERO_TRANSITION))

    def test_frequency_transition_is_diagnostic_only(self):
        env = self.make_env()
        previous = (RouteService((1, 2), 2),)
        current = (RouteService((1, 2), 4),)
        transition = env.transition_calc.compute(previous, current)
        self.assertAlmostEqual(transition.frequency_change, 1 / 3)
        self.assertEqual(transition.total, 0)
        env.hour = 2
        env.previous_plan = previous
        result = env.evaluate_plan(current)
        self.assertEqual(result.reward["transition_component"], 0)
        self.assertAlmostEqual(result.reward["total_reward"],
                               0.8 - 0.3 * 0.01)
        before = env.evaluate_plan(previous)
        self.assertNotEqual(before.assignment.total_waiting_time, result.assignment.total_waiting_time)
        self.assertEqual(before.reward["total_reward"], result.reward["total_reward"])

    def test_hungarian_reversal_and_edit_distance_are_preserved(self):
        if transition_cost._hungarian is None:
            self.skipTest("SciPy is optional and unavailable")
        calc = TransitionCostCalculator(ExperimentConfig())
        previous = (RouteService((1, 2, 3), 2), RouteService((4, 5, 6), 4))
        current = (RouteService((6, 5, 4), 8), RouteService((1, 2, 7), 6))
        with patch.object(transition_cost, "_hungarian", wraps=transition_cost._hungarian) as hungarian:
            result = calc.compute(previous, current)
        hungarian.assert_called_once()
        self.assertEqual(result.matches, ((0, 1, 1 / 3), (1, 0, 0)))
        self.assertAlmostEqual(result.route_change, 1 / 6)
        self.assertAlmostEqual(result.total, 1 / 6)

    def test_route_addition_removal_threshold_and_empty_plans(self):
        for method in ("hungarian", "greedy"):
            calc = TransitionCostCalculator(dataclasses.replace(
                ExperimentConfig(), route_matching_method=method))
            old = (RouteService((1, 2), 2),)
            different = (RouteService((3, 4), 2),)
            for previous, current in ((old, ()), ((), old), (old, different)):
                with self.subTest(method=method, previous=previous, current=current):
                    result = calc.compute(previous, current)
                    self.assertEqual(result.total, 1)
                    self.assertEqual(result.matches, ())
            self.assertEqual(calc.compute((), ()), ZERO_TRANSITION)

    def test_first_hour_transition_policy_is_preserved(self):
        for apply_cost in (False, True):
            env = self.make_env(apply_transition_cost_first_hour=apply_cost)
            result = env.evaluate_plan((RouteService((1, 2), 2),))
            self.assertEqual(result.reward["transition_cost"], float(apply_cost))

    def test_config_and_cli_coefficients(self):
        cfg = build_config(parse_args(["--alpha", "0.002", "--lambda", "0.004"]))
        self.assertEqual(cfg.operator_cost_alpha, 0.002)
        self.assertEqual(cfg.transition_penalty_lambda, 0.004)
        for field in ("operator_cost_alpha", "transition_penalty_lambda"):
            for value in (-1, float("inf"), float("nan")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    dataclasses.replace(cfg, **{field: value}).validate()
        dataclasses.replace(cfg, operator_cost_alpha=0, transition_penalty_lambda=0).validate()
        self.assertNotIn("w_operator_route_time", dataclasses.asdict(cfg))
        self.assertNotIn("frequency_change_weight", dataclasses.asdict(cfg))

    def test_hourly_export_and_daily_components_match_formulas(self):
        env = self.make_env(initial_service_plan=({"route": [1, 2], "frequency": 2},))
        for _ in range(2):
            env.step(Action("FINISH_PLAN"))
        records = env.hourly_records
        day = summarise_day(records)
        self.assertAlmostEqual(day["served_demand_ratio"], 0.8)
        self.assertAlmostEqual(day["total_service_component"], 0.8 + 0.8)
        self.assertAlmostEqual(day["daily_plan_reward"], day["total_service_component"]
                               - day["total_operator_penalty"] - day["total_transition_component"])
        with tempfile.TemporaryDirectory() as tmp:
            save_hourly_plans(records, Path(tmp))
            with (Path(tmp) / "hourly_service_plans.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                self.assertAlmostEqual(float(row["service_component"]),
                                       float(row["served_passengers"]) / float(row["total_demand"]))
                self.assertAlmostEqual(float(row["total_reward"]), float(row["service_component"])
                                       - float(row["operator_penalty"]) - float(row["transition_component"]))
                self.assertEqual(row["transition_cost"], row["transition_route_change"])
            saved = json.loads((Path(tmp) / "hourly_service_plans.json").read_text())
            self.assertEqual(list(saved.values()), records)

    def test_training_csv_and_progress_match_reward_components(self):
        env = self.make_env(num_episodes=2, log_every=1)
        with tempfile.TemporaryDirectory() as tmp:
            result = train_replication(1, env.cfg, env.network, env.od_matrices, Path(tmp))
            folder = Path(tmp) / "replication_1"
            with (folder / "learning_curve.csv").open(newline="", encoding="utf-8") as fh:
                curve = list(csv.DictReader(fh))
            with (folder / "progress_log.csv").open(newline="", encoding="utf-8") as fh:
                progress = list(csv.DictReader(fh))
            for row, log in zip(curve, progress):
                self.assertAlmostEqual(float(row["daily_plan_reward"]), float(row["total_service_component"])
                                       - float(row["total_operator_penalty"]) - float(row["total_transition_component"]))
                self.assertEqual(float(log["mean_service_component_last_window"]),
                                 float(row["total_service_component"]))
            summary = result["summary"]
            self.assertAlmostEqual(summary["evaluation_daily_plan_reward"], summary["total_service_component"]
                                   - summary["total_operator_penalty"] - summary["total_transition_component"])


if __name__ == "__main__":
    unittest.main()
