"""Reward conservation and immediate feedback on a small synthetic network."""

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transit_rl.agent.q_learning_agent import QLearningAgent
from transit_rl.config import ExperimentConfig
from transit_rl.data.network import build_network
from transit_rl.environment.actions import Action
from transit_rl.environment.transit_environment import TransitEnvironment
from transit_rl.experiment.trainer import run_episode


class DenseRewardTests(unittest.TestCase):
    def make_env(self, **overrides):
        cfg = dataclasses.replace(
            ExperimentConfig(), num_stops=4, num_hours=2,
            od_start_file_hour=0, max_route_length=4, **overrides)
        cfg.validate()
        network = build_network([(1, 2, 8), (2, 3, 2), (3, 4, 3), (1, 4, 5)], 4, True)
        od = np.full((4, 4), 100.0)
        np.fill_diagonal(od, 0)
        return TransitEnvironment(network, [od, od * 2], cfg)

    def test_construction_feedback_and_commit_not_double_rewarded(self):
        env = self.make_env()
        first = env.step(Action("ADD_STOP", 1))
        second = env.step(Action("ADD_STOP", 2))
        self.assertEqual(first.reward, 0)
        self.assertGreater(second.reward, 0)
        self.assertAlmostEqual(second.reward, second.info["reward_breakdown"]["total_reward"])
        self.assertTrue(second.info["tentative_route_scored"])
        self.assertEqual(env.current_plan, ())
        self.assertEqual(env.state.fleet_remaining, 99)
        commit = env.step(Action("END_ROUTE"))
        self.assertAlmostEqual(commit.reward, 0)
        self.assertLess(env.state.fleet_remaining, 99)

    def test_all_actions_and_inherited_hour_conserve_rewards(self):
        env = self.make_env()
        actions = [Action("ADD_STOP", 1), Action("ADD_STOP", 2),
                   Action("ADD_STOP", 3), Action("REMOVE_LAST_STOP"),
                   Action("ADD_STOP", 3), Action("END_ROUTE"),
                   Action("INCREASE_FREQUENCY", 0), Action("DECREASE_FREQUENCY", 0),
                   Action("KEEP_ROUTE", 0), Action("ADD_STOP", 3),
                   Action("ADD_STOP", 4), Action("END_ROUTE"),
                   Action("REMOVE_ROUTE", 1), Action("FINISH_PLAN")]
        rewards = []
        for action in actions:
            result = env.step(action)
            self.assertIn("reward_breakdown", result.info)
            rewards.append(result.reward)
        # This frequency increase changes waiting time, but not served count or fleet.
        self.assertEqual(rewards[6], 0)
        self.assertAlmostEqual(rewards[6] + rewards[7], 0)
        self.assertEqual(rewards[8], 0)
        record = env.hourly_records[0]
        self.assertAlmostEqual(sum(rewards), record["total_reward"])
        self.assertAlmostEqual(sum(rewards), record["rl_reward_in_hour"])
        self.assertEqual(env.current_plan, env.previous_plan)
        second = env.step(Action("FINISH_PLAN"))
        self.assertTrue(second.done)
        self.assertAlmostEqual(second.reward, env.hourly_records[1]["total_reward"])
        self.assertAlmostEqual(sum(rewards) + second.reward,
                               sum(r["total_reward"] for r in env.hourly_records))

    def test_forced_finish_removes_tentative_service_reward(self):
        env = self.make_env(max_steps_per_hour=2)
        first = env.step(Action("ADD_STOP", 1))
        last = env.step(Action("ADD_STOP", 2))
        record = env.hourly_records[0]
        self.assertEqual(last.info["discarded_unfinished_route"], [1, 2])
        self.assertTrue(record["forced_finish"])
        self.assertEqual(record["total_fleet_used"], 0)
        self.assertAlmostEqual(first.reward + last.reward, record["total_reward"])
        self.assertAlmostEqual(record["rl_reward_in_hour"], record["total_reward"])

    def test_forced_frequency_change_logged_once(self):
        env = self.make_env(max_steps_per_hour=1,
                            initial_service_plan=({"route": [1, 2], "frequency": 4},))
        result = env.step(Action("INCREASE_FREQUENCY", 0))
        record = env.hourly_records[0]
        self.assertTrue(record["forced_finish"])
        self.assertAlmostEqual(result.reward, record["total_reward"])
        self.assertAlmostEqual(result.reward, record["rl_reward_in_hour"])

    def test_training_return_matches_daily_objective(self):
        env = self.make_env()
        agent = QLearningAgent(env.cfg, seed=2024)
        self.assertEqual(agent.gamma, 1.0)
        for train in (True, False):
            total, records, _, _ = run_episode(env, agent, train=train)
            self.assertAlmostEqual(total, sum(r["total_reward"] for r in records))
            for record in records:
                self.assertAlmostEqual(record["rl_reward_in_hour"], record["total_reward"])

    def test_legacy_modes_remain_available(self):
        for mode in ("terminal", "delta"):
            env = self.make_env(reward_mode=mode)
            start = env.evaluate_plan(env.current_plan).reward["total_reward"]
            total = sum(env.step(action).reward for action in
                        (Action("ADD_STOP", 1), Action("ADD_STOP", 2),
                         Action("END_ROUTE"), Action("FINISH_PLAN")))
            final = env.hourly_records[0]["total_reward"]
            self.assertAlmostEqual(total, final if mode == "terminal" else final - start)


if __name__ == "__main__":
    unittest.main()
