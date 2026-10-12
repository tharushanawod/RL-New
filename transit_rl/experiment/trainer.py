"""
Training / evaluation loop for one replication (glue between agent and environment).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..agent.q_learning_agent import QLearningAgent
from ..config import ExperimentConfig
from ..data.network import TransitNetwork
from ..environment.state import state_to_key
from ..environment.transit_environment import TransitEnvironment
from ..utils import metrics, plotting, qtable_io

CURVE_COLUMNS = [
    "episode", "total_reward", "q_table_size", "unique_states", "epsilon",
    "daily_plan_reward", "new_states_in_episode", "rl_steps",
    "served_passengers", "total_demand", "served_demand_ratio", "total_passenger_travel_time",
    "total_service_component", "total_operator_component", "total_operator_penalty",
    "total_transition_component",
    "average_travel_time", "average_waiting_time", "unserved_passengers",
    "direct_passengers", "transfer_passengers", "average_fleet_used", "max_fleet_used",
    "total_transition_cost", "forced_hour_finishes", "hourly_rewards",
]

HOURLY_PLAN_COLUMNS = [
    "hour", "step", "clock_hour", "od_file", "routes", "frequencies", "buses_per_route", "total_fleet_used",
    "total_passenger_travel_time", "total_in_vehicle_time", "total_waiting_time", "average_waiting_time",
    "average_travel_time", "direct_passengers", "one_transfer_passengers", "two_transfer_passengers",
    "transfer_passengers", "unserved_passengers", "overload_passengers", "total_demand",
    "served_passengers", "served_demand_ratio",
    "transition_route_change", "transition_frequency_change", "transition_cost",
    "service_component", "operator_component", "operator_penalty", "transition_component", "total_reward",
    "forced_finish", "rl_steps_in_hour",
]


def run_episode(env: TransitEnvironment, agent: QLearningAgent, train: bool
                ) -> Tuple[float, List[Dict[str, Any]], int, int]:
    """Run one 24-hour episode.

    Returns (sum of RL rewards, hourly records, RL steps, steps taken in
    states that were not in the Q-table at decision time).
    """
    state = env.reset()
    key = state_to_key(state)
    valid = env.valid_actions()
    total, steps, unseen = 0.0, 0, 0
    while True:
        if key not in agent.q:
            unseen += 1
        action = agent.select_action(key, valid, greedy=not train)
        result = env.step(action)
        steps += 1
        total += result.reward
        if result.done:
            if train:
                agent.update(key, action, result.reward, None, [], True)
            break
        next_key = state_to_key(result.next_state)
        next_valid = env.valid_actions()
        if train:
            agent.update(key, action, result.reward, next_key, next_valid, False)
        key, valid = next_key, next_valid
    return total, env.hourly_records, steps, unseen


def summarise_day(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum hourly components; daily served ratio is demand-weighted.

The objective is sum_t R_t, not a reward recomputed from daily totals.
"""
    served = sum(r["served_passengers"] for r in records)
    demand = sum(r["total_demand"] for r in records)
    return {
        "daily_plan_reward": sum(r["total_reward"] for r in records),
        "hourly_rewards": [r["total_reward"] for r in records],
        "served_passengers": served,
        "total_demand": demand,
        "served_demand_ratio": served / demand if demand > 0 else 0.0,
        "total_service_component": sum(r["service_component"] for r in records),
        "total_operator_component": sum(r["operator_component"] for r in records),
        "total_operator_penalty": sum(r["operator_penalty"] for r in records),
        "total_transition_component": sum(r["transition_component"] for r in records),
        "total_passenger_travel_time": sum(r["total_passenger_travel_time"] for r in records),
        "average_travel_time": (sum(r["total_passenger_travel_time"] for r in records) / served) if served else 0.0,
        "average_waiting_time": (sum(r["total_waiting_time"] for r in records) / served) if served else 0.0,
        "unserved_passengers": sum(r["unserved_passengers"] for r in records),
        "direct_passengers": sum(r["direct_passengers"] for r in records),
        "transfer_passengers": sum(r["transfer_passengers"] for r in records),
        "average_fleet_used": float(np.mean([r["total_fleet_used"] for r in records])),
        "max_fleet_used": max(r["total_fleet_used"] for r in records),
        "total_transition_cost": sum(r["transition_cost"] for r in records),
        "forced_hour_finishes": sum(1 for r in records if r["forced_finish"]),
    }


def _json_cell(v: Any) -> Any:
    return json.dumps(v) if isinstance(v, (list, dict, tuple)) else v


def _hour_label(record: Dict[str, Any]) -> str:
    """Clock-hour label, e.g. 'hour_08' for the plan built from od_hour_08."""
    return f"hour_{record['clock_hour']:02d}"


def save_hourly_plans(records: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    with (out_dir / "hourly_service_plans.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HOURLY_PLAN_COLUMNS)
        for r in records:
            row = dict(r, hour=_hour_label(r), step=r["hour"])
            w.writerow([_json_cell(row[c]) for c in HOURLY_PLAN_COLUMNS])
    with (out_dir / "hourly_service_plans.json").open("w", encoding="utf-8") as fh:
        json.dump({_hour_label(r): r for r in records}, fh, indent=2)


def train_replication(replication: int, cfg: ExperimentConfig, network: TransitNetwork,
                      od_matrices: Sequence[np.ndarray], out_root: Path) -> Dict[str, Any]:
    """Train one replication, evaluate greedily, save everything, return a summary."""
    seed = cfg.seed_for_replication(replication)
    out_dir = Path(out_root) / f"replication_{replication}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = TransitEnvironment(network, od_matrices, cfg)
    agent = QLearningAgent(cfg, seed)

    curve: List[Dict[str, Any]] = []
    progress_path = out_dir / "progress_log.csv"
    t0 = time.perf_counter()
    with progress_path.open("w", newline="", encoding="utf-8") as pfh:
        progress = csv.writer(pfh)
        progress.writerow(["episode", "elapsed_s", "epsilon", "unique_states", "q_table_size",
                           "mean_reward_last_window", "memory_mb", "assignment_cache_hit_rate",
                           "mean_service_component_last_window", "mean_operator_penalty_last_window",
                           "mean_transition_component_last_window"])
        for episode in range(1, cfg.num_episodes + 1):
            states_before = agent.num_states
            eps_used = agent.epsilon
            total, records, steps, _ = run_episode(env, agent, train=True)
            day = summarise_day(records)
            curve.append({
                "episode": episode, "total_reward": total, "q_table_size": agent.num_state_action_pairs,
                "unique_states": agent.num_states, "epsilon": eps_used,
                "new_states_in_episode": agent.num_states - states_before, "rl_steps": steps,
                **{k: day[k] for k in CURVE_COLUMNS if k in day},
            })
            agent.decay_epsilon()

            if episode % cfg.log_every == 0 or episode == cfg.num_episodes:
                window = curve[-cfg.log_every:]
                mean_r = float(np.mean([c["total_reward"] for c in window]))
                mean_s, mean_o, mean_c = (
                    float(np.mean([c[column] for c in window])) for column in
                    ("total_service_component", "total_operator_penalty", "total_transition_component"))
                a = env.assigner
                hit = a.cache_hits / max(1, a.cache_hits + a.cache_misses)
                mem = metrics.process_memory_mb()
                elapsed = time.perf_counter() - t0
                progress.writerow([episode, round(elapsed, 2), eps_used, agent.num_states,
                                   agent.num_state_action_pairs, mean_r, round(mem, 1), round(hit, 4),
                                   mean_s, mean_o, mean_c])
                pfh.flush()
                print(f"[rep {replication}] ep {episode:>6}/{cfg.num_episodes} | eps {eps_used:.3f} | "
                      f"mean reward {mean_r:.6g} | S {mean_s:.6g}, alpha*O {mean_o:.6g}, "
                      f"lambda*C {mean_c:.6g} | states {agent.num_states:>9,} | "
                      f"Q pairs {agent.num_state_action_pairs:>9,} | mem {mem:8.1f} MB | {elapsed:7.1f}s",
                      flush=True)
    train_seconds = time.perf_counter() - t0

    # ---- learning curve -----------------------------------------------------
    with (out_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CURVE_COLUMNS)
        for row in curve:
            w.writerow([_json_cell(row[c]) for c in CURVE_COLUMNS])

    # ---- greedy evaluation episode (epsilon = 0, no learning) ----------------
    eval_reward, eval_records, eval_steps, eval_unseen = run_episode(env, agent, train=False)
    save_hourly_plans(eval_records, out_dir)
    eval_day = summarise_day(eval_records)

    # ---- Q-table + state-space statistics -----------------------------------
    qpath = out_dir / "q_table.csv"
    rows = qtable_io.save_q_table_csv(
        qpath, agent.q, agent.visits, replication, cfg.qtable_csv_od_mode,
        {t: cfg.od_file_number(t) for t in range(1, cfg.num_hours + 1)})
    stats = {
        "replication": replication,
        "random_seed": seed,
        **metrics.visit_statistics(agent.q, agent.visits),
        **metrics.state_space_estimate(network, cfg),
        "q_table_csv_rows": rows,
        "q_table_csv_megabytes": qpath.stat().st_size / 2**20,
        "process_memory_mb_after_training": metrics.process_memory_mb(),
        "training_seconds": train_seconds,
        "evaluation_steps": eval_steps,
        "evaluation_steps_in_unseen_states": eval_unseen,
        "evaluation_unseen_state_fraction": eval_unseen / eval_steps if eval_steps else 0.0,
    }
    with (out_dir / "state_space_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if cfg.make_plots:
        plotting.plot_learning_curves(curve, out_dir / "plots", f"Replication {replication}: ")

    last = curve[-min(len(curve), max(1, cfg.log_every)):]
    summary = {
        "replication": replication,
        "random_seed": seed,
        "evaluation_reward": eval_reward,
        "evaluation_daily_plan_reward": eval_day["daily_plan_reward"],
        "operator_cost_alpha": cfg.operator_cost_alpha,
        "transition_penalty_lambda": cfg.transition_penalty_lambda,
        "served_demand_ratio": eval_day["served_demand_ratio"],
        "total_passenger_travel_time": eval_day["total_passenger_travel_time"],
        "total_service_component": eval_day["total_service_component"],
        "total_operator_component": eval_day["total_operator_component"],
        "total_operator_penalty": eval_day["total_operator_penalty"],
        "total_transition_component": eval_day["total_transition_component"],
        "average_passenger_travel_time": eval_day["average_travel_time"],
        "average_waiting_time": eval_day["average_waiting_time"],
        "total_unserved_demand": eval_day["unserved_passengers"],
        "average_fleet_usage": eval_day["average_fleet_used"],
        "total_transition_cost": eval_day["total_transition_cost"],
        "unique_states": agent.num_states,
        "q_table_size": agent.num_state_action_pairs,
        "states_visited_more_than_once": stats["states_visited_more_than_once"],
        "average_visit_count": stats["average_visit_count"],
        "evaluation_unseen_state_fraction": stats["evaluation_unseen_state_fraction"],
        "mean_training_reward_last_window": float(np.mean([c["total_reward"] for c in last])),
        "training_seconds": train_seconds,
        "q_table_csv_megabytes": stats["q_table_csv_megabytes"],
    }
    return {"summary": summary, "curve": curve}
