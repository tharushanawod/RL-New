"""
Entry point:  python -m transit_rl.main [options]

Runs NUM_REPLICATIONS independent replications x NUM_EPISODES episodes of
raw-state tabular Q-learning, then writes results/replication_summary.csv and
identifies the best replication.

BEST-REPLICATION CRITERION: highest greedy evaluation reward (sum of the 24
hourly plan rewards with epsilon = 0); ties broken by the higher mean
training reward over the final logging window.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import sys
from pathlib import Path

from .config import ExperimentConfig
from .data.network import PlaceholderDataError, load_network_from_csv
from .data.od_loader import load_od_matrices
from .experiment.trainer import train_replication
from .utils import plotting


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw-state tabular Q-learning for hourly bus network design")
    p.add_argument("--episodes", type=int, help="episodes per replication (default: config)")
    p.add_argument("--replications", type=int, help="number of replications (default: config)")
    p.add_argument("--data-dir", type=Path, help="folder containing mandl_links.csv and od/")
    p.add_argument("--results-dir", type=Path, help="output folder")
    p.add_argument("--seed", type=int, help="base random seed")
    p.add_argument("--log-every", type=int, help="console/progress logging interval (episodes)")
    p.add_argument("--no-plots", action="store_true", help="disable matplotlib plots")
    p.add_argument("--start-hour", type=int, metavar="H",
                   help="OD file number to start each episode with, e.g. 8 -> od_hour_08, 09, ..., 23, 00, ..., 07 "
                        "(default: OD_START_FILE_HOUR in config)")
    p.add_argument("--od-mode", choices=["inline", "reference"], help="Q-table CSV OD storage mode")
    p.add_argument("--alpha", type=float, help="operational cost coefficient (default: config)")
    p.add_argument("--lambda", dest="transition_lambda", type=float,
                   help="route transition cost coefficient (default: config)")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {}
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.replications is not None:
        overrides["num_replications"] = args.replications
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir
    if args.seed is not None:
        overrides["base_random_seed"] = args.seed
    if args.log_every is not None:
        overrides["log_every"] = args.log_every
    if args.no_plots:
        overrides["make_plots"] = False
    if args.start_hour is not None:
        overrides["od_start_file_hour"] = args.start_hour
    if args.od_mode is not None:
        overrides["qtable_csv_od_mode"] = args.od_mode
    if args.alpha is not None:
        overrides["operator_cost_alpha"] = args.alpha
    if args.transition_lambda is not None:
        overrides["transition_penalty_lambda"] = args.transition_lambda
    cfg = dataclasses.replace(ExperimentConfig(), **overrides)
    cfg.validate()
    return cfg


def main(argv=None) -> int:
    cfg = build_config(parse_args(argv))
    try:
        network = load_network_from_csv(cfg.links_path, cfg.num_stops, cfg.links_are_bidirectional,
                                        cfg.input_stop_id_base)
        od_matrices = load_od_matrices(cfg.od_dir, cfg.od_filename_pattern, cfg.num_hours, cfg.num_stops,
                                       cfg.od_file_numbers())
    except PlaceholderDataError as exc:
        print(f"\nINPUT DATA MISSING:\n  {exc}\nSee transit_rl/data/input/README.md\n", file=sys.stderr)
        return 2

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Network: {network.num_stops} stops, {network.num_links()} directed links | "
          f"{len(od_matrices)} OD matrices | {cfg.num_replications} replications x {cfg.num_episodes} episodes "
          f"| epsilon decay {cfg.resolved_epsilon_decay():.6f}/episode")
    order = [cfg.od_filename_pattern.format(hour=n) for n in cfg.od_file_numbers()]
    print(f"Episode hour order: {order[0]} -> {order[1]} -> ... -> {order[-1]}")
    print(f"Reward: S - alpha*O - lambda*C | alpha={cfg.operator_cost_alpha:g}, "
          f"lambda={cfg.transition_penalty_lambda:g} | S=served passengers/total demand, "
          "O=fleet/max fleet, C=route change")
    if cfg.make_plots and not plotting.plotting_available():
        print("matplotlib not installed - plots will be skipped.")
    with (results_dir / "config_used.json").open("w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2, default=str)

    summaries, curves = [], {}
    for rep in range(1, cfg.num_replications + 1):
        out = train_replication(rep, cfg, network, od_matrices, results_dir)
        summaries.append(out["summary"])
        curves[rep] = out["curve"]
        print(f"[rep {rep}] evaluation reward {out['summary']['evaluation_reward']:.6g} | "
              f"unique states {out['summary']['unique_states']:,}")

    with (results_dir / "replication_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    best = max(summaries, key=lambda s: (s["evaluation_reward"], s["mean_training_reward_last_window"]))
    best_info = {
        "criterion": "highest greedy evaluation reward; tie-break: mean training reward of last window",
        "best_replication": best["replication"],
        "random_seed": best["random_seed"],
        "evaluation_reward": best["evaluation_reward"],
    }
    with (results_dir / "best_replication.json").open("w", encoding="utf-8") as fh:
        json.dump(best_info, fh, indent=2)
    best_dir = results_dir / f"replication_{best['replication']}"
    if cfg.copy_best_qtable:
        shutil.copyfile(best_dir / "q_table.csv", results_dir / "best_q_table.csv")
        shutil.copyfile(best_dir / "od_matrices.json", results_dir / "best_od_matrices.json")
        shutil.copyfile(best_dir / "hourly_service_plans.json", results_dir / "best_hourly_service_plans.json")
    if cfg.make_plots:
        plotting.plot_replication_comparison(curves, results_dir / "plots")
    print(f"Best replication: {best['replication']} (evaluation reward {best['evaluation_reward']:.6g}). "
          f"Results in {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
