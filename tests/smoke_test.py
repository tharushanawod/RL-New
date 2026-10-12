"""
End-to-end smoke test on SYNTHETIC data.

!!! The network and OD matrices generated here are RANDOM TEST FIXTURES. !!!
!!! They are NOT Mandl's network and NOT real demand.  They exist only   !!!
!!! to check that the code runs; never report results from them.        !!!

Run:  python tests/smoke_test.py
"""

from __future__ import annotations

import dataclasses
import csv
import json
import math
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from transit_rl import main as entry  # noqa: E402
from transit_rl.config import ExperimentConfig  # noqa: E402
from transit_rl.data.network import load_network_from_csv  # noqa: E402
from transit_rl.data.od_loader import load_od_matrices  # noqa: E402
from transit_rl.environment.state import state_to_key, key_to_state  # noqa: E402
from transit_rl.environment.transit_environment import TransitEnvironment  # noqa: E402
from transit_rl.agent.q_learning_agent import QLearningAgent  # noqa: E402
from transit_rl.experiment.trainer import run_episode  # noqa: E402
from transit_rl.utils.qtable_io import load_q_table_csv, save_q_table_csv  # noqa: E402


def write_synthetic_data(folder: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    links = {(i, i + 1) for i in range(1, 15)}  # a path guarantees connectivity
    while len(links) < 21:
        a, b = sorted(rng.sample(range(1, 16), 2))
        links.add((a, b))
    (folder / "od").mkdir(parents=True)
    with (folder / "mandl_links.csv").open("w") as fh:
        fh.write("# SYNTHETIC TEST FIXTURE - NOT MANDL DATA\nfrom_stop,to_stop,travel_time\n")
        for a, b in sorted(links):
            fh.write(f"{a},{b},{rng.randint(2, 10)}\n")
    nrng = np.random.default_rng(seed)
    for h in range(1, 25):
        m = nrng.integers(0, 60, size=(15, 15))
        np.fill_diagonal(m, 0)
        np.savetxt(folder / "od" / f"od_hour_{h - 1:02d}.csv", m, fmt="%d", delimiter=",")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        results = Path(tmp) / "results"
        write_synthetic_data(data)

        cfg = dataclasses.replace(ExperimentConfig(), data_dir=data, num_episodes=30)
        net = load_network_from_csv(cfg.links_path, 15, True)
        ods = load_od_matrices(cfg.od_dir, cfg.od_filename_pattern, 24, 15, cfg.od_file_numbers())
        env = TransitEnvironment(net, ods, cfg)

        # OD matrices must stay untouched by training.
        before = [m.copy() for m in ods]
        agent = QLearningAgent(cfg, seed=1)
        for _ in range(5):
            run_episode(env, agent, train=True)
        trained_records = env.hourly_records
        assert all(np.array_equal(a, b) for a, b in zip(before, ods)), "OD matrix was modified"

        # Lossless state key round trip.
        s = env.reset()
        assert state_to_key(key_to_state(state_to_key(s))) == state_to_key(s)

        # Lossless CSV round trip of the whole Q-table (both OD modes).
        for mode in ("inline", "reference"):
            path = Path(tmp) / mode / "q_table.csv"
            save_q_table_csv(path, agent.q, agent.visits, 1, mode)
            q, v, _ = load_q_table_csv(path)
            assert q == agent.q and v == agent.visits, f"Q-table CSV round trip failed ({mode})"

        # Fleet accounting and consecutive-plan linkage.
        assert len(trained_records) == 24
        assert [r["clock_hour"] for r in trained_records] == cfg.od_file_numbers()
        for r in trained_records:
            assert r["total_fleet_used"] <= cfg.max_fleet
            assert math.isclose(r["operator_component"], r["total_fleet_used"] / cfg.max_fleet)
            expected_s = r["served_passengers"] / r["total_demand"] if r["total_demand"] > 0 else 0
            assert math.isclose(r["service_component"], expected_s)
            assert r["transition_cost"] == r["transition_route_change"]
            assert math.isclose(r["total_reward"], expected_s - cfg.operator_cost_alpha * r["operator_component"]
                                - cfg.transition_penalty_lambda * r["transition_cost"], abs_tol=1e-12)
        for prev, cur in zip(trained_records, trained_records[1:]):
            assert cur["previous_service_plan"] == prev["service_plan"], "P_{t-1} not preserved"

        # Full pipeline with 2 short replications.
        code = entry.main(["--episodes", "30", "--replications", "2", "--data-dir", str(data),
                           "--results-dir", str(results), "--log-every", "10",
                           "--alpha", "0.002", "--lambda", "0.004"])
        assert code == 0
        for name in ("replication_summary.csv", "best_replication.json", "best_q_table.csv"):
            assert (results / name).exists(), name
        used = json.loads((results / "config_used.json").read_text())
        assert used["operator_cost_alpha"] == 0.002
        assert used["transition_penalty_lambda"] == 0.004
        assert "w_service_unserved" not in used and "frequency_change_weight" not in used
        for rep in (1, 2):
            d = results / f"replication_{rep}"
            for name in ("q_table.csv", "learning_curve.csv", "hourly_service_plans.csv",
                         "hourly_service_plans.json", "state_space_statistics.json"):
                assert (d / name).exists(), f"{d / name} missing"
            for filename, service, operator, transition, reward in (
                ("hourly_service_plans.csv", "service_component", "operator_penalty", "transition_component", "total_reward"),
                ("learning_curve.csv", "total_service_component", "total_operator_penalty", "total_transition_component", "daily_plan_reward"),
            ):
                with (d / filename).open(newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        assert math.isclose(float(row[reward]), float(row[service]) - float(row[operator])
                                            - float(row[transition]), abs_tol=1e-12)
        print((results / "replication_summary.csv").read_text())
        print((results / "replication_1" / "state_space_statistics.json").read_text())
    print("SMOKE TEST PASSED (synthetic data only)")


if __name__ == "__main__":
    main()
