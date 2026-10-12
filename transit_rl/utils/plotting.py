"""
Learning-curve plots (matplotlib is optional; plots are skipped without it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PLOTS = [
    ("total_reward", "Total daily reward", "reward_vs_episode.png"),
    ("unique_states", "Unique states encountered", "unique_states_vs_episode.png"),
    ("q_table_size", "Q-table size (state-action pairs)", "q_table_size_vs_episode.png"),
    ("total_transition_cost", "Total daily transition cost", "transition_cost_vs_episode.png"),
    ("average_fleet_used", "Average hourly fleet used (buses)", "fleet_usage_vs_episode.png"),
]


def plotting_available() -> bool:
    return plt is not None


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([np.full(window - 1, np.nan), (c[window:] - c[:-window]) / window])


def plot_learning_curves(rows: List[Dict[str, float]], out_dir: Path, title_prefix: str = "") -> List[Path]:
    if plt is None or not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.array([r["episode"] for r in rows])
    window = max(1, len(rows) // 100)
    written = []
    for column, label, filename in PLOTS:
        y = np.array([r[column] for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(episodes, y, lw=0.6, alpha=0.4, label="per episode")
        if window > 1 and column in ("total_reward", "total_transition_cost", "average_fleet_used"):
            ax.plot(episodes, _moving_average(y, window), lw=1.6, label=f"moving avg ({window})")
            ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.set_title(f"{title_prefix}{label} vs episode")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def plot_replication_comparison(curves: Dict[int, Sequence[Dict[str, float]]], out_dir: Path) -> List[Path]:
    if plt is None or not curves:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for column, label, filename in PLOTS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rep, rows in sorted(curves.items()):
            ep = np.array([r["episode"] for r in rows])
            y = np.array([r[column] for r in rows], dtype=float)
            w = max(1, len(rows) // 100)
            ax.plot(ep, _moving_average(y, w) if column not in ("unique_states", "q_table_size") else y,
                    lw=1.2, label=f"replication {rep}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: all replications")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"all_replications_{filename}"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
