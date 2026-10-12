"""
Tabular Q-learning agent with a sparse dictionary Q-table.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

* ``q[state_key][action]`` holds Q-values; a state is inserted only when an
  action is actually taken in it (nothing is pre-generated).
* ``visits[state_key][action]`` counts how often each pair was updated.
* Unseen pairs have value INITIAL_Q_VALUE.
* epsilon-greedy selection operates ONLY on the valid (masked) action list,
  greedy ties are broken uniformly at random with the replication's RNG.

The agent knows nothing about transport: it sees hashable state keys and
lists of valid actions.
"""

from __future__ import annotations

import random
from typing import Dict, Hashable, List, Sequence

from ..config import ExperimentConfig
from ..environment.actions import Action

QTable = Dict[Hashable, Dict[Action, float]]
VisitTable = Dict[Hashable, Dict[Action, int]]


class QLearningAgent:
    def __init__(self, cfg: ExperimentConfig, seed: int) -> None:
        self.alpha = cfg.learning_rate
        self.gamma = cfg.discount_factor
        self.epsilon = cfg.initial_epsilon
        self.min_epsilon = cfg.min_epsilon
        self.epsilon_decay = cfg.resolved_epsilon_decay()
        self.q0 = cfg.initial_q_value
        self.rng = random.Random(seed)
        self.q: QTable = {}
        self.visits: VisitTable = {}
        self._num_pairs = 0  # maintained incrementally (O(1) size queries)

    # ------------------------------------------------------------- queries
    def q_value(self, key: Hashable, action: Action) -> float:
        row = self.q.get(key)
        return row.get(action, self.q0) if row else self.q0

    def max_q(self, key: Hashable, valid_actions: Sequence[Action]) -> float:
        row = self.q.get(key)
        if not row:
            return self.q0
        return max(row.get(a, self.q0) for a in valid_actions)

    def greedy_action(self, key: Hashable, valid_actions: Sequence[Action]) -> Action:
        row = self.q.get(key)
        if not row:
            return self.rng.choice(valid_actions)
        values = [row.get(a, self.q0) for a in valid_actions]
        best = max(values)
        ties = [a for a, v in zip(valid_actions, values) if v == best]
        return ties[0] if len(ties) == 1 else self.rng.choice(ties)

    def select_action(self, key: Hashable, valid_actions: Sequence[Action], greedy: bool = False) -> Action:
        """epsilon-greedy over valid actions only (epsilon = 0 if ``greedy``)."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        return self.greedy_action(key, valid_actions)

    # -------------------------------------------------------------- update
    def update(self, key: Hashable, action: Action, reward: float, next_key: Hashable,
               next_valid_actions: Sequence[Action], done: bool) -> float:
        """One Q-learning backup; returns the TD error."""
        target = reward if done or not next_valid_actions else (
            reward + self.gamma * self.max_q(next_key, next_valid_actions))
        row = self.q.setdefault(key, {})
        if action not in row:
            self._num_pairs += 1
        old = row.get(action, self.q0)
        td = target - old
        row[action] = old + self.alpha * td
        vrow = self.visits.setdefault(key, {})
        vrow[action] = vrow.get(action, 0) + 1
        return td

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------- sizes
    @property
    def num_states(self) -> int:
        return len(self.q)

    @property
    def num_state_action_pairs(self) -> int:
        return self._num_pairs
