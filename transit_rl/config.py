"""
Central experiment configuration.

Every tunable parameter and every transport-modelling assumption lives here.
Lines marked ``# ASSUMPTION:`` are modelling decisions that were NOT given by
the experiment specification and must be reviewed by the researcher.

The module exposes plain UPPER_CASE constants (easy to edit) and an immutable
``ExperimentConfig`` dataclass built from them.  Components receive a config
object explicitly, so there is no global mutable state; command-line overrides
create a modified copy via ``dataclasses.replace``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent

# =============================================================================
# 1. Q-LEARNING PARAMETERS
# =============================================================================
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1.0
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
NUM_EPISODES = 10000
NUM_REPLICATIONS = 5

# ASSUMPTION: epsilon decays multiplicatively once per EPISODE and reaches
# MIN_EPSILON after EPSILON_DECAY_FRACTION of all episodes.  Set EPSILON_DECAY
# to an explicit float to override the computed value.
EPSILON_DECAY_FRACTION = 0.8
EPSILON_DECAY: Optional[float] = None  # None -> computed from the line above

# ASSUMPTION: unseen state-action pairs have Q = 0 (optimistic relative to the
# mostly negative rewards, which mildly encourages trying untested actions).
INITIAL_Q_VALUE = 0.0

# Replication r (1-based) uses seed BASE_RANDOM_SEED + r - 1.
BASE_RANDOM_SEED = 2024

# =============================================================================
# 2. NETWORK / DATA
# =============================================================================
NUM_STOPS = 15
NUM_HOURS = 24
DATA_DIR = PACKAGE_DIR / "data" / "input"
LINKS_FILENAME = "mandl_links.csv"          # columns: from_stop,to_stop,travel_time
OD_SUBDIR = "od"
OD_FILENAME_PATTERN = "od_hour_{hour:02d}.csv"  # 15 rows x 15 comma-separated values
# Smallest number used in the OD file names (0 -> files od_hour_00 .. od_hour_23).
# The file number is treated as the clock hour (od_hour_08 = 08:00-09:00).
OD_FILE_HOUR_BASE = 0
# Clock hour (file number) of the FIRST hour of every episode.  The episode
# then runs forward and wraps around midnight:
#     8 -> od_hour_08, 09, ..., 23, 00, 01, ..., 07
# Rationale: the high-demand morning peak lets the agent build a meaningful
# network first, which later hours then adapt.  Episode step t = 1..24 is the
# position in this sequence (state field ``hour``); the clock hour is reported
# alongside it in all outputs.  There is no transition cost between the last
# step (07:00) and the first step (08:00), because each episode is one pass.
OD_START_FILE_HOUR = 8

# Stop numbering used in YOUR input files (1 -> stops are 1..15, 0 -> 0..14).
# Internally the program always uses 1..15.
INPUT_STOP_ID_BASE = 1

# ASSUMPTION: each link listed once is usable in both directions with the same
# travel time.  If both directions are listed, each direction keeps its own time.
LINKS_ARE_BIDIRECTIONAL = True

# =============================================================================
# 3. FLEET AND FREQUENCY
# =============================================================================
MAX_FLEET = 99
BUS_CAPACITY = 40            # seats per bus
BUS_LOAD_FACTOR = 1.25       # maximum passengers / seats: 40 * 1.25 = 50 per bus

MIN_FREQUENCY = 1.0          # ASSUMPTION: buses/hour
MAX_FREQUENCY = 12.0         # ASSUMPTION: buses/hour (5-minute headway)
FREQUENCY_STEP = 1.0         # ASSUMPTION: buses/hour per INCREASE/DECREASE action
INITIAL_ROUTE_FREQUENCY = 4.0  # ASSUMPTION: frequency given by END_ROUTE

# ASSUMPTION: if INITIAL_ROUTE_FREQUENCY does not fit in the remaining fleet,
# END_ROUTE uses the highest frequency on the grid (>= MIN_FREQUENCY) that fits.
# If even MIN_FREQUENCY does not fit, END_ROUTE is masked as invalid.
END_ROUTE_FREQUENCY_FALLBACK = True

# ASSUMPTION: round-trip time T_k = forward time + backward time + 2 * layover.
TERMINAL_LAYOVER_MINUTES = 0.0

# ASSUMPTION (rounding rule): buses v_k = ceil(f_k * T_k / 60).  A fractional
# vehicle cannot be operated, so we always round UP (a tiny tolerance avoids
# float artefacts such as 4.0000000001 -> 5).
FLEET_ROUNDING_TOLERANCE = 1e-9

# =============================================================================
# 4. ROUTE CONSTRUCTION RULES
# =============================================================================
MIN_ROUTE_LENGTH = 2         # ASSUMPTION: stops (a route needs at least one link)
MAX_ROUTE_LENGTH = 8         # ASSUMPTION: stops
MAX_ROUTES = 8               # ASSUMPTION: max routes in one hourly plan
ALLOW_REPEATED_STOPS = False  # ASSUMPTION: no loops inside a route
# ASSUMPTION: a route identical to an existing one (or its reverse, because
# routes are bidirectional) cannot be committed - use INCREASE_FREQUENCY instead.
ALLOW_DUPLICATE_ROUTES = False

# ASSUMPTION: buses run in both directions of every route.
ROUTES_ARE_BIDIRECTIONAL = True

# ASSUMPTION: at the start of hour t the current plan is initialised as a copy
# of P_{t-1} ("copy_previous") so the agent ADAPTS it (keep / modify / remove /
# add).  "empty" makes the agent build every hour from scratch.
CURRENT_PLAN_START_MODE = "copy_previous"

# ASSUMPTION: hour-1 "previous plan" P_0.  Empty by default.  To use a real
# initial plan, insert records such as {"route": [1, 2, 3], "frequency": 4.0}.
INITIAL_SERVICE_PLAN: Tuple[dict, ...] = ()

# ASSUMPTION: an hour is forcibly finished after this many RL steps (prevents
# endless INCREASE/DECREASE cycles).  An unfinished route is discarded, then
# FINISH_PLAN is executed automatically.  The step counter is NOT part of the
# state (the specification fixes the state contents).
MAX_STEPS_PER_HOUR = 60

# =============================================================================
# 5. PASSENGER ASSIGNMENT
# =============================================================================
# ASSUMPTION: expected wait = WAITING_TIME_FACTOR * headway = 0.5 * 60 / f
# (random passenger arrivals, regular headways).
WAITING_TIME_FACTOR = 0.5
TRANSFER_PENALTY_MINUTES = 5.0        # ASSUMPTION: per transfer
ENABLE_TWO_TRANSFER_PATHS = True      # ASSUMPTION
# ASSUMPTION: "min_transfers_first" = use a direct path if any exists, else a
# 1-transfer path, else a 2-transfer path (classical Mandl-style hierarchy).
# "min_generalised_time" = pick the category with the lowest generalised time.
PATH_CHOICE_RULE = "min_transfers_first"
# ASSUMPTION: direct routes whose in-vehicle time is within this tolerance of
# the fastest direct route form a "common-lines" set; demand is split among
# them in proportion to frequency and they share a combined waiting time.
DIRECT_PATH_TOLERANCE_MINUTES = 2.0
# ASSUMPTION: trips whose generalised time exceeds this are left unserved
# (None = no limit).
MAX_ACCEPTABLE_TRIP_MINUTES: Optional[float] = None
# ASSUMPTION: OD diagonal (i -> i) is not a bus trip and is ignored.
IGNORE_OD_DIAGONAL = True
# Hourly capacity per route segment and direction is frequency * BUS_CAPACITY
# * BUS_LOAD_FACTOR. Boarding requires space on every leg; unmet demand is
# unserved. Seats can be reused after passengers alight.
# In terminal/delta modes, assignment during route construction is optional.
# Dense mode always evaluates feasible tentative routes for immediate feedback.
ASSIGN_AFTER_ROUTE_CONSTRUCTION_STEPS = False
# Re-run assignment after every meaningful operational change (spec default).
EVALUATE_AFTER_EACH_CHANGE = True
# LRU cache size for assignment results keyed by (hour, routes, frequencies).
# This is a pure speed-up: identical inputs always give identical outputs.
ASSIGNMENT_CACHE_SIZE = 20000

# =============================================================================
# 6. TRANSITION COST
# =============================================================================
# ASSUMPTION: previous and current routes are paired by minimum-cost bipartite
# matching on normalised edit distance ("hungarian"; falls back to "greedy"
# if SciPy is not installed).
ROUTE_MATCHING_METHOD = "hungarian"
# ASSUMPTION: a matched pair with normalised distance above this value is
# treated as "old route removed + new route added".
ROUTE_MATCH_MAX_DISTANCE = 0.75
# ASSUMPTION: a route and its reverse are the same bidirectional service.
CONSIDER_ROUTE_REVERSAL = True
# ASSUMPTION: no transition penalty in hour 1 (P_0 is artificial).
APPLY_TRANSITION_COST_FIRST_HOUR = False

# =============================================================================
# 7. REWARD
# =============================================================================
# "terminal": reward 0 for intermediate steps, R(P_t) when the hour's plan is
#             finished.  Daily return = sum_t R(P_t) (the true objective).
# "delta":    after each operational change reward = R(new plan) - R(old plan)
#             (denser; the hour's rewards telescope to R(P_t) - R(start plan)).
# "dense":    after every action reward = change in evaluated plan score,
#             including feasible tentative routes. At hour end, reconcile to
#             the committed plan and add the starting score. With gamma = 1,
#             hourly rewards sum exactly to R(P_t), preserving the objective.
REWARD_MODE = "dense"  # ASSUMPTION

# S_t = served_passengers / total_demand (actual assignment counts).
# ASSUMPTION: S_t = 0 for zero demand; travel time is diagnostic only.
# O_t = buses_used / MAX_FLEET; C_t = route_change (frequency diagnostic only).
# R_t = S_t - alpha * O_t - lambda * C_t. No additional reward terms.
# ASSUMPTION: preserve the former fleet coefficient as alpha and the former
# lambda default. S_t, O_t and C_t are all in [0, 1]; alpha and lambda control
# the cost tradeoffs. Defaults are not calibrated by performance experiments.
OPERATOR_COST_ALPHA = 0.3
TRANSITION_PENALTY_LAMBDA = 0.3

# Invalid actions are always masked; if one is nevertheless submitted the
# environment raises an error (safe default) instead of silently continuing.
RAISE_ON_INVALID_ACTION = True

# =============================================================================
# 8. LOGGING / OUTPUT
# =============================================================================
LOG_EVERY = 100
RESULTS_DIR = PACKAGE_DIR / "results"
MAKE_PLOTS = True
# "inline": every Q-table CSV row carries the full OD matrix as JSON (spec).
# "reference": the od_matrix cell holds "REF:<hour>" and the full matrices are
#              stored once in od_matrices.json (still lossless, much smaller).
QTABLE_CSV_OD_MODE = "inline"
COPY_BEST_QTABLE = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable bundle of all parameters (defaults = module constants)."""

    # Q-learning
    learning_rate: float = LEARNING_RATE
    discount_factor: float = DISCOUNT_FACTOR
    initial_epsilon: float = INITIAL_EPSILON
    min_epsilon: float = MIN_EPSILON
    epsilon_decay: Optional[float] = EPSILON_DECAY
    epsilon_decay_fraction: float = EPSILON_DECAY_FRACTION
    num_episodes: int = NUM_EPISODES
    num_replications: int = NUM_REPLICATIONS
    initial_q_value: float = INITIAL_Q_VALUE
    base_random_seed: int = BASE_RANDOM_SEED

    # network / data
    num_stops: int = NUM_STOPS
    num_hours: int = NUM_HOURS
    data_dir: Path = DATA_DIR
    links_filename: str = LINKS_FILENAME
    od_subdir: str = OD_SUBDIR
    od_filename_pattern: str = OD_FILENAME_PATTERN
    od_file_hour_base: int = OD_FILE_HOUR_BASE
    od_start_file_hour: int = OD_START_FILE_HOUR
    input_stop_id_base: int = INPUT_STOP_ID_BASE
    links_are_bidirectional: bool = LINKS_ARE_BIDIRECTIONAL

    # fleet / frequency
    max_fleet: int = MAX_FLEET
    bus_capacity: int = BUS_CAPACITY
    bus_load_factor: float = BUS_LOAD_FACTOR
    min_frequency: float = MIN_FREQUENCY
    max_frequency: float = MAX_FREQUENCY
    frequency_step: float = FREQUENCY_STEP
    initial_route_frequency: float = INITIAL_ROUTE_FREQUENCY
    end_route_frequency_fallback: bool = END_ROUTE_FREQUENCY_FALLBACK
    terminal_layover_minutes: float = TERMINAL_LAYOVER_MINUTES
    fleet_rounding_tolerance: float = FLEET_ROUNDING_TOLERANCE

    # route construction
    min_route_length: int = MIN_ROUTE_LENGTH
    max_route_length: int = MAX_ROUTE_LENGTH
    max_routes: int = MAX_ROUTES
    allow_repeated_stops: bool = ALLOW_REPEATED_STOPS
    allow_duplicate_routes: bool = ALLOW_DUPLICATE_ROUTES
    routes_are_bidirectional: bool = ROUTES_ARE_BIDIRECTIONAL
    current_plan_start_mode: str = CURRENT_PLAN_START_MODE
    initial_service_plan: Tuple[dict, ...] = field(default=INITIAL_SERVICE_PLAN)
    max_steps_per_hour: int = MAX_STEPS_PER_HOUR

    # passenger assignment
    waiting_time_factor: float = WAITING_TIME_FACTOR
    transfer_penalty_minutes: float = TRANSFER_PENALTY_MINUTES
    enable_two_transfer_paths: bool = ENABLE_TWO_TRANSFER_PATHS
    path_choice_rule: str = PATH_CHOICE_RULE
    direct_path_tolerance_minutes: float = DIRECT_PATH_TOLERANCE_MINUTES
    max_acceptable_trip_minutes: Optional[float] = MAX_ACCEPTABLE_TRIP_MINUTES
    ignore_od_diagonal: bool = IGNORE_OD_DIAGONAL
    assign_after_route_construction_steps: bool = ASSIGN_AFTER_ROUTE_CONSTRUCTION_STEPS
    evaluate_after_each_change: bool = EVALUATE_AFTER_EACH_CHANGE
    assignment_cache_size: int = ASSIGNMENT_CACHE_SIZE

    # transition cost
    route_matching_method: str = ROUTE_MATCHING_METHOD
    route_match_max_distance: float = ROUTE_MATCH_MAX_DISTANCE
    consider_route_reversal: bool = CONSIDER_ROUTE_REVERSAL
    apply_transition_cost_first_hour: bool = APPLY_TRANSITION_COST_FIRST_HOUR

    # reward
    reward_mode: str = REWARD_MODE
    operator_cost_alpha: float = OPERATOR_COST_ALPHA
    transition_penalty_lambda: float = TRANSITION_PENALTY_LAMBDA
    raise_on_invalid_action: bool = RAISE_ON_INVALID_ACTION

    # logging / output
    log_every: int = LOG_EVERY
    results_dir: Path = RESULTS_DIR
    make_plots: bool = MAKE_PLOTS
    qtable_csv_od_mode: str = QTABLE_CSV_OD_MODE
    copy_best_qtable: bool = COPY_BEST_QTABLE

    # ------------------------------------------------------------------ helpers
    def resolved_epsilon_decay(self) -> float:
        """Per-episode multiplicative decay factor."""
        if self.epsilon_decay is not None:
            return float(self.epsilon_decay)
        decay_episodes = max(1.0, self.epsilon_decay_fraction * self.num_episodes)
        return (self.min_epsilon / self.initial_epsilon) ** (1.0 / decay_episodes)

    def seed_for_replication(self, replication: int) -> int:
        """Replication numbers are 1-based."""
        return self.base_random_seed + replication - 1

    def od_file_number(self, step: int) -> int:
        """OD file number (= clock hour) used at episode step ``step`` (1-based)."""
        offset = self.od_start_file_hour - self.od_file_hour_base
        return self.od_file_hour_base + (offset + step - 1) % self.num_hours

    def od_file_numbers(self) -> list:
        """File numbers in episode order, e.g. [8, 9, ..., 23, 0, ..., 7]."""
        return [self.od_file_number(t) for t in range(1, self.num_hours + 1)]

    @property
    def links_path(self) -> Path:
        return Path(self.data_dir) / self.links_filename

    @property
    def od_dir(self) -> Path:
        return Path(self.data_dir) / self.od_subdir

    def validate(self) -> None:
        """Fail fast on inconsistent settings."""
        checks = [
            (0 < self.learning_rate <= 1, "learning_rate must be in (0, 1]"),
            (0 <= self.discount_factor <= 1, "discount_factor must be in [0, 1]"),
            (0 <= self.min_epsilon <= self.initial_epsilon <= 1, "need 0 <= min_eps <= init_eps <= 1"),
            (self.num_episodes >= 1 and self.num_replications >= 1, "episodes/replications >= 1"),
            (0 < self.min_frequency <= self.initial_route_frequency <= self.max_frequency,
             "need 0 < MIN_FREQUENCY <= INITIAL_ROUTE_FREQUENCY <= MAX_FREQUENCY"),
            (self.frequency_step > 0, "frequency_step must be > 0"),
            (2 <= self.min_route_length <= self.max_route_length, "need 2 <= MIN_ROUTE_LENGTH <= MAX_ROUTE_LENGTH"),
            (self.max_routes >= 1, "max_routes >= 1"),
            (self.max_fleet >= 1 and self.bus_capacity >= 1, "fleet and capacity must be >= 1"),
            (math.isfinite(self.bus_load_factor) and self.bus_load_factor > 0,
             "bus_load_factor must be finite and > 0"),
            (self.current_plan_start_mode in ("copy_previous", "empty"), "bad CURRENT_PLAN_START_MODE"),
            (self.path_choice_rule in ("min_transfers_first", "min_generalised_time"), "bad PATH_CHOICE_RULE"),
            (self.route_matching_method in ("hungarian", "greedy"), "bad ROUTE_MATCHING_METHOD"),
            (self.reward_mode in ("terminal", "delta", "dense"), "bad REWARD_MODE"),
            (math.isfinite(self.operator_cost_alpha) and self.operator_cost_alpha >= 0,
             "operator_cost_alpha must be finite and >= 0"),
            (math.isfinite(self.transition_penalty_lambda) and self.transition_penalty_lambda >= 0,
             "transition_penalty_lambda must be finite and >= 0"),
            (self.reward_mode != "dense" or self.discount_factor == 1.0,
             "dense rewards require discount_factor = 1 to preserve the daily objective"),
            (self.qtable_csv_od_mode in ("inline", "reference"), "bad QTABLE_CSV_OD_MODE"),
            (self.max_steps_per_hour >= 1, "max_steps_per_hour >= 1"),
            (self.input_stop_id_base in (0, 1), "input_stop_id_base must be 0 or 1"),
            (self.od_file_hour_base <= self.od_start_file_hour < self.od_file_hour_base + self.num_hours,
             "OD_START_FILE_HOUR must be one of the OD file numbers"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(f"Invalid configuration: {message}")
