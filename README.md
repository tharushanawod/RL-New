# Raw-state tabular Q-learning for time-dependent bus network design (Mandl)

The agent builds 24 sequential hourly service plans P_1..P_24 (routes, frequencies,
buses) for Mandl's Swiss network. It uses tabular Q-learning on the **complete raw
state**: no compression, aggregation or discretisation. The point of the experiment
is to measure how badly this raw state space explodes.

## Run

```bash
pip install -r requirements.txt          # numpy required; matplotlib/scipy/psutil/pandas optional
# 1. insert real data (see transit_rl/data/input/README.md)
# 2. full experiment: 5 replications x 10,000 episodes
python -m transit_rl.main
# quick check
python -m transit_rl.main --episodes 50 --replications 1 --log-every 10
# code check on SYNTHETIC random data (not Mandl)
python tests/smoke_test.py
```

## Architecture

| # | Topic | Implementation |
|---|---|---|
| 1 | Episode | One full day of 24 hourly steps, starting at the morning peak: `od_hour_08` → … → `od_hour_23` → `od_hour_00` → … → `od_hour_07` (`OD_START_FILE_HOUR` in config). |
| 2 | RL step | One agent action (`TransitEnvironment.step`). An hour takes many steps. |
| 3 | State | `(t, OD_t, R_current, P_current, P_{t-1}, F_remaining)` (`environment/state.py`). OD is the full 15×15 tuple, the route is the full stop sequence, both plans are full `(route, frequency[, kept])` records. |
| 4 | Action | `Action(type, arg)`: `ADD_STOP(j)`, `REMOVE_LAST_STOP`, `END_ROUTE`, `INCREASE_FREQUENCY(k)`, `DECREASE_FREQUENCY(k)`, `KEEP_ROUTE(k)`, `REMOVE_ROUTE(k)`, `FINISH_PLAN`; `k` = index in the current plan. |
| 5 | Passenger assignment | In default dense mode, evaluate after every action, including feasible tentative routes during `ADD_STOP`/`REMOVE_LAST_STOP`. Legacy modes evaluate operational changes and final plans; construction feedback is optional via `ASSIGN_AFTER_ROUTE_CONSTRUCTION_STEPS`. |
| 6 | OD never reduced | Matrices are loaded read-only. Every assignment starts from the original OD_t plus the current plan. Results go into separate `assigned/unserved/direct/transfer_demand` arrays. |
| 7 | Fleet | `v_k = ceil(f_k·T_k/60)`, `F_remaining = MAX_FLEET − Σv_k`, recomputed from the plan. Removing a route or lowering a frequency returns its buses automatically. Masking blocks increases that would exceed the fleet. |
| 8 | Previous plan | At hour start `previous_plan = P[t−1]`. Plans are immutable tuples (equivalent to deepcopy), so `P[t−1]` can never be changed. At `FINISH_PLAN`, `P[t] = current_plan`. |
| 9 | Transition cost | Hungarian matching on normalised (reversal-aware) Levenshtein distance. `C = C_route` in [0,1]; frequency change is a diagnostic only (`environment/transition_cost.py`). The existing greedy fallback applies without SciPy. |
| 10 | Q-table | `dict[state_key][action] → q` plus `visits[state_key][action] → n`. A state is added only when an action is taken in it. `state_to_key` returns the raw tuples unchanged (lossless). Identical sub-objects are shared by reference to save memory. |
| 11 | ε-greedy | Chooses only from `env.valid_actions()`. Greedy ties are broken randomly with the seeded RNG. |
| 12 | Hour t → t+1 | `FINISH_PLAN` (or the step limit) → evaluate P_t → store the hourly record → `hour += 1` → `previous_plan = P_t`, `current_plan = copy of P_t` (kept flags cleared). The Q-update bootstraps across the hour boundary, so hours are linked through the return. Hour 24 is terminal. |

All modelling assumptions are marked `# ASSUMPTION:` in `transit_rl/config.py`.

The default discount factor is **1.0**. The default `REWARD_MODE="dense"`
returns immediate score differences after every action, including construction
of a feasible tentative route. A single stop, an unchanged service decision,
or committing an already scored tentative route can have zero reward. Tentative
routes do not reserve buses or enter the saved service plan until committed.
At each hourly finish, a correction restores the starting score and removes
any discarded tentative service, so action rewards sum to the final hourly
plan reward. The daily learning return therefore equals the sum of all 24
hourly plan rewards. Legacy `terminal` and `delta` modes remain available;
`delta` alone optimizes differences from hourly starting plans.

## Hourly reward

Passenger service quality is the fraction of demand served, using the
existing passenger-assignment counts:

```text
D_t   = assignment.total_demand
A_t   = assignment.served_passengers
S_t   = A_t / D_t                         if D_t > 0, otherwise 0

v_k   = ceil(f_k * round_trip_minutes_k / 60 - FLEET_ROUNDING_TOLERANCE)
B_t   = sum_k v_k                         (existing fleet calculation)
O_t   = B_t / MAX_FLEET

d(a,b) = min(Levenshtein(a,b), Levenshtein(a,reverse(b))) / max(len(a),len(b))
C_t    = (sum of matched distances + U_prev + U_curr) / (M + U_prev + U_curr)
R_t    = S_t - alpha * O_t - lambda * C_t
```

`M` counts accepted matched route pairs; `U_prev` and `U_curr` count unmatched
routes. Matching still minimises total edit distance using Hungarian matching
when SciPy is installed. Pairs above `ROUTE_MATCH_MAX_DISTANCE` are discarded.
Two empty plans have `C_t = 0`; the first hour also has `C_t = 0` by default
(`APPLY_TRANSITION_COST_FIRST_HOUR`). Reversal handling remains configurable.

There are no separate unserved, transfer, overload, direct-service, route
operating time, or frequency-change reward terms. Assignment still uses its
transfer penalty to choose paths. Travel time (in-vehicle + waiting
passenger-minutes) is reported as a diagnostic and does not enter the reward.
Round-trip times still determine fleet use and remain in hourly JSON diagnostics.
Frequency changes can still affect waiting time, assignment and fleet use;
they affect reward through served counts and fleet use only.

`OPERATOR_COST_ALPHA` / `ExperimentConfig.operator_cost_alpha` and
`TRANSITION_PENALTY_LAMBDA` / `ExperimentConfig.transition_penalty_lambda`
both default to `0.3` and accept finite, nonnegative values. Alpha is distinct
from the Q-learning `LEARNING_RATE`. Command-line overrides:

```bash
python -m transit_rl.main --alpha 0.3 --lambda 0.3
python -m unittest discover -s tests -v
python tests/smoke_test.py
```

### Numerical scales and interpretation

For `D=100` and `A=80`, `S=80/100=0.8`, regardless of total travel time.
If `B=10`, `MAX_FLEET=100`, and `C=0.2`, then `O=0.1` and, with the default
coefficients, `R=0.8-0.3*0.1-0.3*0.2=0.71`. The fleet and transition
penalties are `0.03` and `0.06`. For the same assignment with one bus and no
route change, `R=0.797`. Serving 40 out of 100 gives `S=0.4`; serving 160 out
of 200 gives `S=0.8`.
An empty, unserved plan has `S=0`; fleet and route-change costs still apply
when present. Zero demand gives `S=0`. Zero travel time does not change the
served-demand ratio and requires no reward division guard.

`S`, `O` and `C` now all lie in `[0,1]`; removing the division by passenger
travel time removes the previous service-scale suppression. The coefficients
still control the tradeoffs: low coverage or sufficiently large coefficients
can produce negative rewards. The retained defaults are not calibrated by
performance experiments. No improvement in transit service or learning
performance is established by correctness tests; that requires controlled
experiments.

### Code review and output changes

Before modifying the reward, all Python modules, tests, configuration, CLI,
data-loading instructions and output helpers were reviewed. The former score
combined a direct-service bonus with normalised generalised time (including
imputed time for unserved demand), separate unserved/transfer/overload penalties,
weighted fleet and route-time costs, and weighted route/frequency transitions.

The raw six-field state, eight action types and their masks, Q-learning backup,
epsilon schedule, 24-hour OD ordering, immutable OD matrices and plans,
assignment path enumeration/common-lines splitting/cache, fleet rounding and
frequency fallback are preserved. Capacity retains its existing interpretation:
assignment is uncapacitated and reports peak overload relative to
`BUS_CAPACITY * frequency`; it does not enforce a hard boarding limit. Overload
is now diagnostic only, as required by removing its reward penalty.

Hourly CSV/JSON expose `served_passengers`, `served_demand_ratio`, raw
`service_component=S_t`, raw `operator_component=O_t`, weighted
`operator_penalty=alpha*O_t`, raw `transition_cost=C_t`, and weighted
`transition_component=lambda*C_t`. Thus `operator_component` now means the
raw fleet fraction rather than the former weighted mixed operator cost.
`transition_frequency_change` is retained explicitly as a diagnostic.

Learning curves and replication summaries expose total passenger travel time,
the demand-weighted daily served ratio, and sums of hourly raw components and
weighted penalties. Daily reward is `sum_t R_t`, not a reward recomputed from
daily demand/time totals. Progress logs show window means of daily `S_t` sums,
`alpha*O_t` sums and `lambda*C_t` sums. Console output uses six significant
digits; CSV/JSON retain float precision. Saved configuration includes the new
coefficients and omits obsolete reward weights/normalisers. Historical reward
CSV/config consumers must account for these changed fields and meanings.
Q-table serialization is unchanged.

| Modified file | Change |
|---|---|
| `transit_rl/environment/reward.py` | Uses served count / total demand with zero-demand handling; separates raw cost from weighted penalties and reports travel time diagnostically. |
| `transit_rl/config.py` | Replaces obsolete reward and transition weights with alpha; retains lambda, validates both, and corrects the capacity comment. |
| `transit_rl/environment/transition_cost.py` | Makes total transition cost equal route change; preserves edit distance, matching and frequency diagnostics. |
| `transit_rl/environment/transit_environment.py` | Passes existing assignment and fleet totals to the new reward API; removes the route-time reward argument. |
| `transit_rl/experiment/trainer.py` | Exports assignment inputs and reward components, aggregates hourly scores, and logs component scales. |
| `transit_rl/main.py` | Adds coefficient CLI overrides and prints the reward definition with suitable numeric precision. |
| `tests/test_reward.py` | Adds hand-calculated assignment/reward examples, zero guards, matching, coefficient, fleet-mask and export checks. |
| `tests/test_dense_rewards.py` | Checks feedback and conservation; unchanged coverage and fleet give zero reward for a frequency-only change. |
| `tests/smoke_test.py` | Checks the 24-hour formulas and new outputs; saves records before reset so fleet/linkage assertions actually execute. |
| `README.md` | Documents the review, final formulas, numerical scales, compatibility and every modified file. |

## Outputs (`transit_rl/results/`)

```
config_used.json, replication_summary.csv, best_replication.json,
best_q_table.csv, best_hourly_service_plans.json, plots/
replication_<r>/
    q_table.csv, od_matrices.json        # load with utils.qtable_io.load_q_table_csv
    learning_curve.csv                   # per episode
    progress_log.csv                     # every LOG_EVERY episodes, incl. memory
    hourly_service_plans.csv / .json     # greedy evaluation episode
    state_space_statistics.json          # visit counts, revisits, size estimate
    plots/*.png                          # if matplotlib installed
```

Best replication = highest greedy evaluation reward (tie-break: mean training reward
over the last logging window).

## Resource warning

On synthetic test data, an episode takes about 0.3 s and creates about 750 new
states. A full replication is therefore about 1 h and about 7–8 M Q-table rows.
Expect several GB of RAM per replication. With `QTABLE_CSV_OD_MODE="inline"`
(the default, as specified) expect about 1.4 KB per CSV row, i.e. roughly 10 GB of
CSV per replication. `--od-mode reference` stores each OD matrix once
(`REF:<hour>` + `od_matrices.json`). It is equally lossless and about 5× smaller.
