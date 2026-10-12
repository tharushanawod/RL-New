"""
Q-table CSV save/load (lossless).

One CSV row per visited state-action pair:

    replication, hour, od_matrix, current_route, current_service_plan,
    previous_service_plan, fleet_remaining, action, q_value, visit_count

Nested structures are stored as JSON strings.  JSON is only a storage format:
ints stay ints, floats are written with ``repr`` (exact round-trip) and the
loader rebuilds byte-for-byte identical state keys.

QTABLE_CSV_OD_MODE = "reference" writes ``REF:<hour>`` in the od_matrix
column and stores the full matrices once in ``od_matrices.json`` next to the
CSV; the loader substitutes them back, so this is equally lossless.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..agent.q_learning_agent import QTable, VisitTable
from ..environment.actions import action_from_string
from ..environment.state import json_fields_to_key, key_to_json_fields

COLUMNS = ["replication", "hour", "clock_hour", "od_matrix", "current_route", "current_service_plan",
           "previous_service_plan", "fleet_remaining", "action", "q_value", "visit_count"]

# OD JSON cells can be large; make sure the csv module accepts them on load.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def save_q_table_csv(path: Path, q: QTable, visits: VisitTable, replication: int,
                     od_mode: str = "inline", clock_hours: Optional[Dict[int, int]] = None) -> int:
    """Stream the Q-table to CSV; returns the number of rows written.

    ``hour`` is the episode step t (the state field); ``clock_hour`` is the
    OD file number used at that step (informational, from ``clock_hours``).
    """
    clock_hours = clock_hours or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    od_json_by_hour: Dict[int, str] = {}
    od_by_hour: Dict[int, Any] = {}
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for key, actions in q.items():
            fields = key_to_json_fields(key)
            hour = fields["hour"]
            od_by_hour.setdefault(hour, key[1])
            if od_mode == "reference":
                fields["od_matrix"] = f"REF:{hour}"
            vrow = visits.get(key, {})
            for action, value in actions.items():
                writer.writerow([replication, hour, clock_hours.get(hour, ""), fields["od_matrix"], fields["current_route"],
                                 fields["current_service_plan"], fields["previous_service_plan"],
                                 fields["fleet_remaining"], str(action), repr(float(value)),
                                 vrow.get(action, 0)])
                rows += 1
    with (path.parent / "od_matrices.json").open("w", encoding="utf-8") as fh:
        json.dump({str(h): od for h, od in sorted(od_by_hour.items())}, fh)
    return rows


def load_q_table_csv(path: Path, od_matrices_json: Optional[Path] = None
                     ) -> Tuple[QTable, VisitTable, Optional[int]]:
    """Load a Q-table CSV back into ``(q, visits, replication)``.

    The returned keys are identical to the keys used during training, so the
    tables can be passed straight back into a ``QLearningAgent``::

        q, visits, _ = load_q_table_csv("results/replication_1/q_table.csv")
        agent.q, agent.visits = q, visits
    """
    path = Path(path)
    refs: Dict[str, str] = {}
    ref_file = Path(od_matrices_json) if od_matrices_json else path.parent / "od_matrices.json"
    if ref_file.exists():
        with ref_file.open(encoding="utf-8") as fh:
            refs = {f"REF:{h}": json.dumps(od, separators=(",", ":")) for h, od in json.load(fh).items()}

    q: QTable = {}
    visits: VisitTable = {}
    od_cache: Dict[str, Any] = {}
    replication: Optional[int] = None
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["od_matrix"].startswith("REF:"):
                if row["od_matrix"] not in refs:
                    raise ValueError(f"{row['od_matrix']} not found in {ref_file}")
                row["od_matrix"] = refs[row["od_matrix"]]
            key = json_fields_to_key(row, od_cache)
            action = action_from_string(row["action"])
            q.setdefault(key, {})[action] = float(row["q_value"])
            visits.setdefault(key, {})[action] = int(row["visit_count"])
            replication = int(row["replication"])
    return q, visits, replication


def load_q_table_dataframe(path: Path):
    """Convenience: load the raw CSV with pandas (JSON columns stay strings).

    Use ``json.loads`` on a cell to recover the nested structure.
    """
    import pandas as pd  # optional dependency
    return pd.read_csv(path)
