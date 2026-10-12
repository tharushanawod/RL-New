"""
Loading and validation of the 24 hourly OD matrices.

The real matrices must be supplied by the researcher as
``data/input/od/OD_1.csv`` ... ``OD_24.csv`` (15 rows x 15 comma-separated
values, no header; row = origin, column = destination).

The matrices are returned UNCHANGED.  No aggregation, scaling, rounding or
discretisation is applied.  Values that are all whole numbers are stored as
integers (e.g. ``880.0`` in the file -> ``880``), which is value-preserving;
otherwise floats are kept exactly as parsed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

from .network import PlaceholderDataError

Number = Union[int, float]
ODKey = Tuple[Tuple[Number, ...], ...]


def _read_matrix(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(v) for v in stripped.replace(";", ",").split(",") if v.strip() != ""])
    if not rows:
        raise PlaceholderDataError(f"PLACEHOLDER: OD file {path} is empty. Insert the real OD matrix.")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: rows have different lengths {sorted(widths)}")
    return np.array(rows, dtype=float)


def validate_od_matrix(matrix: np.ndarray, num_stops: int, label: str) -> None:
    if matrix.shape != (num_stops, num_stops):
        raise ValueError(f"{label} must be {num_stops}x{num_stops}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    if np.any(matrix < 0):
        raise ValueError(f"{label} contains negative demand")


def load_od_matrices(od_dir: Path, pattern: str, num_hours: int, num_stops: int,
                     file_numbers: Sequence[int]) -> List[np.ndarray]:
    """Return ``num_hours`` matrices in EPISODE order.

    Episode step t (1-based) uses file ``pattern.format(hour=file_numbers[t-1])``,
    e.g. file_numbers = [8, ..., 23, 0, ..., 7] -> step 1 = od_hour_08.csv.
    Only the order of the matrices changes; their values are untouched.
    """
    od_dir = Path(od_dir)
    matrices: List[np.ndarray] = []
    if len(file_numbers) != num_hours or len(set(file_numbers)) != num_hours:
        raise ValueError(f"Need {num_hours} distinct OD file numbers, got {list(file_numbers)}")

    def file_for(hour: int) -> Path:
        return od_dir / pattern.format(hour=file_numbers[hour - 1])

    missing = [file_for(h).name for h in range(1, num_hours + 1) if not file_for(h).exists()]
    if missing:
        raise PlaceholderDataError(
            f"PLACEHOLDER: OD files missing in {od_dir}: {missing}. "
            f"Expected files named like {file_for(1).name}."
        )
    for hour in range(1, num_hours + 1):
        path = file_for(hour)
        m = _read_matrix(path)
        validate_od_matrix(m, num_stops, f"step {hour} ({path.name})")
        if np.all(m == np.round(m)):
            m = m.astype(np.int64)
        m.setflags(write=False)  # the OD matrix must never be modified
        matrices.append(m)
    return matrices


def od_to_key(matrix: np.ndarray) -> ODKey:
    """Lossless conversion of a matrix to a hashable tuple-of-tuples.

    This conversion exists only to make the raw state hashable.
    No state information is removed.
    """
    return tuple(tuple(v.item() for v in row) for row in matrix)


def od_key_to_array(key: ODKey) -> np.ndarray:
    return np.array(key)
