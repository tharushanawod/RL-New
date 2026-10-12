"""
Mandl's Swiss network: loading and validation.

The real network data must be supplied by the researcher in
``data/input/mandl_links.csv`` (see ``data/input/README.md``).  No link or
travel-time value is hard-coded or invented here.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class PlaceholderDataError(RuntimeError):
    """Raised when the real input data has not been inserted yet."""


@dataclass(frozen=True)
class TransitNetwork:
    """Road/stop network used for route construction.

    Stops are numbered 1..num_stops internally.

    Attributes:
        num_stops: number of stops (15 for Mandl).
        travel_time: (n, n) array of link travel times in minutes;
            ``travel_time[i-1, j-1]`` is the time from stop i to stop j,
            ``np.inf`` where no physical link exists, 0 on the diagonal.
        neighbours: ``neighbours[i-1]`` = sorted tuple of stops reachable from
            stop i by one physical link.
    """

    num_stops: int
    travel_time: np.ndarray
    neighbours: Tuple[Tuple[int, ...], ...]

    @property
    def stops(self) -> range:
        return range(1, self.num_stops + 1)

    def has_link(self, i: int, j: int) -> bool:
        return i != j and np.isfinite(self.travel_time[i - 1, j - 1])

    def link_time(self, i: int, j: int) -> float:
        t = self.travel_time[i - 1, j - 1]
        if not np.isfinite(t) or i == j:
            raise ValueError(f"No physical link between stop {i} and stop {j}")
        return float(t)

    def num_links(self) -> int:
        """Number of directed links."""
        return int(np.isfinite(self.travel_time).sum() - self.num_stops)


def build_network(
    links: Iterable[Tuple[int, int, float]],
    num_stops: int,
    bidirectional: bool,
) -> TransitNetwork:
    """Build a validated ``TransitNetwork`` from (from_stop, to_stop, minutes).

    Stop ids must already be 1-based.
    """
    tt = np.full((num_stops, num_stops), np.inf, dtype=float)
    np.fill_diagonal(tt, 0.0)
    explicitly_given = np.zeros((num_stops, num_stops), dtype=bool)

    count = 0
    for raw_i, raw_j, raw_t in links:
        i, j, t = int(raw_i), int(raw_j), float(raw_t)
        count += 1
        if not (1 <= i <= num_stops and 1 <= j <= num_stops):
            raise ValueError(f"Link ({i},{j}) uses a stop outside 1..{num_stops}")
        if i == j:
            raise ValueError(f"Self-loop link at stop {i} is not allowed")
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"Link ({i},{j}) has invalid travel time {t}")
        if explicitly_given[i - 1, j - 1] and tt[i - 1, j - 1] != t:
            raise ValueError(f"Link ({i},{j}) listed twice with different times")
        tt[i - 1, j - 1] = t
        explicitly_given[i - 1, j - 1] = True

    if count == 0:
        raise PlaceholderDataError(
            "PLACEHOLDER: the Mandl link file contains no links. Insert the real "
            "Mandl network (from_stop,to_stop,travel_time) before running."
        )

    if bidirectional:
        # Mirror one-directional entries; keep explicit opposite-direction values.
        for i in range(num_stops):
            for j in range(num_stops):
                if explicitly_given[i, j] and not explicitly_given[j, i]:
                    tt[j, i] = tt[i, j]

    neighbours = tuple(
        tuple(j + 1 for j in range(num_stops) if j != i and np.isfinite(tt[i, j]))
        for i in range(num_stops)
    )
    network = TransitNetwork(num_stops=num_stops, travel_time=tt, neighbours=neighbours)
    validate_network(network)
    return network


def validate_network(network: TransitNetwork) -> None:
    """Check dimensions, isolated stops and (strong) connectivity."""
    n = network.num_stops
    if network.travel_time.shape != (n, n):
        raise ValueError(f"Travel-time matrix must be {n}x{n}, got {network.travel_time.shape}")
    isolated = [s for s in network.stops if not network.neighbours[s - 1]]
    if isolated:
        raise ValueError(f"Stops without any link: {isolated}")

    # Every stop must be reachable from every other stop (directed BFS from
    # stop 1 on the graph and on the reversed graph).
    def reachable(forward: bool) -> set:
        seen, queue = {1}, deque([1])
        while queue:
            u = queue.popleft()
            for v in network.stops:
                linked = network.has_link(u, v) if forward else network.has_link(v, u)
                if linked and v not in seen:
                    seen.add(v)
                    queue.append(v)
        return seen

    for direction in (True, False):
        seen = reachable(direction)
        if len(seen) != n:
            missing = sorted(set(network.stops) - seen)
            raise ValueError(f"Network is not strongly connected; unreachable stops: {missing}")


def load_network_from_csv(
    path: Path,
    num_stops: int,
    bidirectional: bool,
    input_stop_id_base: int = 1,
) -> TransitNetwork:
    """Load Mandl links from CSV with header ``from_stop,to_stop,travel_time``.

    Lines starting with ``#`` are comments.  ``input_stop_id_base`` = 0 converts
    0..14 numbering to the internal 1..15 numbering.
    """
    path = Path(path)
    if not path.exists():
        raise PlaceholderDataError(
            f"PLACEHOLDER: Mandl network file not found at {path}. "
            "Create it with columns from_stop,to_stop,travel_time."
        )
    links: List[Tuple[int, int, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = (line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        required = {"from_stop", "to_stop", "travel_time"}
        if reader.fieldnames is None or not required.issubset({f.strip() for f in reader.fieldnames}):
            raise ValueError(f"{path} must have header from_stop,to_stop,travel_time")
        offset = 1 - input_stop_id_base
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            links.append((int(row["from_stop"]) + offset, int(row["to_stop"]) + offset,
                          float(row["travel_time"])))
    return build_network(links, num_stops, bidirectional)


def route_is_physical_path(network: TransitNetwork, route: Sequence[int]) -> bool:
    """True if consecutive stops of ``route`` are joined by physical links."""
    return all(network.has_link(a, b) for a, b in zip(route[:-1], route[1:]))
