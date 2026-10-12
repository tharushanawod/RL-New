# Input data — PLACEHOLDERS

No real network or OD values are included. The program refuses to run
(`INPUT DATA MISSING`) until both items below are supplied.

## 1. `mandl_links.csv` — Mandl's Swiss network

```
from_stop,to_stop,travel_time
<i>,<j>,<minutes>
...
```

* One row per physical link. With `LINKS_ARE_BIDIRECTIONAL = True` (config)
  a link listed once is usable in both directions with the same time; if both
  directions are listed each keeps its own time.
* Stops 1..15 by default. If your data numbers stops 0..14, set
  `INPUT_STOP_ID_BASE = 0` in `config.py`.
* Lines starting with `#` are ignored.
* Validation: 15 stops, positive finite times, no self-loops, no stop without
  links, strongly connected network.

## 2. `od/od_hour_00.csv` … `od/od_hour_23.csv` — hourly OD matrices

* 15 rows × 15 comma-separated values, no header. Row = origin stop,
  column = destination stop, value = passengers in that hour.
* The file number is the clock hour (`od_hour_08.csv` = 08:00–09:00). Each episode starts at
  `OD_START_FILE_HOUR` (default 8) and wraps around midnight: 08, 09, …, 23, 00, …, 07.
* Validation: exactly 15×15, finite, non-negative. The diagonal is ignored by
  the assignment (`IGNORE_OD_DIAGONAL`).
* The matrices are used exactly as supplied: no scaling, rounding,
  aggregation or discretisation.
