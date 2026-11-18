from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

def normalize_trip_id(trip_id):
    trip_id = str(trip_id).strip()

    if trip_id.isdigit():
        return trip_id.zfill(8)

    return trip_id

@dataclass
class TripData:
    trip_id: str
    nodes_lonlat: np.ndarray   # shape (N, 2), columns longitude, latitude
    arcs: np.ndarray           # shape (M, 2), columns from_node_id, to_node_id
    track_lonlat_time: np.ndarray  # shape (T, 3), lon, lat, timestamp_seconds
    route_arc_ids: np.ndarray | None


def _load_matrix(path: Path, columns: int, dtype=float) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, delimiter=None)
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.empty((0, columns), dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != columns:
        raise ValueError(f"{path} expected {columns} columns, found {arr.shape[1]}")
    return arr


def load_trip(data_root, split, trip_id, include_route=True):

    trip_id = normalize_trip_id(trip_id)

    root = Path(data_root)
    base = root / split / trip_id / trip_id

    nodes_path = base.with_suffix('.nodes')
    arcs_path = base.with_suffix('.arcs')
    track_path = base.with_suffix('.track')
    route_path = base.with_suffix('.route')
    for p in (nodes_path, arcs_path, track_path):
        if not p.exists():
            raise FileNotFoundError(p)

    nodes = _load_matrix(nodes_path, 2, float)
    arcs = _load_matrix(arcs_path, 2, np.int64).astype(np.int64, copy=False)
    track = _load_matrix(track_path, 3, float)
    route = None
    if include_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64).reshape(-1)

    if not np.isfinite(nodes).all() or not np.isfinite(track).all():
        raise ValueError(f"Non-finite coordinate/timestamp value in trip {trip_id}")
    if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"Arc endpoint outside node range in trip {trip_id}")
    if len(track) > 1 and np.any(np.diff(track[:, 2]) < 0):
        raise ValueError(f"Timestamps decrease in trip {trip_id}")
    if route is not None and len(route) and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"Route arc ID outside arc range in trip {trip_id}")

    return TripData(str(trip_id), nodes, arcs, track, route)


def list_trip_ids(data_root: str | Path, split: str = 'train') -> list[str]:
    d = Path(data_root) / split
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.nodes").exists())
