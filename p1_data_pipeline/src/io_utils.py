"""Loading raw PathSnap trips and projecting them to metres.

Each trip folder holds four whitespace-separated, header-less files:
  .track  lon lat t        (noisy GPS, ~1 Hz)
  .nodes  lon lat          (row index = node id)
  .arcs   from_node to_node (row index = arc id, directed)
  .route  arc_id           (reference route, ordered)

IMPORTANT: trips come from DIFFERENT cities (lon ~2.9, ~8.6, ~35.5 ...), so a
single global projection would be wrong. We pick the UTM zone per trip from the
centroid of its GPS track and keep every spatial calculation in metres.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

DATA_ROOT = Path("data/Student folder")
SPLITS = ("train", "student_test")


def list_trips(split: str, root: Path = DATA_ROOT) -> list[str]:
    return sorted(p.name for p in (Path(root) / split).iterdir() if p.is_dir())


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Trip:
    trip_id: str
    split: str
    track: pd.DataFrame        # columns lon, lat, t
    nodes_ll: np.ndarray       # (n_nodes, 2) lon, lat
    arcs: np.ndarray           # (n_arcs, 2) int  from_node, to_node
    route: np.ndarray | None   # (n_route,) arc ids, or None if not supplied
    epsg: int
    pts: np.ndarray            # (N, 2) projected GPS, metres
    t: np.ndarray              # (N,) seconds
    nodes: np.ndarray          # (n_nodes, 2) projected nodes, metres

    @property
    def A(self) -> np.ndarray:  # start point of every arc (metres)
        return self.nodes[self.arcs[:, 0]]

    @property
    def B(self) -> np.ndarray:  # end point of every arc (metres)
        return self.nodes[self.arcs[:, 1]]


def _read(path: Path, names: list[str], dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(split: str, trip_id: str, root: Path = DATA_ROOT) -> Trip:
    base = Path(root) / split / trip_id / trip_id
    track = _read(base.with_suffix(".track"), ["lon", "lat", "t"])
    nodes = _read(base.with_suffix(".nodes"), ["lon", "lat"])
    arcs = _read(base.with_suffix(".arcs"), ["u", "v"], dtype=np.int64)
    route_file = base.with_suffix(".route")
    route = None
    if route_file.exists():
        route = pd.read_csv(route_file, header=None)[0].to_numpy(dtype=np.int64)

    # Drop rows with missing / non-finite GPS values (reported in the EDA).
    ok = np.isfinite(track[["lon", "lat", "t"]].to_numpy()).all(axis=1)
    track = track.loc[ok].reset_index(drop=True)

    epsg = utm_epsg(track["lon"].mean(), track["lat"].mean())
    tf = Transformer.from_crs(4326, epsg, always_xy=True)  # (lon, lat) -> (x, y)
    px, py = tf.transform(track["lon"].to_numpy(), track["lat"].to_numpy())
    nx, ny = tf.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    return Trip(
        trip_id=trip_id, split=split, track=track,
        nodes_ll=nodes.to_numpy(), arcs=arcs.to_numpy(), route=route, epsg=epsg,
        pts=np.column_stack([px, py]), t=track["t"].to_numpy(float),
        nodes=np.column_stack([nx, ny]),
    )
