from __future__ import annotations
import numpy as np
from pyproj import CRS, Transformer


class LocalProjection:
    """Trip-local azimuthal-equidistant projection; input/output order is x=lon, y=lat."""
    def __init__(self, lonlat: np.ndarray):
        lon0 = float(np.median(lonlat[:, 0]))
        lat0 = float(np.median(lonlat[:, 1]))
        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        )
        self.forward = Transformer.from_crs("EPSG:4326", local, always_xy=True)
        self.inverse = Transformer.from_crs(local, "EPSG:4326", always_xy=True)

    def project(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self.forward.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y]).astype(float)

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self.inverse.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat]).astype(float)


def segment_geometry(nodes_xy: np.ndarray, arcs: np.ndarray):
    p1 = nodes_xy[arcs[:, 0]]
    p2 = nodes_xy[arcs[:, 1]]
    vec = p2 - p1
    lengths = np.linalg.norm(vec, axis=1)
    # Bearing clockwise from north, in degrees.
    bearings = (np.degrees(np.arctan2(vec[:, 0], vec[:, 1])) + 360.0) % 360.0
    return p1, p2, lengths, bearings


def movement_features(track_xy: np.ndarray, timestamps: np.ndarray):
    n = len(track_xy)
    heading = np.zeros(n, dtype=float)
    speed = np.zeros(n, dtype=float)
    gap = np.zeros(n, dtype=float)
    heading_valid = np.zeros(n, dtype=float)
    gap_flag = np.zeros(n, dtype=float)
    for i in range(n):
        if n == 1:
            continue
        # Report the actual interval immediately preceding this point (or the next
        # interval for the first point), rather than a two-sided window.
        j = i - 1 if i > 0 else 0
        k = i if i > 0 else 1
        dt_interval = float(timestamps[k] - timestamps[j])
        gap[i] = max(dt_interval, 0.0)
        gap_flag[i] = float(dt_interval > 10.0)
        if dt_interval > 0 and dt_interval <= 60.0:
            speed[i] = float(np.linalg.norm(track_xy[k] - track_xy[j])) / dt_interval

        # Estimate heading from neighbouring points where the time span is usable.
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        dt = float(timestamps[b] - timestamps[a])
        displacement = track_xy[b] - track_xy[a]
        dist = float(np.linalg.norm(displacement))
        if dt > 0 and dist >= 3.0 and dt <= 60.0:
            heading[i] = (np.degrees(np.arctan2(displacement[0], displacement[1])) + 360.0) % 360.0
            heading_valid[i] = 1.0
    return heading, speed, gap, heading_valid, gap_flag


def wrapped_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)
