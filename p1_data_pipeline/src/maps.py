"""Interactive Folium map for one PathSnap trip.

Usage (from the project folder):
    python -m src.maps --split train --trip 00000005 --candidates-at 300
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass

import folium
import numpy as np
import pandas as pd
from pyproj import Transformer

DEFAULT_DATA_DIR = os.path.join("data", "Student folder")
DEFAULT_OUT_DIR = os.path.join("results", "maps")


@dataclass
class Trip:
    split: str
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    node_lon: np.ndarray
    node_lat: np.ndarray
    arc_from: np.ndarray
    arc_to: np.ndarray
    route: np.ndarray
    x: np.ndarray
    y: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray


def _find_file(data_dir, split, trip_id, ext):
    patterns = [
        os.path.join(data_dir, split, trip_id, f"*.{ext}"),
        os.path.join(data_dir, split, f"{trip_id}*.{ext}"),
        os.path.join(data_dir, "**", split, trip_id, f"*.{ext}"),
        os.path.join(data_dir, "**", split, f"{trip_id}*.{ext}"),
    ]
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No .{ext} file for trip {trip_id} in split '{split}' under '{data_dir}'.")


def _read_table(path):
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


def load_trip(split, trip_id, data_dir=DEFAULT_DATA_DIR):
    track = _read_table(_find_file(data_dir, split, trip_id, "track"))
    nodes = _read_table(_find_file(data_dir, split, trip_id, "nodes"))
    arcs = _read_table(_find_file(data_dir, split, trip_id, "arcs"))
    route = _read_table(_find_file(data_dir, split, trip_id, "route"))

    lon, lat, t = track[:, 0].astype(float), track[:, 1].astype(float), track[:, 2].astype(float)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(t)
    lon, lat, t = lon[ok], lat[ok], t[ok]

    node_lon, node_lat = nodes[:, 0].astype(float), nodes[:, 1].astype(float)
    arc_from, arc_to = arcs[:, 0].astype(int), arcs[:, 1].astype(int)
    route = route[:, 0].astype(int)

    zone = int((float(np.mean(lon)) + 180.0) // 6) + 1
    epsg = (32600 if float(np.mean(lat)) >= 0 else 32700) + zone
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tf.transform(lon, lat)
    node_x, node_y = tf.transform(node_lon, node_lat)

    return Trip(split, trip_id, lon, lat, t, node_lon, node_lat, arc_from, arc_to,
                route, np.asarray(x), np.asarray(y), np.asarray(node_x), np.asarray(node_y))


def _arc_xy(trip):
    return (trip.node_x[trip.arc_from], trip.node_y[trip.arc_from],
            trip.node_x[trip.arc_to], trip.node_y[trip.arc_to])


def point_to_segments(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    u = np.clip(((px - ax) * dx + (py - ay) * dy) / safe, 0.0, 1.0)
    u = np.where(l2 > 0, u, 0.0)
    return np.hypot(px - (ax + u * dx), py - (ay + u * dy))


def candidates_for_point(trip, idx, radius=100.0, k=20):
    ax, ay, bx, by = _arc_xy(trip)
    d = point_to_segments(trip.x[idx], trip.y[idx], ax, ay, bx, by)
    near = np.where(d <= radius)[0]
    near = near[np.argsort(d[near])][:k]
    return near, d[near]


def _arc_line(trip, arc_id):
    a, b = trip.arc_from[arc_id], trip.arc_to[arc_id]
    return [[float(trip.node_lat[a]), float(trip.node_lon[a])],
            [float(trip.node_lat[b]), float(trip.node_lon[b])]]


def _network_arcs_near_track(trip, max_arcs=20000):
    ax, ay, bx, by = _arc_xy(trip)
    pair = np.sort(np.stack([trip.arc_from, trip.arc_to], axis=1), axis=1)
    _, first = np.unique(pair, axis=0, return_index=True)
    keep = np.zeros(len(trip.arc_from), dtype=bool)
    keep[first] = True

    xmin, xmax, ymin, ymax = trip.x.min(), trip.x.max(), trip.y.min(), trip.y.max()
    chosen = np.where(keep)[0]
    for margin in (300.0, 150.0, 75.0, 0.0):
        inside = ((np.maximum(ax, bx) >= xmin - margin) & (np.minimum(ax, bx) <= xmax + margin) &
                  (np.maximum(ay, by) >= ymin - margin) & (np.minimum(ay, by) <= ymax + margin))
        chosen = np.where(keep & inside)[0]
        if len(chosen) <= max_arcs:
            break
    if len(chosen) > max_arcs:
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_arcs).astype(int)]
    return chosen


LEGEND_HTML = """
<div style="position: fixed; bottom: 28px; left: 12px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #888; border-radius: 6px;
            font-size: 12px; line-height: 1.5; color: #222;">
  <b>PathSnap trip {trip}</b><br>
  <span style="color:#1f77b4">&#9679;</span> GPS points<br>
  <span style="color:#999">&#9473;</span> Road network<br>
  <span style="color:#2ca02c">&#9473;</span> Reference route (.route)<br>
  <span style="color:#ff7f0e">&#9473;</span> Candidate roads{cand_note}<br>
  <span style="color:#d62728">&#9473;</span> Predicted route
</div>
"""


def make_trip_map(trip, pred_arcs=None, candidates_at=None, radius=100.0, k=20,
                  tiles=True, max_points=2500):
    m = folium.Map(location=[float(trip.lat.mean()), float(trip.lon.mean())],
                   zoom_start=14, control_scale=True, tiles=None)
    m.get_root().header.add_child(
        folium.Element("<style>.leaflet-container{background:#eef1f4;}</style>"))

    if tiles:
        folium.TileLayer(
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            attr="&copy; OpenStreetMap contributors &copy; CARTO",
            name="Street map (CARTO)", max_zoom=19).add_to(m)
        esri = ("https://server.arcgisonline.com/ArcGIS/rest/services/{}"
                "/MapServer/tile/{{z}}/{{y}}/{{x}}")
        folium.TileLayer(esri.format("World_Street_Map"), attr="Tiles &copy; Esri",
                         name="Street map (Esri)", max_zoom=19, show=False).add_to(m)
        folium.TileLayer(esri.format("World_Imagery"), attr="Tiles &copy; Esri",
                         name="Satellite (Esri)", max_zoom=19, show=False).add_to(m)

    net = folium.FeatureGroup(name="Road network", show=True)
    arcs = _network_arcs_near_track(trip)
    if len(arcs):
        folium.PolyLine([_arc_line(trip, int(a)) for a in arcs],
                        color="#999999", weight=1.5, opacity=0.7).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name="Reference route", show=True)
    valid = trip.route[(trip.route >= 0) & (trip.route < len(trip.arc_from))]
    if len(valid):
        folium.PolyLine([_arc_line(trip, int(a)) for a in valid],
                        color="#2ca02c", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name="GPS points", show=True)
    folium.PolyLine(list(zip(trip.lat.tolist(), trip.lon.tolist())),
                    color="#1f77b4", weight=1, opacity=0.4).add_to(gps)
    n = len(trip.lat)
    step = max(1, math.ceil(n / max_points))
    for i in range(0, n, step):
        folium.CircleMarker([float(trip.lat[i]), float(trip.lon[i])], radius=2,
                            color="#1f77b4", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"point {i}, t={trip.t[i]:.0f}s").add_to(gps)
    folium.Marker([float(trip.lat[0]), float(trip.lon[0])], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(gps)
    folium.Marker([float(trip.lat[-1]), float(trip.lon[-1])], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(gps)
    gps.add_to(m)

    cand_note = ""
    if candidates_at is not None:
        if not 0 <= candidates_at < n:
            raise ValueError(f"--candidates-at must be between 0 and {n - 1}")
        ids, dists = candidates_for_point(trip, candidates_at, radius, k)
        cand = folium.FeatureGroup(name=f"Candidates (point {candidates_at})", show=True)
        for a, d in zip(ids, dists):
            folium.PolyLine(_arc_line(trip, int(a)), color="#ff7f0e", weight=5, opacity=0.9,
                            tooltip=f"arc {int(a)}, {d:.1f} m").add_to(cand)
        folium.CircleMarker([float(trip.lat[candidates_at]), float(trip.lon[candidates_at])],
                            radius=7, color="#6a0dad", fill=True, fill_opacity=1,
                            tooltip=f"selected point {candidates_at}").add_to(cand)
        folium.Circle([float(trip.lat[candidates_at]), float(trip.lon[candidates_at])],
                      radius=radius, color="#6a0dad", weight=1, fill=False,
                      dash_array="5").add_to(cand)
        cand.add_to(m)
        cand_note = f" (point {candidates_at}, {len(ids)} found)"

    pred = folium.FeatureGroup(name="Predicted route", show=True)
    if pred_arcs is not None and len(pred_arcs):
        p = np.asarray(pred_arcs, dtype=int)
        p = p[(p >= 0) & (p < len(trip.arc_from))]
        if len(p):
            folium.PolyLine([_arc_line(trip, int(a)) for a in p],
                            color="#d62728", weight=4, opacity=0.9,
                            dash_array="8").add_to(pred)
    pred.add_to(m)

    m.fit_bounds([[float(trip.lat.min()), float(trip.lon.min())],
                  [float(trip.lat.max()), float(trip.lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        LEGEND_HTML.format(trip=f"{trip.split}/{trip.trip_id}", cand_note=cand_note)))
    return m


def main():
    ap = argparse.ArgumentParser(description="Make an interactive Folium map for one trip.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--trip", required=True)
    ap.add_argument("--candidates-at", type=int, default=None)
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args()

    trip = load_trip(args.split, args.trip, args.data_dir)
    pred = np.loadtxt(args.pred, dtype=int, ndmin=1) if args.pred else None
    m = make_trip_map(trip, pred_arcs=pred, candidates_at=args.candidates_at,
                      radius=args.radius, k=args.k, tiles=not args.no_tiles)
                      
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.split}_{args.trip}.html")
    m.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()