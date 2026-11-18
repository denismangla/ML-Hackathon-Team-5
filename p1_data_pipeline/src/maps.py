"""Interactive Folium map of one trip.

Usage:
    python -m src.maps --split train --trip 00000005
    python -m src.maps --split student_test --trip 00000052 --candidates-at 120

Layers (toggle with the layer control, top right):
    GPS observations (blue)  | Road network (grey) | Reference route (green)
    Predicted route (red, only when `pred_arcs` is supplied by P4/the API)
    Candidates for one chosen GPS point (orange, optional)
Coordinates are drawn from the ORIGINAL lon/lat (no projection needed for display).
"""
import argparse
from pathlib import Path

import folium
import numpy as np

from .candidates import generate_candidates
from .io_utils import DATA_ROOT, Trip, load_trip

LEGEND = """
<div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
 padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font: 13px sans-serif;">
 <b>PathSnap trip {tid}</b><br>
 <span style="color:#1f77b4">&#9679;</span> GPS observations<br>
 <span style="color:#999">&#9473;</span> Road network<br>
 <span style="color:#2ca02c">&#9473;</span> Reference route<br>
 <span style="color:#d62728">&#9473;</span> Predicted route<br>
 <span style="color:#ff7f0e">&#9473;</span> Candidates (one GPS point)
</div>"""


def _arc_lines(trip: Trip, arc_ids):
    ll = trip.nodes_ll                       # (lon, lat) -> folium wants (lat, lon)
    arcs = trip.arcs[np.asarray(arc_ids, dtype=int)]
    return [[[ll[u, 1], ll[u, 0]], [ll[v, 1], ll[v, 0]]] for u, v in arcs]


def make_trip_map(trip: Trip, pred_arcs=None, out_path=None, candidates_at=None,
                  max_points=4000, margin_m=300.0) -> folium.Map:
    lat, lon = trip.track["lat"].to_numpy(), trip.track["lon"].to_numpy()
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=13, control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # Road network: only arcs near the track, undirected duplicates removed
    near = (trip.A[:, 0] > trip.pts[:, 0].min() - margin_m) & (trip.A[:, 0] < trip.pts[:, 0].max() + margin_m) & \
           (trip.A[:, 1] > trip.pts[:, 1].min() - margin_m) & (trip.A[:, 1] < trip.pts[:, 1].max() + margin_m)
    ids = np.where(near)[0]
    uv = np.sort(trip.arcs[ids], axis=1)
    _, keep = np.unique(uv, axis=0, return_index=True)
    net = folium.FeatureGroup(name=f"Road network ({len(keep)} segments)", show=True)
    folium.PolyLine(_arc_lines(trip, ids[keep]), color="#999999", weight=2, opacity=0.7).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lat) // max_points)
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"pt {i}  t={trip.t[i]:.0f}s").add_to(gps)
    gps.add_to(m)

    if trip.route is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, trip.route), color="#2ca02c", weight=5, opacity=0.8).add_to(ref)
        ref.add_to(m)
    if pred_arcs is not None and len(pred_arcs):
        pr = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_arc_lines(trip, pred_arcs), color="#d62728", weight=3, opacity=0.9, dash_array="6").add_to(pr)
        pr.add_to(m)
    if candidates_at is not None:
        c = generate_candidates(trip.pts[[candidates_at]], trip.A, trip.B)
        cg = folium.FeatureGroup(name=f"Candidates for point {candidates_at}", show=True)
        folium.PolyLine(_arc_lines(trip, c.arc_id.to_numpy()), color="#ff7f0e", weight=4).add_to(cg)
        folium.Marker([lat[candidates_at], lon[candidates_at]], tooltip=f"point {candidates_at}").add_to(cg)
        cg.add_to(m)

    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=trip.trip_id)))
    folium.LayerControl(collapsed=False).add_to(m)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--trip", default="00000005")
    ap.add_argument("--candidates-at", type=int, default=None)
    ap.add_argument("--root", default=str(DATA_ROOT))
    a = ap.parse_args()
    tr = load_trip(a.split, a.trip, a.root)
    out = f"results/maps/{a.split}_{a.trip}.html"
    make_trip_map(tr, out_path=out, candidates_at=a.candidates_at)
    print("saved", out)
