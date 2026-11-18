from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import folium
from .io import load_trip
from .geo import LocalProjection
from .graph import RoadGraph


def make_map(data_root: str, trip_id: str, split: str = 'train', output: str | None = None,
             max_network_arcs: int = 3500):
    trip = load_trip(data_root, split, trip_id, include_route=(split == 'train'))
    proj = LocalProjection(np.vstack([trip.nodes_lonlat, trip.track_lonlat_time[:, :2]]))
    nodes_xy = proj.project(trip.nodes_lonlat)
    track_xy = proj.project(trip.track_lonlat_time[:, :2])
    graph = RoadGraph.build(nodes_xy, trip.arcs)
    center = np.median(trip.track_lonlat_time[:, :2], axis=0)
    fmap = folium.Map(location=[float(center[1]), float(center[0])], zoom_start=14,
                      tiles='OpenStreetMap', control_scale=True)

    # Draw a local network window around the GPS trace to keep the HTML light.
    lo = track_xy.min(axis=0) - 250.0
    hi = track_xy.max(axis=0) + 250.0
    mids = (graph.arc_p1 + graph.arc_p2) / 2.0
    keep = np.where((mids[:, 0] >= lo[0]) & (mids[:, 0] <= hi[0]) &
                    (mids[:, 1] >= lo[1]) & (mids[:, 1] <= hi[1]))[0]
    if len(keep) > max_network_arcs:
        # Deterministic sample, not used for any modelling.
        take = np.linspace(0, len(keep) - 1, max_network_arcs).astype(int)
        keep = keep[take]
    for arc_id in keep:
        u, v = trip.arcs[arc_id]
        p1 = trip.nodes_lonlat[u]
        p2 = trip.nodes_lonlat[v]
        folium.PolyLine([[float(p1[1]), float(p1[0])], [float(p2[1]), float(p2[0])]],
                        color='#9AA0A6', weight=1, opacity=0.35).add_to(fmap)

    gps_latlon = [[float(lat), float(lon)] for lon, lat in trip.track_lonlat_time[:, :2]]
    folium.PolyLine(gps_latlon, color='#1565C0', weight=4, opacity=0.9,
                    tooltip='Observed GPS trajectory').add_to(fmap)
    folium.CircleMarker(gps_latlon[0], radius=5, color='green', fill=True,
                        tooltip='GPS start').add_to(fmap)
    folium.CircleMarker(gps_latlon[-1], radius=5, color='red', fill=True,
                        tooltip='GPS end').add_to(fmap)

    if trip.route_arc_ids is not None:
        route_coords = []
        for arc_id in trip.route_arc_ids:
            u, v = trip.arcs[int(arc_id)]
            p1, p2 = trip.nodes_lonlat[u], trip.nodes_lonlat[v]
            if not route_coords:
                route_coords.append([float(p1[1]), float(p1[0])])
            route_coords.append([float(p2[1]), float(p2[0])])
        folium.PolyLine(route_coords, color='#E53935', weight=3, opacity=0.85,
                        tooltip='Reference route (training data only)').add_to(fmap)

    folium.LayerControl().add_to(fmap)
    out = Path(output or f'reports/trajectory_{trip_id}.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return out, len(keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--trip-id', default='00000026')
    p.add_argument('--split', default='train', choices=['train', 'student_test'])
    p.add_argument('--output', default='reports/trajectory_map.html')
    args = p.parse_args()
    out, n_arcs = make_map(args.data_root, args.trip_id, args.split, args.output)
    print(f'Saved {out}; rendered {n_arcs} nearby road arcs.')


if __name__ == '__main__':
    main()
