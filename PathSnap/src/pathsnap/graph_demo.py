from __future__ import annotations
import argparse
import numpy as np
from .io import load_trip
from .geo import LocalProjection
from .graph import RoadGraph


def main():
    p = argparse.ArgumentParser(description='Smoke-test directed graph construction and Dijkstra.')
    p.add_argument('--data-root', required=True)
    p.add_argument('--trip-id', default='00000026')
    args = p.parse_args()
    trip = load_trip(args.data_root, 'train', args.trip_id)
    proj = LocalProjection(np.vstack([trip.nodes_lonlat, trip.track_lonlat_time[:, :2]]))
    graph = RoadGraph.build(proj.project(trip.nodes_lonlat), trip.arcs)
    print(f'Trip: {trip.trip_id}')
    print(f'Nodes: {len(graph.nodes_xy):,}; arcs: {len(graph.arcs):,}')
    print('First 5 adjacency lists:')
    for node_id in range(min(5, len(graph.adjacency))):
        print(f'  {node_id}: {graph.adjacency[node_id][:5]}')
    arc_id = 0
    source, target = map(int, graph.arcs[arc_id])
    distance, path = graph.shortest_path(source, target, max_distance_m=5000.0)
    print(f'\nDijkstra test from node {source} to node {target}:')
    print(f'  distance_m={distance:.2f}; arc_ids={path[:20]}')
    assert np.isfinite(distance), 'Expected arc 0 endpoints to be connected.'
    print('GRAPH_SMOKE_TEST=PASS')


if __name__ == '__main__':
    main()
