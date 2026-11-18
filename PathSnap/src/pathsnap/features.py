from __future__ import annotations
import numpy as np
import pandas as pd
from .alignment import align_track_to_route
from .candidates import CandidateGenerator
from .geo import LocalProjection
from .graph import RoadGraph
from .io import TripData

FEATURE_COLUMNS = [
    'distance_to_segment',
    'projection_fraction',
    'heading_difference',
    'heading_valid',
    'gps_speed',
    'time_delta',
    'gps_gap_flag',
    'segment_length',
    'endpoint_distance',
    'from_in_degree',
    'from_out_degree',
    'to_in_degree',
    'to_out_degree',
]


def build_trip_features(trip: TripData, radius_m: float = 100.0, top_k: int = 8,
                        label_search_radius_m: float = 150.0):
    if trip.route_arc_ids is None:
        raise ValueError(f'Trip {trip.trip_id} has no route labels; training features require .route')
    projection = LocalProjection(np.vstack([trip.nodes_lonlat, trip.track_lonlat_time[:, :2]]))
    nodes_xy = projection.project(trip.nodes_lonlat)
    track_xy = projection.project(trip.track_lonlat_time[:, :2])
    graph = RoadGraph.build(nodes_xy, trip.arcs)
    generator = CandidateGenerator(graph, radius_m=radius_m, top_k=top_k)
    rows, by_observation = generator.generate(track_xy, trip.track_lonlat_time[:, 2])
    aligned_arc, aligned_dist, aligned_progress, label_valid = align_track_to_route(
        graph, track_xy, trip.track_lonlat_time[:, 2], trip.route_arc_ids,
        search_radius_m=label_search_radius_m
    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f'No candidates generated for trip {trip.trip_id}')
    df['trip_id'] = trip.trip_id
    df['aligned_arc_id'] = df['observation_index'].map(lambda x: int(aligned_arc[int(x)]))
    df['alignment_distance'] = df['observation_index'].map(lambda x: float(aligned_dist[int(x)]))
    df['label_valid'] = df['observation_index'].map(lambda x: bool(label_valid[int(x)]))
    df['label'] = ((df['candidate_arc_id'] == df['aligned_arc_id']) & df['label_valid']).astype(int)
    df['candidate_hit'] = df.groupby('observation_index')['label'].transform('max').astype(int)
    # Keep the raw candidate set and label-valid flags. Candidate misses are reported,
    # not silently converted into positives.
    metadata = {
        'trip_id': trip.trip_id,
        'n_nodes': int(len(trip.nodes_lonlat)),
        'n_arcs': int(len(trip.arcs)),
        'n_observations': int(len(trip.track_lonlat_time)),
        'n_route_occurrences': int(len(trip.route_arc_ids)),
        'candidate_rows': int(len(df)),
        'label_valid_observations': int(label_valid.sum()),
        'candidate_hit_observations': int(df.groupby('observation_index')['candidate_hit'].max().sum()),
        'candidate_recall_at_k': float(
            df.groupby('observation_index')['candidate_hit'].max().reindex(range(len(track_xy)), fill_value=0)[label_valid].mean()
        ) if label_valid.any() else 0.0,
        'mean_alignment_distance_m': float(np.mean(aligned_dist[label_valid])) if label_valid.any() else None,
        'alignment_valid_fraction': float(label_valid.mean()),
    }
    return df, metadata, graph, projection, track_xy, by_observation
