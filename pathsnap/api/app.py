import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from ..alignment import align_reference
from ..data import load_reference, load_trip
from ..decoder.config import selected_settings
from ..decoder.viterbi import decode
from ..features import build_features
from ..models.common import metrics, predict

app = FastAPI(title='PathSnap Team 5', version='1.1')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET','POST'],
                   allow_headers=['Content-Type'])
logger = logging.getLogger('uvicorn.error')


def data_root():
    directory = os.getenv('PATHSNAP_DATA')
    if not directory or not Path(directory).is_dir():
        raise HTTPException(503, 'Trip directory is unavailable. Set PATHSNAP_DATA.')
    return Path(directory)


@lru_cache(maxsize=1)
def model():
    return joblib.load(os.getenv('PATHSNAP_MODEL', 'artifacts/selected_model.joblib'))


class Request(BaseModel):
    trip_id: str = Field(pattern=r'^[0-9]{8}$')


@app.get('/health')
def health():
    data_root()
    return dict(status='ready', model=model()['model_name'], trip_directory_exists=True,
                mode=os.getenv('PATHSNAP_MODE','development'))


@app.get('/trips')
def trips():
    return dict(trips=sorted(p.name for p in data_root().iterdir() if p.is_dir()
                             and len(p.name)==8 and p.name.isascii() and p.name.isdigit()))


def display_roads(raw):
    edges = raw.nodes[raw.arcs]
    gps = raw.track[:, :2]
    lo, hi = gps.min(axis=0) - .0015, gps.max(axis=0) + .0015
    nearby = edges[((edges.max(axis=1) >= lo) & (edges.min(axis=1) <= hi)).all(axis=1)]
    if len(nearby) <= 2600:
        return nearby.tolist()
    sample = gps[np.linspace(0, len(gps)-1, min(320, len(gps)), dtype=int)]
    factor = np.array([max(.01, np.cos(np.deg2rad(gps[:,1].mean()))), 1.])
    _, nearest = cKDTree(nearby.mean(axis=1)*factor).query(sample*factor,k=8)
    return nearby[np.unique(nearest)].tolist()


def compare_predictions(folder, raw, features, pointwise, decoded, probabilities):
    labels = align_reference(raw, load_reference(folder,len(raw.arcs)))
    observations = features['observation_index']
    matches = features['candidate_arc_id'] == labels['reference_arc_id'][observations]
    present = np.zeros(len(raw.track),dtype=bool)
    np.logical_or.at(present,observations,matches)
    labels.update(candidate_match=matches,candidate_present=present)
    data = {**features,'labels':labels,'arcs':raw.arcs}
    before = metrics(data,pointwise,probabilities)
    after = metrics(data,decoded,probabilities)
    summary = {key: after[key] for key in ('top1_accuracy','macro_f1','candidate_miss_rate',
                                           'usable_observations','confident_observations')}
    for prefix, values in [('pointwise_',before),('',after)]:
        denominator = values['direct_continuity_denominator']
        summary[prefix+'direct_continuity'] = (
            values['direct_continuity_numerator']/denominator if denominator else None
        )
    summary.update(pointwise_top1_accuracy=before['top1_accuracy'],pointwise_macro_f1=before['macro_f1'])
    return summary


@app.post('/predict')
def inference(request: Request):
    folder = data_root()/request.trip_id
    if not folder.is_dir():
        raise HTTPException(404, 'Unknown trip')
    started = time.perf_counter()
    try:
        raw = load_trip(folder)
        features = build_features(raw)
        features['starts'] = np.r_[0,np.flatnonzero(np.diff(features['observation_index']))+1]
        bundle = model()
        with threadpool_limits(limits=2):
            pointwise, probabilities = predict(bundle['model'],features)
            decoded = decode(raw,features,probabilities,**selected_settings())
        prediction = decoded['predicted_arc_id']
        sections = decoded['route_sections']
        lines = [raw.nodes[np.r_[raw.arcs[section[0],0],raw.arcs[section,1]]].tolist() for section in sections]
        elapsed = time.perf_counter()-started
        response = dict(trip_id=request.trip_id,predicted_arc_ids=prediction.tolist(),
                        observation_indices=list(range(len(prediction))),gps_lonlat=raw.track[:,:2].tolist(),
                        predicted_segments_lonlat=raw.nodes[raw.arcs[np.unique(np.concatenate(sections))]].tolist(),
                        road_network_lonlat=display_roads(raw),
                        pointwise_segments_lonlat=raw.nodes[raw.arcs[np.unique(pointwise)]].tolist(),
                        route_lines_lonlat=lines,route_arc_sections=sections,bridge_count=decoded['bridge_count'],
                        restart_observations=decoded['restart_observations'].tolist(),timing_seconds=elapsed,
                        model=bundle['model_name'],metrics=None)
        if os.getenv('PATHSNAP_SCORE','0')=='1' and (folder/f'{request.trip_id}.route').is_file():
            response['metrics'] = compare_predictions(folder,raw,features,pointwise,prediction,probabilities)
            response['metrics_note'] = 'Scores use derived reference alignment, including ambiguous labels.'
    except (ValueError,FileNotFoundError) as error:
        raise HTTPException(422,str(error)) from error
    logger.info('POST /predict trip=%s observations=%s sections=%s prediction_seconds=%.3f score=%s',
                request.trip_id,len(prediction),len(sections),elapsed,response['metrics'])
    return response
