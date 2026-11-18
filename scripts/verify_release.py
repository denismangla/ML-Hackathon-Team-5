"""Exercise real HTTP handlers through ASGI for every development trip."""
import argparse, json, os
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from pathsnap.api.app import app
from pathsnap.data import load_trip

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=os.environ.get('PATHSNAP_DATA'),required='PATHSNAP_DATA' not in os.environ)
    os.environ['PATHSNAP_DATA']=str(parser.parse_args().data)
    os.environ['PATHSNAP_SCORE']='1'
    api = TestClient(app)
    root = Path(os.environ['PATHSNAP_DATA'])
    out = Path('results/release')
    out.mkdir(exist_ok=True, parents=True)
    records = []
    for trip in api.get('/trips').json()['trips']:
        response = api.post('/predict', json={'trip_id':trip})
        response.raise_for_status()
        payload = response.json()
        raw = load_trip(root/trip)
        expected = np.load(Path('results/decoder') / (trip+'.npz'))['predicted_arc_id']
        np.testing.assert_equal(payload['predicted_arc_ids'], expected)
        for section in payload['route_arc_sections']:
            edges = raw.arcs[section]
            assert np.array_equal(edges[:-1,1],edges[1:,0])
        assert len(payload['route_lines_lonlat']) == len(payload['restart_observations'])
        records.append(dict(trip=trip, status=response.status_code, observations=len(expected),
                            bridge_arcs=payload['bridge_count'], sections=len(payload['route_arc_sections']),
                            prediction_seconds=payload['timing_seconds']))
        print(records[-1],flush=True)
        if trip=='00000009':
            (out/'sample_response.json').write_text(json.dumps(payload))
            (Path('results')/'api_response_00000009.json').write_text(json.dumps(payload))
    (out/'api_sweep.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
