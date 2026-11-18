"""Start either side of the PathSnap demo or print an actual HTTP response summary."""
import argparse
import json
import os
import sys
from pathlib import Path
from subprocess import run
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    api = commands.add_parser('server')
    api.add_argument('--data', required=True, type=Path, help='Folder containing eight-digit trip directories')
    api.add_argument('--mode', choices=['development','evaluator'],default='development')
    api.add_argument('--port', type=int, default=8000)
    ui = commands.add_parser('ui')
    ui.add_argument('--port', type=int, default=8090)
    probe = commands.add_parser('probe')
    probe.add_argument('--api', required=True)
    probe.add_argument('--trip', required=True)
    args = parser.parse_args()
    if args.command=='server':
        data=args.data.expanduser().resolve()
        if not data.is_dir() or not any(p.is_dir() and len(p.name)==8 and p.name.isdigit() for p in data.iterdir()):
            parser.error('--data must contain trip directories such as 00000009')
        env=dict(os.environ,PATHSNAP_DATA=str(data),PATHSNAP_MODE=args.mode,PATHSNAP_SCORE='1')
        run([sys.executable,'-m','uvicorn','pathsnap.api.app:app','--host','0.0.0.0','--port',str(args.port)],cwd=ROOT,env=env,check=True)
    elif args.command=='ui':
        run([sys.executable,'-m','http.server',str(args.port),'--bind','127.0.0.1','--directory',str(ROOT/'pathsnap/ui')],check=True)
    else:
        request=Request(args.api.rstrip('/')+'/predict',data=json.dumps({'trip_id':args.trip}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=120) as response:
            data=json.load(response)
            print('HTTP',response.status,request.full_url)
            print(json.dumps(dict(trip=data['trip_id'],observations=len(data['predicted_arc_ids']),sections=len(data['route_arc_sections']),seconds=data['timing_seconds'],metrics=data['metrics']),indent=2))

if __name__=='__main__':
    main()
