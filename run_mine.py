"""Train the selected team member's models using the same grouped evaluation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    profiles=json.loads((ROOT/'team.json').read_text())
    personal=ROOT/'assignment.json'
    default=json.loads(personal.read_text())['srn'] if personal.exists() else None
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True,type=Path)
    parser.add_argument('--member',choices=sorted(profiles),default=default,required=default is None)
    args=parser.parse_args()
    assignment=profiles[args.member]
    print('Assigned contributor:',assignment['name'],assignment['srn'],flush=True)
    for model in assignment['models']:
        subprocess.run([sys.executable,'-m','pathsnap.models.train','--data','data_stage1',
                        '--raw',str(args.raw.resolve()),'--model',model,'--output','results_local/'+model,
                        '--artifacts','artifacts_local'],cwd=ROOT,check=True)
    print('Finished. Review results_local before committing your work.')

if __name__=='__main__':
    main()
