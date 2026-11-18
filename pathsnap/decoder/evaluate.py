from .config import selected_settings
import argparse,json,time
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from ..models.common import Dataset,predict,metrics,aggregate
from ..data import load_trip
from .viterbi import decode

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();out=Path('results/decoder');out.mkdir(parents=True,exist_ok=True);d=Dataset('data_stage1',a.raw);model=joblib.load('artifacts/selected_model.joblib')['model'];before=[];after=[]
 ids=d.splits['student_test_trips'][:a.limit] if a.limit else d.splits['student_test_trips']
 with threadpool_limits(limits=2):
  for trip in ids:
   f=d.get('student_test',trip);t=time.perf_counter();pred,prob=predict(model,f);b=metrics(f,pred,prob);b.update(trip_id=trip,inference_seconds=time.perf_counter()-t);before.append(b)
   raw=load_trip(a.raw/'student_test'/trip);t=time.perf_counter();decoded=decode(raw,f,prob,**selected_settings());z=metrics(f,decoded['predicted_arc_id'],prob);z.update(trip_id=trip,inference_seconds=time.perf_counter()-t,restarts=len(decoded['restart_observations'])-1);after.append(z)
   np.savez_compressed(out/f'{trip}.npz',**{k:v for k,v in decoded.items() if k in ('predicted_arc_id','restart_observations')});print(trip,'before',b['top1_accuracy'],'after',z['top1_accuracy'],'seconds',z['inference_seconds'],flush=True)
   summary=dict(classifier=aggregate(before),decoder=aggregate(after),per_trip_before=before,per_trip_after=after,settings=decoded['settings'],log_loss_note='Decoder table log loss remains the classifier candidate-probability log loss, not an HMM sequence likelihood.',continuity_note='Direct continuity counts identical or adjacent segments; longer valid network connections can be counted as discontinuous by this strict diagnostic.',evaluation_scope='student_test only; decoder settings selected on training fold 0')
   (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
