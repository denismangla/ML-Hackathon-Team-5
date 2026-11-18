import json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
class Dataset:
 def __init__(self,root,raw=None):
  self.root=Path(root);self.raw=Path(raw) if raw else None;self.splits=json.loads((self.root/'splits.json').read_text());self.cache={}
 def get(self,split,trip):
  key=(split,trip)
  if key not in self.cache:
   f=dict(np.load(self.root/'features'/split/f'{trip}.npz'));l=dict(np.load(self.root/'labels'/split/f'{trip}.npz'));f['labels']=l;f['starts']=np.r_[0,np.flatnonzero(np.diff(f['observation_index']))+1]
   if self.raw is not None:f['arcs']=np.loadtxt(self.raw/split/trip/f'{trip}.arcs',dtype=int,ndmin=2)
   self.cache[key]=f
  return self.cache[key]
 def training(self,ids,cap=20000):
  xs=[];ys=[];counts={}
  for trip in sorted(ids):
   f=self.get('train',trip);l=f['labels'];oi=f['observation_index'];valid=l['alignment_usable']&~l['alignment_ambiguous']&l['candidate_present'];selected=np.flatnonzero(valid);rng=np.random.default_rng(42+int(trip))
   if len(selected)>80:selected=rng.choice(selected,80,replace=False)
   mask=np.isin(oi,selected);xs.append(f['X'][mask]);ys.append(l['candidate_match'][mask]);counts[trip]=int(mask.sum())
  X=np.vstack(xs);y=np.concatenate(ys)
  if len(y)>cap:
   ix=np.random.default_rng(42).choice(len(y),cap,replace=False);X,y=X[ix],y[ix]
  if len(np.unique(y))<2:raise ValueError('Training sample lacks both labels')
  return X,y,dict(rows=len(y),positives=int(y.sum()),trips=len(ids),rows_before_global_cap=sum(counts.values()),per_trip_rows_before_global_cap=counts)
def predict(model,f):
 scores=np.asarray(model.predict_proba(f['X'])[:,1],dtype=float);scores=np.clip(np.nan_to_num(scores,nan=1e-12,posinf=1.,neginf=1e-12),1e-12,1.)
 starts=f['starts'];ends=np.r_[starts[1:],len(scores)];picked=np.array([s+np.argmax(scores[s:e]) for s,e in zip(starts,ends)]);probs=scores/np.repeat(np.add.reduceat(scores,starts),ends-starts)
 return f['candidate_arc_id'][picked],probs
def metrics(f,pred,probs):
 l=f['labels'];truth=l['reference_arc_id'];u=l['alignment_usable'];c=u&~l['alignment_ambiguous'];correct=pred==truth
 r=dict(observations=len(truth),correct=int(correct.sum()),top1_accuracy=float(correct.mean()),macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),macro_precision=float(precision_score(truth,pred,average='macro',zero_division=0)),macro_recall=float(recall_score(truth,pred,average='macro',zero_division=0)),candidate_miss_rate=float((~l['candidate_present']).mean()),usable_observations=int(u.sum()),confident_observations=int(c.sum()),usable_correct=int((correct&u).sum()),confident_correct=int((correct&c).sum()))
 epsilon=1e-12;truep=np.zeros(len(truth));np.add.at(truep,f['observation_index'],probs*l['candidate_match']);truep=np.where(l['candidate_present'],truep*(1-epsilon),epsilon);r['log_loss']=float(-np.log(np.clip(truep,epsilon,1)).mean())
 if 'arcs' in f:
  arcs=f['arcs'];valid=np.diff(f['timestamp'][f['starts']])<=120;connected=(pred[:-1]==pred[1:])|(arcs[pred[:-1],1]==arcs[pred[1:],0]);r['direct_continuity_numerator']=int((connected&valid).sum());r['direct_continuity_denominator']=int(valid.sum())
 return r
def aggregate(rows):
 n=sum(r['observations'] for r in rows);u=sum(r['usable_observations'] for r in rows);c=sum(r['confident_observations'] for r in rows);den=sum(r.get('direct_continuity_denominator',0) for r in rows)
 return dict(trips=len(rows),observations=n,top1_accuracy=sum(r['correct'] for r in rows)/n,macro_f1=float(np.mean([r['macro_f1'] for r in rows])),macro_precision=float(np.mean([r['macro_precision'] for r in rows])),macro_recall=float(np.mean([r['macro_recall'] for r in rows])),log_loss=sum(r['log_loss']*r['observations'] for r in rows)/n,candidate_miss_rate=sum(r['candidate_miss_rate']*r['observations'] for r in rows)/n,usable_top1_accuracy=sum(r['usable_correct'] for r in rows)/u if u else None,usable_observations=u,confident_top1_accuracy=sum(r['confident_correct'] for r in rows)/c if c else None,confident_observations=c,direct_continuity=sum(r.get('direct_continuity_numerator',0) for r in rows)/den if den else None,inference_seconds=sum(r['inference_seconds'] for r in rows))
def evaluate(model,dataset,split,ids,prediction_dir=None):
 rows=[]
 for trip in ids:
  f=dataset.get(split,trip);t=time.perf_counter();pred,probs=predict(model,f);elapsed=time.perf_counter()-t;r=metrics(f,pred,probs);r.update(trip_id=trip,inference_seconds=elapsed);rows.append(r)
  if prediction_dir:
   p=Path(prediction_dir);p.mkdir(parents=True,exist_ok=True);np.savez_compressed(p/f'{trip}.npz',predicted_arc_id=pred,candidate_probability=probs)
 return aggregate(rows),rows
