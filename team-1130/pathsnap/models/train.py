import argparse,csv,json,time,platform
from pathlib import Path
import joblib,numpy as np,sklearn
from threadpoolctl import threadpool_limits
from .definitions import make_model,OWNERS
from .common import Dataset,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data_stage1');p.add_argument('--raw',required=True);p.add_argument('--output',default='results/models');p.add_argument('--artifacts',default='artifacts');p.add_argument('--model',choices=list(OWNERS));a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);art=Path(a.artifacts);art.mkdir(parents=True,exist_ok=True);d=Dataset(a.data,a.raw);train=d.splits['training_trips'];test=d.splits['student_test_trips'];folds=d.splits['cv_validation_fold_by_training_trip'];summary=[]
 with threadpool_limits(limits=2):
  for name in ([a.model] if a.model else OWNERS):
   print('START',name,flush=True);cv=[];config=make_model(name).get_params(deep=True);config={k:str(v) if not isinstance(v,(str,int,float,bool,type(None))) else v for k,v in config.items()}
   for fold in range(3):
    fit_ids=[x for x in train if folds[x]!=fold];val_ids=[x for x in train if folds[x]==fold];X,y,sampling=d.training(fit_ids);model=make_model(name);t=time.perf_counter();model.fit(X,y);fit_time=time.perf_counter()-t;scores,rows=evaluate(model,d,'train',val_ids);cv.append(dict(fold=fold,training_seconds=fit_time,sampling=sampling,metrics=scores,per_trip=rows));print(name,'FOLD',fold,'F1',scores['macro_f1'],'seconds',round(fit_time+scores['inference_seconds'],2),flush=True)
   X,y,sampling=d.training(train);model=make_model(name);t=time.perf_counter();model.fit(X,y);fit_time=time.perf_counter()-t;scores,rows=evaluate(model,d,'student_test',test,out/name/'student_predictions')
   bundle=dict(model=model,model_name=name,owner=OWNERS[name],feature_schema=json.loads((Path(a.data)/'feature_schema.json').read_text()),versions={'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},training_sampling=sampling);joblib.dump(bundle,art/f'{name}.joblib',compress=3)
   result=dict(model=name,owner=OWNERS[name],configuration=config,cv=cv,cv_macro_f1=float(np.mean([x['metrics']['macro_f1'] for x in cv])),student_test=scores,student_test_per_trip=rows,final_fit_seconds=fit_time,sampling=sampling);(out/f'{name}.json').write_text(json.dumps(result,indent=2));summary.append(dict(model=name,owner=OWNERS[name],cv_macro_f1=result['cv_macro_f1'],student_top1=scores['top1_accuracy'],student_macro_f1=scores['macro_f1'],student_log_loss=scores['log_loss'],student_direct_continuity=scores['direct_continuity'],student_inference_seconds=scores['inference_seconds'],fit_seconds=fit_time,training_rows=sampling['rows']))
   with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
   print('DONE',json.dumps(summary[-1]),flush=True)
 if len(summary)>1:
  selected=max(summary,key=lambda x:x['cv_macro_f1']);(out/'selection.json').write_text(json.dumps({'criterion':'Mean of three training-trip-group CV Macro F1 scores; per-trip F1 averaged within each fold','selected':selected,'model_count':len(summary),'warning':'All metrics use derived reference alignment; training capped at 20000 candidate rows consistently across models. Student testing does not choose the model.'},indent=2))
 make_plots(out,summary)
def make_plots(out,rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=[r['model'].replace('_','\n') for r in rows];x=np.arange(len(rows));fig,ax=plt.subplots(1,2,figsize=(12,5));ax[0].bar(x-.2,[r['cv_macro_f1'] for r in rows],.4,label='Training group-CV Macro F1');ax[0].bar(x+.2,[r['student_macro_f1'] for r in rows],.4,label='Student-test Macro F1');ax[0].set_ylim(0,1);ax[0].legend(fontsize=8);ax[1].bar(x,[r['student_top1'] for r in rows],label='Student-test Top-1');ax[1].set_ylim(0,1);ax[1].legend()
 for a in ax:a.set_xticks(x,names,fontsize=8);a.set_ylabel('Score')
 fig.suptitle('Actual model runs - derived reference labels');fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].bar(x,[r['student_log_loss'] for r in rows]);ax[0].set_title('Student-test candidate log loss');ax[1].bar(x,[r['student_inference_seconds'] for r in rows]);ax[1].set_title('Student-test scoring time (seconds)')
 for a in ax:a.set_xticks(x,names,fontsize=8)
 fig.tight_layout();fig.savefig(out/'loss_runtime.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
