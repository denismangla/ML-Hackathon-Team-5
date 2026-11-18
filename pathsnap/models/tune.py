"""Gaussian-process Bayesian optimisation, expected improvement acquisition."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np,joblib
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel,ConstantKernel
from threadpoolctl import threadpool_limits
from .definitions import make_model
from .common import Dataset,evaluate

def parameters(name,x):
 if name=='random_forest':return dict(n_estimators=int(50+x[0]*100),max_depth=int(6+x[1]*20),min_samples_leaf=int(1+x[2]*9))
 if name=='hist_gradient_boosting':return dict(learning_rate=float(10**(-2+x[0]*1.4)),max_leaf_nodes=int(7+x[1]*25),l2_regularization=float(10**(-2+x[2]*3)))
 if name=='gaussian_nb':return dict(var_smoothing=float(10**(-12+x[0]*6)))
 if name=='logistic_regression':return dict(C=float(10**(-2+x[0]*4)))
 if name=='knn':return dict(n_neighbors=int(3+x[0]*18))

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--trials',type=int,default=10);a=p.parse_args();out=Path('results/tuning');out.mkdir(parents=True,exist_ok=True)
 selection=json.loads(Path('results/models/selection.json').read_text());name=selection['selected']['model'];d=Dataset('data_stage1',a.raw);folds=d.splits['cv_validation_fold_by_training_trip'];train=d.splits['training_trips'];cache=[]
 for fold in range(3):
  fit=[i for i in train if folds[i]!=fold];val=[i for i in train if folds[i]==fold];X,y,s=d.training(fit);cache.append((X,y,val))
 rng=np.random.default_rng(42);vectors=[];scores=[];trials=[]
 with threadpool_limits(limits=2):
  for i in range(a.trials):
   if i<3:x=rng.random(3);method='random_initialisation'
   else:
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1.)*Matern(length_scale=np.ones(3),nu=2.5)+WhiteKernel(1e-5),normalize_y=True,random_state=42,n_restarts_optimizer=0)
    with warnings.catch_warnings():warnings.simplefilter('ignore');gp.fit(np.array(vectors),np.array(scores))
    pool=rng.random((512,3));mu,sd=gp.predict(pool,return_std=True);improvement=mu-max(scores)-.001;z=improvement/np.maximum(sd,1e-12);ei=improvement*norm.cdf(z)+sd*norm.pdf(z);x=pool[int(np.argmax(ei))];method='GP_expected_improvement'
   params=parameters(name,x);t=time.perf_counter();fold_scores=[]
   for X,y,val in cache:
    m=make_model(name,params);m.fit(X,y);s,_=evaluate(m,d,'train',val);fold_scores.append(s['macro_f1'])
   score=float(np.mean(fold_scores));vectors.append(x.tolist());scores.append(score);trials.append(dict(trial=i,method=method,unit_coordinates=x.tolist(),parameters=params,fold_macro_f1=fold_scores,cv_macro_f1=score,seconds=time.perf_counter()-t));(out/'trials.json').write_text(json.dumps(trials,indent=2));print('TRIAL',i,score,params,flush=True)
  best=trials[int(np.argmax(scores))];X,y,sampling=d.training(train);model=make_model(name,best['parameters']);t=time.perf_counter();model.fit(X,y);fit_seconds=time.perf_counter()-t;student,per_trip=evaluate(model,d,'student_test',d.splits['student_test_trips'],out/'student_predictions')
  bundle=joblib.load(f'artifacts/{name}.joblib');bundle.update(model=model,tuned_parameters=best['parameters']);joblib.dump(bundle,'artifacts/tuned_model.joblib',compress=3)
  baseline=json.loads(Path(f'results/models/{name}.json').read_text());accept=best['cv_macro_f1']>baseline['cv_macro_f1'];chosen='tuned_model.joblib' if accept else f'{name}.joblib';joblib.dump(joblib.load('artifacts/'+chosen),'artifacts/selected_model.joblib',compress=3)
  result=dict(model=name,strategy='Gaussian process with Matern kernel and expected improvement; 3 seeded random initial trials then Bayesian proposals',seed=42,trial_count=len(trials),objective='3-fold group-by-trip CV Macro F1 on 80 training trips',search_space='See parameters() mapping from [0,1]^3; RF: trees 50..149, depth 6..25, leaf size 1..9',best_trial=best,untuned_cv_macro_f1=baseline['cv_macro_f1'],tuned_cv_macro_f1=best['cv_macro_f1'],untuned_student_test=baseline['student_test'],tuned_student_test=student,tuned_student_test_per_trip=per_trip,final_fit_seconds=fit_seconds,tuned_selected=accept,selected_artifact=chosen,training_sampling=sampling)
  (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['model','trial_count','untuned_cv_macro_f1','tuned_cv_macro_f1','tuned_selected']},indent=2))
if __name__=='__main__':main()
