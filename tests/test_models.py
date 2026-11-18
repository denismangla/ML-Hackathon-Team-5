import unittest,json,os
from pathlib import Path
import numpy as np,joblib
from pathsnap.models.common import Dataset,predict
from pathsnap.models.definitions import OWNERS
from pathsnap.data import Trip
from pathsnap.features import build_features
from pathsnap.decoder.viterbi import decode
class ModelTests(unittest.TestCase):
 def test_all_artifacts_and_probability_groups(self):
  d=Dataset('data_stage1');f=d.get('student_test',d.splits['student_test_trips'][0])
  for name in OWNERS:
   b=joblib.load('artifacts/'+name+'.joblib');pred,p=predict(b['model'],f);self.assertEqual(len(pred),len(f['starts']));np.testing.assert_allclose(np.add.reduceat(p,f['starts']),1,atol=1e-8)
 def test_split_groups(self):
  s=json.loads(Path('data_stage1/splits.json').read_text());self.assertEqual(set(s['cv_validation_fold_by_training_trip']),set(s['training_trips']));self.assertFalse(set(s['training_trips'])&set(s['student_test_trips']))
 def test_decoder_gap_restart(self):
  t=Trip('toy',np.array([[0.,0.],[.001,0.],[.002,0.]]),np.array([[0,1],[1,2]]),np.array([[.0001,0,0],[.0015,0,1000]]));f=build_features(t);p=np.ones(len(f['X']));r=decode(t,f,p);self.assertEqual(r['restart_observations'].tolist(),[0,1]);self.assertEqual(len(r['predicted_arc_id']),2)
if __name__=='__main__':unittest.main()
