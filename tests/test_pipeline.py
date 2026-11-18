import tempfile,unittest
from pathlib import Path
import numpy as np
from pathsnap.data import Trip,geometry,distances,load_trip,load_reference,project
from pathsnap.features import build_features
from pathsnap.alignment import align_reference

class PipelineTests(unittest.TestCase):
    def trip(self):
        nodes=np.array([[10,50],[10.001,50],[10.002,50],[10.001,50.001]])
        arcs=np.array([[0,1],[1,2],[1,0],[2,1],[1,3],[3,1]])
        track=np.array([[10.0001,50,0],[10.0008,50,1],[10.0014,50,2],[10.0018,50,3]])
        return Trip('toy',nodes,arcs,track)
    def test_projection_metres(self):
        p=project(np.array([[0,0],[0,.001]]),np.array([0,0]));self.assertAlmostEqual(p[1,1],111.195,places=2)
    def test_candidates_equal_bruteforce(self):
        t=self.trip();f=build_features(t,k=3,radius_m=1e6);_,pts,a,b,_=geometry(t)
        for i,p in enumerate(pts):
            d,_=distances(p,a,b);expected=np.lexsort((np.arange(len(a)),d))[:3]
            np.testing.assert_array_equal(f['candidate_arc_id'][f['observation_index']==i],expected)
    def test_long_segment_search(self):
        t=Trip('long',np.array([[0.,0.],[1.,0.],[.49,.001],[.51,.001]]),np.array([[0,1],[2,3]]),np.array([[.5,.00001,0.]]))
        self.assertEqual(build_features(t,k=1)['candidate_arc_id'][0],0)
    def test_repeated_occurrence_alignment(self):
        t=self.trip();t.track=np.array([[10.0001,50,0],[10.0009,50,1],[10.0019,50,2],[10.0011,50,3],[10.0001,50,4],[10.0009,50,5]])
        r=align_reference(t,np.array([0,1,3,2,0]));self.assertTrue((np.diff(r['reference_occurrence'])>=0).all());self.assertGreater(r['reference_occurrence'][-1],0)
    def test_gap_and_fallback(self):
        t=self.trip();t.track[1,2]=1000;t.track[2:,2]=[1001,1002];t.track[:,1]+=.1
        f=build_features(t,k=2,radius_m=1);self.assertTrue((f['X'][:,11]==1).all());self.assertTrue((f['X'][f['observation_index']==1,9]==1).all());self.assertTrue(np.isfinite(f['X']).all())
    def test_reference_file_cannot_affect_features(self):
        t=self.trip()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'toy';d.mkdir()
            for ext,x in [('nodes',t.nodes),('arcs',t.arcs),('track',t.track)]:np.savetxt(d/f'toy.{ext}',x)
            before=build_features(load_trip(d));(d/'toy.route').write_text('999999\n')
            after=build_features(load_trip(d));np.testing.assert_array_equal(before['X'],after['X']);np.testing.assert_array_equal(before['candidate_arc_id'],after['candidate_arc_id'])
            with self.assertRaises(ValueError):load_reference(d,len(t.arcs))
    def test_invalid_ids_rejected(self):
        t=self.trip()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'toy';d.mkdir()
            for ext,x in [('nodes',t.nodes),('arcs',np.array([[0,999]])),('track',t.track)]:np.savetxt(d/f'toy.{ext}',x)
            with self.assertRaises(ValueError):load_trip(d)
    def test_zero_length_segment(self):
        d,f=distances(np.array([1.,0.]),np.array([[0.,0.]]),np.array([[0.,0.]]));self.assertEqual(d[0],1);self.assertEqual(f[0],0)
if __name__=='__main__':unittest.main()
