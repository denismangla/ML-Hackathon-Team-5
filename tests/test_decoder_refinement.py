import unittest

import numpy as np

from pathsnap.data import Trip
from pathsnap.decoder.viterbi import decode


class DecoderRefinementTests(unittest.TestCase):
    def test_small_backward_projection_does_not_require_a_u_turn(self):
        trip = Trip('synthetic', np.array([[10.,50.],[10.001,50.]]),
                    np.array([[0,1]]),np.array([[10.0005,50.,0.],[10.0004,50.,1.]]))
        features = {
            'observation_index': np.array([0,1]), 'candidate_arc_id': np.array([0,0]),
            'X': np.array([[0.,.5],[0.,.4]]),
        }
        original = decode(trip,features,np.ones(2),reverse_tolerance_m=0.)
        refined = decode(trip,features,np.ones(2),reverse_tolerance_m=25.)
        self.assertEqual(original['route_sections'], [[0],[0]])
        self.assertEqual(refined['route_sections'], [[0]])
        np.testing.assert_array_equal(refined['predicted_arc_id'], [0,0])
