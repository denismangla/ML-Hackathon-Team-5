"""Offline derived references only. Never imported by inference feature generation."""
import numpy as np
from .data import geometry,distances

def align_reference(trip,route,gap_s=120.,max_distance_m=100.):
    _,points,a,b,_=geometry(trip)
    a,b=a[route],b[route]; total=len(points); n=len(route)
    occurrence=np.empty(total,dtype=np.int32)
    boundaries=np.r_[0,np.flatnonzero(np.diff(trip.track[:,2])>gap_s)+1,total]
    # Nondecreasing occurrence DP with a small penalty per skipped occurrence.
    # Gaps start an independent alignment window. Repeated arcs keep separate occurrences.
    ranks=np.arange(n); jump_penalty=.03
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        back=np.empty((hi-lo,n),dtype=np.int32)
        cost=None
        for offset,i in enumerate(range(lo,hi)):
            d,_=distances(points[i],a,b); emission=np.minimum((d/20.)**2,2500.)
            if cost is None: cost=emission; back[offset]=ranks
            else:
                previous=cost-jump_penalty*ranks
                prefix=np.minimum.accumulate(previous)
                best=np.maximum.accumulate(np.where(previous==prefix,ranks,0))
                cost=emission+prefix+jump_penalty*ranks
                cost-=cost.min(); back[offset]=best
        state=int(np.argmin(cost))
        for offset in range(hi-lo-1,-1,-1):
            occurrence[lo+offset]=state; state=int(back[offset,state])
    reference_arc=route[occurrence]
    residual=np.zeros(total); ambiguous=np.zeros(total,dtype=bool)
    for i,p in enumerate(points):
        d,_=distances(p,a,b); residual[i]=d[occurrence[i]]
        # Alternative DISTINCT arc, not another occurrence of the selected arc.
        alternatives=d[route!=reference_arc[i]]
        ambiguous[i]=bool(len(alternatives) and alternatives.min()<=residual[i]+5.)
    return dict(reference_arc_id=reference_arc.astype(np.int32),reference_occurrence=occurrence,alignment_distance_m=residual.astype(np.float32),alignment_ambiguous=ambiguous,alignment_usable=residual<=max_distance_m)
