"""Geometry-only inference features; no ground-truth argument or file access."""
import numpy as np
from scipy.spatial import cKDTree
from .data import geometry, distances

FEATURE_NAMES=['distance_m','projection_fraction','distance_start_m','distance_end_m','segment_length_m','heading_difference_rad','dt_s','displacement_m','speed_mps','gap_flag','heading_valid','candidate_fallback','from_degree','to_degree']

def build_features(trip,k=8,radius_m=100.,gap_s=120.):
    if k<1 or radius_m<=0 or gap_s<=0: raise ValueError('Invalid candidate configuration')
    nodes,points,a,b,origin=geometry(trip); k=min(k,len(a))
    lengths=np.linalg.norm(b-a,axis=1)
    # Subdivide only the SEARCH representation. IDs and final distances refer to full original arcs.
    pieces=np.maximum(1,np.ceil(lengths/100).astype(int))
    ids=np.repeat(np.arange(len(a)),pieces)
    starts=np.repeat(np.cumsum(pieces)-pieces,pieces)
    fractions=(np.arange(len(ids))-starts+0.5)/pieces[ids]
    samples=a[ids]+fractions[:,None]*(b-a)[ids]
    tree=cKDTree(samples)
    nearest=tree.query(points,k=min(max(k*4,32),len(samples)))[1]
    if nearest.ndim==1: nearest=nearest[:,None]
    rows=[]; obs=[]; arcids=[]
    degree=np.bincount(trip.arcs.ravel(),minlength=len(nodes))
    for i,point in enumerate(points):
        initial=np.unique(ids[np.atleast_1d(nearest[i])])
        count=min(max(k*4,32),len(samples))
        while len(initial)<k and count<len(samples):
            count=min(count*2,len(samples)); initial=np.unique(ids[np.atleast_1d(tree.query(point,k=count)[1])])
        d,_=distances(point,a[initial],b[initial]); upper=np.partition(d,k-1)[k-1]
        # Every segment piece has half-length <=50m, so this ball contains every segment
        # closer than the current kth candidate. This guarantees exact top-K in projected space.
        pool=np.unique(ids[tree.query_ball_point(point,float(upper)+50.000001)])
        d,f=distances(point,a[pool],b[pool]); order=np.lexsort((pool,d))[:k]
        chosen=pool[order]; d=d[order]; f=f[order]
        within=d<=radius_m
        fallback=not within.any()
        take=np.ones(len(d),dtype=bool) if fallback else within
        chosen,d,f=chosen[take],d[take],f[take]
        dt=0. if i==0 else float(trip.track[i,2]-trip.track[i-1,2])
        delta=np.zeros(2) if i==0 else points[i]-points[i-1]
        displacement=float(np.linalg.norm(delta)); gap=float(dt>gap_s)
        valid=float(i>0 and dt>0 and dt<=gap_s and displacement>=2.)
        bearing=np.arctan2(delta[1],delta[0]); v=b[chosen]-a[chosen]
        angles=np.abs((np.arctan2(v[:,1],v[:,0])-bearing+np.pi)%(2*np.pi)-np.pi)*valid
        m=len(chosen)
        rows.append(np.column_stack((d,f,np.linalg.norm(point-a[chosen],axis=1),np.linalg.norm(point-b[chosen],axis=1),lengths[chosen],angles,np.full(m,dt),np.full(m,displacement),np.full(m,displacement/dt if dt>0 and not gap else 0.),np.full(m,gap),np.full(m,valid),np.full(m,float(fallback)),degree[trip.arcs[chosen,0]],degree[trip.arcs[chosen,1]])))
        obs.extend([i]*m); arcids.extend(chosen)
    return dict(X=np.vstack(rows).astype(np.float32),observation_index=np.array(obs,dtype=np.int32),candidate_arc_id=np.array(arcids,dtype=np.int32),timestamp=trip.track[np.array(obs),2],feature_names=np.array(FEATURE_NAMES),projection_origin=origin)
