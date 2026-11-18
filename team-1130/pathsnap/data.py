"""Input validation and metric geometry. Prediction loading never reads route labels."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

R = 6371008.8
@dataclass
class Trip:
    trip_id: str
    nodes: np.ndarray
    arcs: np.ndarray
    track: np.ndarray

def read_matrix(path, cols, integer=False):
    x = np.loadtxt(path, ndmin=2)
    if x.shape[1] != cols or not len(x) or not np.isfinite(x).all():
        raise ValueError(f'{path}: empty, invalid columns or nonfinite numbers')
    if integer:
        if not np.equal(x, np.floor(x)).all(): raise ValueError(f'{path}: noninteger IDs')
        x = x.astype(np.int64)
    return x

def load_trip(folder):
    folder = Path(folder); name = folder.name
    nodes = read_matrix(folder/f'{name}.nodes', 2)
    arcs = read_matrix(folder/f'{name}.arcs', 2, True)
    track = read_matrix(folder/f'{name}.track', 3)
    for coords in (nodes, track[:, :2]):
        if (np.abs(coords[:,0])>180).any() or (np.abs(coords[:,1])>90).any():
            raise ValueError(f'{name}: coordinates outside lon/lat range')
    if (arcs<0).any() or (arcs>=len(nodes)).any(): raise ValueError(f'{name}: arc endpoint out of range')
    if (np.diff(track[:,2])<0).any(): raise ValueError(f'{name}: timestamps decrease')
    return Trip(name,nodes,arcs,track)

def load_reference(folder, arc_count):
    folder=Path(folder); route=read_matrix(folder/f'{folder.name}.route',1,True).ravel()
    if (route<0).any() or (route>=arc_count).any(): raise ValueError('Reference ID out of range')
    return route

def project(lonlat, origin):
    """Spherical azimuthal equidistant projection about each trip's centre (metres)."""
    lon,lat=np.radians(lonlat).T; lon0,lat0=np.radians(origin)
    dl=(lon-lon0+np.pi)%(2*np.pi)-np.pi
    cosc=np.clip(np.sin(lat0)*np.sin(lat)+np.cos(lat0)*np.cos(lat)*np.cos(dl),-1,1)
    c=np.arccos(cosc); sinc=np.sin(c)
    k=np.divide(c,sinc,out=np.ones_like(c),where=np.abs(sinc)>1e-12)
    return R*np.column_stack((k*np.cos(lat)*np.sin(dl),k*(np.cos(lat0)*np.sin(lat)-np.sin(lat0)*np.cos(lat)*np.cos(dl))))

def geometry(trip):
    origin=np.median(trip.track[:,:2],axis=0)
    nodes=project(trip.nodes,origin); points=project(trip.track[:,:2],origin)
    return nodes,points,nodes[trip.arcs[:,0]],nodes[trip.arcs[:,1]],origin

def distances(point,a,b):
    v=b-a; ll=np.einsum('ij,ij->i',v,v)
    fraction=np.clip(np.divide(np.einsum('ij,ij->i',point-a,v),ll,out=np.zeros_like(ll),where=ll>0),0,1)
    return np.linalg.norm(point-(a+fraction[:,None]*v),axis=1),fraction

def quality(trip,route):
    n,p,a,b,origin=geometry(trip); gaps=np.diff(trip.track[:,2])
    graph=coo_matrix((np.ones(len(trip.arcs)),(trip.arcs[:,0],trip.arcs[:,1])),shape=(len(n),len(n))).tocsr()
    degree=np.bincount(trip.arcs.ravel(),minlength=len(n))
    return dict(trip_id=trip.trip_id,nodes=len(n),arcs=len(a),observations=len(p),reference_arcs=len(route),
      weak_components=int(connected_components(graph,directed=True,connection='weak',return_labels=False)),
      isolated_nodes=int((degree==0).sum()),zero_length_arcs=int((np.linalg.norm(b-a,axis=1)==0).sum()),
      duplicate_timestamps=int((gaps==0).sum()),gaps_over_10s=int((gaps>10).sum()),max_gap_s=float(gaps.max(initial=0)),
      reference_breaks=int((trip.arcs[route[:-1],1]!=trip.arcs[route[1:],0]).sum()),
      lon_min=float(trip.track[:,0].min()),lon_max=float(trip.track[:,0].max()),lat_min=float(trip.track[:,1].min()),lat_max=float(trip.track[:,1].max()),
      max_projection_radius_m=float(np.linalg.norm(n,axis=1).max()),origin_lon=float(origin[0]),origin_lat=float(origin[1]))
