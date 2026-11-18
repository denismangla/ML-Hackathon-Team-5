"""Development-only Folium map with reference route, never an evaluator UI."""
import argparse
from pathlib import Path
from .data import load_trip,load_reference

def folium_map(trip,route,output):
    import folium
    m=folium.Map(location=trip.track[:,:2].mean(axis=0)[::-1].tolist(),zoom_start=12)
    roads=folium.FeatureGroup(name='Road network',show=False)
    # GeoJSON uses longitude, latitude. Folium polylines use latitude, longitude.
    features=[{'type':'Feature','geometry':{'type':'LineString','coordinates':trip.nodes[edge].tolist()},'properties':{'arc_id':i}} for i,edge in enumerate(trip.arcs)]
    folium.GeoJson({'type':'FeatureCollection','features':features},style_function=lambda _:dict(color='#888888',weight=1)).add_to(roads);roads.add_to(m)
    gps=folium.FeatureGroup(name='Raw GPS (blue)')
    folium.PolyLine(trip.track[:,:2][:,::-1].tolist(),color='#225eab',weight=2,tooltip='Raw GPS').add_to(gps);gps.add_to(m)
    reference=folium.FeatureGroup(name='Reference route (green; development only)')
    for arc in route:folium.PolyLine(trip.nodes[trip.arcs[arc]][:,::-1].tolist(),color='#278447',weight=4).add_to(reference)
    reference.add_to(m);folium.LayerControl().add_to(m)
    m.fit_bounds([trip.track[:,:2].min(axis=0)[::-1].tolist(),trip.track[:,:2].max(axis=0)[::-1].tolist()]);m.save(str(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--trip',default='00000001');p.add_argument('--output',type=Path,default=Path('results/training_trip_folium.html'));a=p.parse_args()
    f=a.data/'train'/a.trip;t=load_trip(f);a.output.parent.mkdir(parents=True,exist_ok=True);folium_map(t,load_reference(f,len(t.arcs)),a.output)
