"""Create a geographic map with independently switchable route layers."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_trip, load_reference
from .features import build_features


def folium_map(trip, route, output, predicted=None, pointwise=None, point=300):
    import folium
    from branca.element import Element, MacroElement, Template

    gps = trip.track[:, :2]
    lo, hi = gps.min(axis=0), gps.max(axis=0)
    view = folium.Map(location=gps.mean(axis=0)[::-1].tolist(), prefer_canvas=True,
                      tiles=None, control_scale=True, zoom_control=True)
    folium.TileLayer('CartoDB Positron', name='Street map · CARTO').add_to(view)
    folium.TileLayer('OpenStreetMap', name='Street map · OpenStreetMap', show=False).add_to(view)
    folium.TileLayer(
        tiles='https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles © Esri, Vantor, Earthstar Geographics, and the GIS User Community',
        name='Satellite · Esri', show=False, max_zoom=19,
    ).add_to(view)
    edges = trip.nodes[trip.arcs]
    visible = ((edges.max(axis=1) >= lo - .0015) & (edges.min(axis=1) <= hi + .0015)).all(axis=1)

    def add_lines(label, coordinates, color, width, opacity=1, dash=None, show=True):
        group = folium.FeatureGroup(name=label, show=show)
        style = dict(color=color, weight=width, opacity=opacity)
        if dash:
            style['dashArray'] = dash
        folium.GeoJson({'type':'MultiLineString','coordinates':coordinates},
                       style_function=lambda _: style).add_to(group)
        group.add_to(view)

    add_lines('Road network', edges[visible].tolist(), '#7f8f9b', 1.4, .40)
    if pointwise:
        add_lines('Pointwise selection', pointwise, '#8259bd', 2, .80, '2 5', show=False)
    if predicted:
        add_lines('Predicted route', predicted, '#007d76', 6, .94)
    add_lines('Reference route', edges[route].tolist(), '#c85815', 3, .96, '6 6')
    points = folium.FeatureGroup(name='GPS observations')
    for index, (lon,lat) in enumerate(gps):
        folium.CircleMarker([lat,lon], radius=2.1, color='#2856a0', weight=0,
                            fill=True, fill_opacity=.74, tooltip=f'GPS #{index+1}').add_to(points)
    points.add_to(view)
    focus_index = min(max(point-1, 0), len(gps)-1)
    features = build_features(trip)
    rows = features['observation_index']==focus_index
    candidate_ids = features['candidate_arc_id'][rows]
    candidates = folium.FeatureGroup(name=f'Candidates · GPS #{focus_index+1}')
    for arc,distance in zip(candidate_ids,features['X'][rows,0]):
        folium.PolyLine(edges[arc][:,::-1].tolist(),color='#9146b8',weight=5,opacity=.9,dash_array='3 6',
                        tooltip=f'Arc {arc} · GPS distance {distance:.1f} m').add_to(candidates)
    latitude,longitude = map(float,gps[focus_index][::-1])
    folium.CircleMarker([latitude,longitude],radius=7,color='#643083',fill=True,fill_color='#fff',
                        tooltip=f'GPS #{focus_index+1} · {len(candidate_ids)} candidates').add_to(candidates)
    candidates.add_to(view)
    markers = folium.FeatureGroup(name='Start and finish')
    for (lon,lat),label,color in [(gps[0],'Start','green'),(gps[-1],'Finish','darkred')]:
        folium.Marker([lat,lon], tooltip=label, icon=folium.Icon(color=color)).add_to(markers)
    markers.add_to(view)
    folium.LayerControl(collapsed=False).add_to(view)
    view.fit_bounds([lo[::-1].tolist(),hi[::-1].tolist()],padding=(32,32),max_zoom=17)

    swatches = [('GPS observations','#2856a0'),('Reference route','#c85815')]
    if predicted:
        swatches.append(('Predicted route','#007d76'))
    swatches.extend([('Candidate roads','#9146b8'),('Road network','#7f8f9b')])
    legend = ''.join(f'<div><b style="color:{color}">●</b> {label}</div>' for label,color in swatches)
    focus_button = f'focus-{trip.trip_id}'
    fit_button = f'fit-{trip.trip_id}'
    view.get_root().html.add_child(Element(f'''
    <div style="position:fixed;z-index:1000;bottom:36px;left:14px;padding:12px 15px;
         background:white;color:#172e32;border:1px solid #b9cbc9;border-radius:8px;
         box-shadow:0 6px 18px #253b4425;font:13px/1.55 system-ui">
         <strong>PathSnap · {trip.trip_id}</strong><br>{len(gps):,} GPS observations{legend}
         <div style="display:flex;gap:6px;margin-top:8px">
           <button id="{focus_button}" type="button">GPS #{focus_index+1}</button>
           <button id="{fit_button}" type="button">Fit trip</button>
         </div>
    </div>'''))
    controls = MacroElement()
    controls._template = Template('{% macro script(this, kwargs) %}      document.getElementById("'+focus_button+'").onclick=function(){'+view.get_name()+'.setView(['+str(latitude)+','+str(longitude)+'],17);};      document.getElementById("'+fit_button+'").onclick=function(){'+view.get_name()+'.fitBounds('+json.dumps([lo[::-1].tolist(),hi[::-1].tolist()])+',{padding:[36,36]});};      {% endmacro %}')
    view.add_child(controls)
    view.save(str(output))
    return view


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--split', default='train', choices=['train','student_test'])
    parser.add_argument('--trip', default='00000001')
    parser.add_argument('--prediction', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--point', type=int, default=300, help='One-based GPS observation for candidate inspection')
    args = parser.parse_args()
    folder = args.data/args.split/args.trip
    trip = load_trip(folder)
    route = load_reference(folder,len(trip.arcs))
    prediction = json.loads(args.prediction.read_text()) if args.prediction else {}
    if prediction and prediction['trip_id'] != args.trip:
        raise ValueError('Prediction belongs to a different trip')
    args.output.parent.mkdir(exist_ok=True,parents=True)
    folium_map(trip,route,args.output,prediction.get('route_lines_lonlat'),prediction.get('pointwise_segments_lonlat'),point=args.point)
