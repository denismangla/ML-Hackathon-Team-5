import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import assert from 'node:assert/strict';
import App from './src/App.jsx';
import RouteMap from './src/RouteMap.jsx';
import Reports from './src/Reports.jsx';
import evidence from './src/evidence.json';

const initial=renderToStaticMarkup(<App/>);
for(const label of ['Connect','Reconstruct route','GPS observations','Model reports'])assert.ok(initial.includes(label));
const sample=evidence.sample;
const map=renderToStaticMarkup(<RouteMap result={sample}/>);
assert.equal((map.match(/<polyline /g)||[]).length,sample.route_lines_lonlat.length+sample.road_network_lonlat.length);
assert.equal((map.match(/<circle /g)||[]).length,sample.gps_lonlat.length);
assert.ok(!map.includes('NaN')&&map.includes(sample.trip_id));
assert.ok(renderToStaticMarkup(<App initialSample/>).includes('Saved student run'));
const reports=renderToStaticMarkup(<Reports evidence={evidence}/>);
for(const model of evidence.models){
  assert.ok(reports.includes(model.name));
  assert.ok(model.chart.startsWith('data:image/png;base64,'));
}
assert.ok(reports.includes('89.31%') && reports.includes('0.8642'));
console.log('Dashboard SSR, map road and GPS geometry, all five report entries, and refined scores verified.');
