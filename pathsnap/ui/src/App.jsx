import React,{useState} from 'react';
import RouteMap from './RouteMap';
import Reports from './Reports';
import evidence from './evidence.json';
import './dashboard.css';

const pct = value=>value==null?'—':(100*value).toFixed(2)+'%';
const number = value=>value==null?'—':value.toFixed(4);

function downloadRoute(result){
  const features=result.route_lines_lonlat.map((coordinates,index)=>({
    type:'Feature',geometry:{type:'LineString',coordinates},properties:{section:index+1,trip_id:result.trip_id},
  }));
  const url=URL.createObjectURL(new Blob([JSON.stringify({type:'FeatureCollection',features})],{type:'application/geo+json'}));
  const link=document.createElement('a');
  link.href=url;
  link.download='PathSnap_'+result.trip_id+'.geojson';
  link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export default function App({initialTab='map', initialSample=typeof location!=='undefined'&&location.protocol==='file:'}){
  const [tab,setTab]=useState(initialTab);
  const [address,setAddress]=useState('http://127.0.0.1:8000'), [server,setServer]=useState('');
  const [trips,setTrips]=useState([]), [trip,setTrip]=useState(''), [mode,setMode]=useState('Disconnected');
  const [result,setResult]=useState(initialSample?evidence.sample:null), [sample,setSample]=useState(initialSample);
  const [busy,setBusy]=useState(''), [error,setError]=useState('');
  const score=result?.metrics;
  async function request(base,path,options={}){
    const response=await fetch(base+path,{...options,signal:AbortSignal.timeout(120000)});
    if(!response.ok){
      const body=await response.json().catch(()=>null);
      throw new Error(typeof body?.detail==='string'?body.detail:'HTTP '+response.status);
    }
    return response.json();
  }
  function clear(){
    setResult(null);setSample(false);setError('');
  }
  async function connect(event){
    event.preventDefault();
    clear();setBusy('Connecting');setTrips([]);setMode('Disconnected');setServer('');
    try{
      const url=new URL(address.trim());
      if(!['http:','https:'].includes(url.protocol))throw new Error('Enter an HTTP or HTTPS address.');
      const health=await request(url.origin,'/health');
      const list=await request(url.origin,'/trips');
      setServer(url.origin);setTrips(list.trips);setTrip(list.trips[0]||'');
      setMode(health.mode==='evaluator'?'Evaluator connected':'Student server connected');
      if(!list.trips.length)setError('No trips found. Check the dataset folder on the model laptop.');
    }catch(problem){setError('Could not connect. Check the server address and laptop network. '+problem.message);}
    finally{setBusy('');}
  }
  async function run(){
    clear();setBusy('Reconstructing');
    try{
      const data=await request(server,'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trip_id:trip})});
      setResult(data);
    }catch(problem){setError('Prediction failed. '+problem.message);}
    finally{setBusy('');}
  }
  function showSample(){
    setError('');setSample(true);setResult(evidence.sample);
  }
  return <>
    <header className="top-bar">
      <div className="top-inner">
        <button className="brand" onClick={()=>setTab('map')}><span aria-hidden="true">⌁</span>PathSnap</button>
        <nav aria-label="Dashboard sections">
          <button onClick={()=>setTab('map')} aria-current={tab==='map'?'page':undefined}>Run demo</button>
          <button onClick={()=>setTab('reports')} aria-current={tab==='reports'?'page':undefined}>Model reports</button>
        </nav>
        <span className="team">Team 05</span>
      </div>
    </header>
    <main>
      {tab==='reports'?<Reports evidence={evidence}/>:<div className="view">
        <section className="intro">
          <div><h1>Route reconstruction</h1><p>Match noisy GPS samples to a connected road sequence.</p></div>
          <div className="intro-actions"><span className={'pill status '+(server||sample?'online':'')}><i/>{sample?'Saved student run':mode}</span>
            <button disabled={!!busy} onClick={showSample}>Load sample</button></div>
        </section>
        <section className="connection panel">
          <form onSubmit={connect}>
            <label>Model server<input value={address} type="url" onChange={e=>{setAddress(e.target.value);setTrips([]);setServer('');setMode('Disconnected');clear();}} disabled={!!busy} required/></label>
            <button disabled={!!busy}>{busy==='Connecting'?'Connecting…':'Connect'}</button>
          </form>
          <div className="trip-input">
            <label>Evaluation trip<select disabled={!trips.length||!!busy} value={trip} onChange={e=>{setTrip(e.target.value);clear();}}>{!trips.length&&<option value="">Connect to select</option>}{trips.map(id=><option key={id}>{id}</option>)}</select></label>
            <button className="primary" disabled={!trips.length||!!busy} onClick={run}>{busy==='Reconstructing'?'Reconstructing…':'Reconstruct route'} <span aria-hidden="true">→</span></button>
          </div>
        </section>
        {(error||busy)&&<div className={'notice '+(error?'error':'')} role="status">{!error&&<i className="spinner"/>}{error||busy+'…'}</div>}
        <div className="workspace-grid">
          <RouteMap key={(result?.trip_id||'none')+'-'+sample} result={result}/>
          <aside className="panel run-summary">
            <header className="panel-header"><div><h2>Run summary</h2><p aria-live="polite">{result?'Completed trip '+result.trip_id:'Awaiting a prediction'}</p></div></header>
            <dl className="stats-grid">
              <div><dt>GPS observations</dt><dd>{result?.gps_lonlat.length.toLocaleString()||'—'}</dd></div>
              <div><dt>Compute time</dt><dd>{result?result.timing_seconds.toFixed(2)+' s':'—'}</dd></div>
              <div><dt>Decoded Top-1</dt><dd>{pct(score?.top1_accuracy)}</dd></div>
              <div><dt>Macro F1</dt><dd>{number(score?.macro_f1)}</dd></div>
            </dl>
            {score&&<div className="metric-comparison">
              <h3>Before and after decoding</h3>
              <table><thead><tr><th>Measure</th><th>Pointwise</th><th>Route</th></tr></thead><tbody>
                <tr><th>Top-1</th><td>{pct(score.pointwise_top1_accuracy)}</td><td>{pct(score.top1_accuracy)}</td></tr>
                <tr><th>Macro F1</th><td>{number(score.pointwise_macro_f1)}</td><td>{number(score.macro_f1)}</td></tr>
                <tr><th>Continuity</th><td>{pct(score.pointwise_direct_continuity)}</td><td>{pct(score.direct_continuity)}</td></tr>
              </tbody></table>
            </div>}
            {result&&!score&&<p className="text-note">Reference scores are unavailable for this trip.</p>}
            {result&&<div className="summary-actions">
              <button onClick={()=>downloadRoute(result)}>Download GeoJSON</button>
              {sample&&typeof location!=='undefined'&&location.protocol!=='file:'&&<a className="button" href="./student_trip_map.html" target="_blank" rel="noopener">Folium map ↗</a>}
            </div>}
          </aside>
        </div>
      </div>}
    </main>
  </>;
}
