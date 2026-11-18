import React,{useEffect,useMemo,useRef,useState} from 'react';

const layers=[['network','Roads'],['route','Reconstructed'],['gps','GPS samples'],['pointwise','Pointwise']];
const W=960,H=450;

export default function RouteMap({result}){
  const [zoom,setZoom]=useState(1),[pan,setPan]=useState([0,0]);
  const [visible,setVisible]=useState({network:true,route:true,gps:true,pointwise:false});
  const [frame,setFrame]=useState(0),[playing,setPlaying]=useState(false);
  const anchor=useRef(null);
  const count=result?.gps_lonlat.length||0;
  const geometry=useMemo(()=>{
    if(!count)return null;
    let west=Infinity,east=-Infinity,south=Infinity,north=-Infinity;
    for(const [x,y] of [...result.gps_lonlat,...result.route_lines_lonlat.flat()]){
      west=Math.min(west,x);east=Math.max(east,x);south=Math.min(south,y);north=Math.max(north,y);
    }
    const cosine=Math.max(.01,Math.cos((south+north)*Math.PI/360));
    const scale=Math.min((W-120)/Math.max(.00009,(east-west)*cosine),(H-100)/Math.max(.00009,north-south));
    const project=([x,y])=>[W/2+(x-(west+east)/2)*cosine*scale,H/2-(y-(north+south)/2)*scale];
    const lines=items=>(items||[]).map(line=>line.map(point=>project(point).join(',')).join(' '));
    return {gps:result.gps_lonlat.map(project),network:lines(result.road_network_lonlat),route:lines(result.route_lines_lonlat),
      pointwise:lines(result.pointwise_segments_lonlat),metresPerPixel:111195/scale};
  },[result,count]);
  useEffect(()=>{
    if(!playing||!count)return;
    const timer=setInterval(()=>setFrame(previous=>Math.min(previous+Math.max(1,Math.ceil(count/190)),count-1)),55);
    return ()=>clearInterval(timer);
  },[playing,count]);
  useEffect(()=>{if(frame===count-1)setPlaying(false);},[frame,count]);

  function dragStart(event){
    if(!geometry||event.button!==0)return;
    anchor.current=[event.clientX,event.clientY,Math.min(event.currentTarget.clientWidth/W,event.currentTarget.clientHeight/H)];
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function drag(event){
    if(!anchor.current)return;
    const [x,y,scale]=anchor.current;
    anchor.current=[event.clientX,event.clientY,scale];
    setPan(([left,top])=>[left+(event.clientX-x)/scale,top+(event.clientY-y)/scale]);
  }
  const marker=geometry?.gps[frame];
  const rough=geometry?.metresPerPixel*75/zoom;
  const step=rough?10**Math.floor(Math.log10(rough)):0;
  const distance=step?[1,2,5].filter(s=>s*step<=rough).at(-1)*step:0;
  const scaleLabel=distance>=1000?distance/1000+' km':+distance.toPrecision(2)+' m';
  return <section className="panel atlas">
    <header className="panel-header"><div><h2>Route map</h2><p>{result?'Trip '+result.trip_id+' · '+result.route_lines_lonlat.length+' connected sections':'Select a trip to display road geometry.'}</p></div>
      {result&&<span className="pill">{result.bridge_count} connecting arcs</span>}</header>
    <div className="map-wrap">
      <svg className={'route-map '+(!geometry?'empty-map':'')} aria-label="Road network, route and GPS points" viewBox={'0 0 '+W+' '+H} onPointerDown={dragStart} onPointerMove={drag} onPointerUp={()=>anchor.current=null} onPointerCancel={()=>anchor.current=null} onLostPointerCapture={()=>anchor.current=null}>
        <defs><pattern id="map-grid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="#dce5dd"/></pattern></defs>
        <rect width={W} height={H} fill="#edf1e8"/><rect width={W} height={H} fill="url(#map-grid)"/>
        {geometry&&<g className="map-result" transform={'translate('+(W/2+pan[0])+' '+(H/2+pan[1])+') scale('+zoom+') translate('+(-W/2)+' '+(-H/2)+')'}>
          {['network','pointwise','route'].map(key=>visible[key]&&<g key={key} className={'map-'+key}>{geometry[key].map((points,i)=><polyline key={i} points={points}/>)}</g>)}
          {visible.gps&&<g className="map-gps">{geometry.gps.map(([x,y],i)=><circle key={i} cx={x} cy={y} r={1.5/zoom}/>)}</g>}
          {marker&&visible.gps&&<g className="playhead" transform={'translate('+marker.join(' ')+') scale('+1/zoom+')'}><ellipse rx="10" ry="10"/><ellipse rx="4" ry="4"/></g>}
        </g>}
        {distance>0&&<g className="map-scale" transform="translate(26 411)"><path d={'M0 4V0H'+distance*zoom/geometry.metresPerPixel+'V4'}/><text y="-8">{scaleLabel}</text></g>}
      </svg>
      {!geometry&&<div className="map-placeholder"><span>⌁</span><b>Connect a model server or load the sample</b></div>}
      {geometry&&<><div className="compass" aria-label="North">N<br/>↑</div>
        <div className="map-controls">
          <button aria-label="Zoom in" onClick={()=>setZoom(z=>Math.min(16,z*1.35))}>+</button>
          <button aria-label="Zoom out" onClick={()=>setZoom(z=>Math.max(.7,z/1.35))}>−</button>
          <button aria-label="Fit route" onClick={()=>{setPan([0,0]);setZoom(1);}}>⛶</button>
        </div></>}
    </div>
    {geometry&&<>
      <div className="map-layers" aria-label="Map layers">{layers.map(([key,label])=><button key={key} className={'layer '+key} aria-pressed={visible[key]} disabled={!geometry[key].length} onClick={()=>setVisible({...visible,[key]:!visible[key]})}><i/>{label}</button>)}</div>
      <div className="replay">
        <button aria-label={playing?'Pause GPS playback':'Play GPS playback'} onClick={()=>{if(frame===count-1)setFrame(0);setPlaying(!playing);}}>{playing?'Ⅱ':'▶'}</button>
        <label>GPS playback<input aria-label="GPS sample" type="range" min="0" max={count-1} value={frame} onChange={event=>{setPlaying(false);setFrame(+event.target.value);}}/></label>
        <span>{frame+1} / {count.toLocaleString()}</span>
      </div>
    </>}
  </section>;
}
