import React, { useState } from 'react'

const API = (path) => `http://localhost:5000${path}`

export default function AnalyzeTab({status}){
  const [audio, setAudio] = useState('')
  const [strobe, setStrobe] = useState(false)
  const [simple, setSimple] = useState(false)
  const [delay, setDelay] = useState('1')
  const [lag, setLag] = useState('0.8955')

  const start = async ()=>{
    if(!audio) return
    await fetch(API('/api/analyze'),{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({audio_name: audio, strobes: strobe, simple, qlc_delay: parseFloat(delay||'0'), qlc_lag: parseFloat(lag||'1')})
    })
  }
  const cancel = async ()=>{
    await fetch(API('/api/cancel/analyze'),{method:'POST'})
  }

  return (
    <div>
      <div className="row"><label>Song name (without extension)</label>
        <input value={audio} onChange={e=>setAudio(e.target.value)} placeholder="e.g. everlong" />
      </div>
      <div className="row inline">
        <label><input type="checkbox" checked={strobe} onChange={e=>setStrobe(e.target.checked)} /> Enable strobe (-st)</label>
        <label><input type="checkbox" checked={simple} onChange={e=>setSimple(e.target.checked)} /> Enable simple (-si)</label>
      </div>
      <div className="row inline">
        <div style={{flex:1}}>
          <label>QLC delay seconds (-d)</label>
          <input value={delay} onChange={e=>setDelay(e.target.value)} />
        </div>
        <div style={{flex:1}}>
          <label>QLC lag scale (-l)</label>
          <input value={lag} onChange={e=>setLag(e.target.value)} />
        </div>
      </div>
      <div className="inline-buttons">
        <button className="btn" onClick={start} disabled={status.analyze_running}>Analyze file</button>
        <button className="btn secondary" onClick={cancel} disabled={!status.analyze_running}>Cancel</button>
      </div>
    </div>
  )
}
