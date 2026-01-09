import React, { useState } from 'react'

const API = (path) => `http://localhost:5000${path}`

export default function PlayTab({status}){
  const [audio, setAudio] = useState('')
  const [delay, setDelay] = useState('0.00')
  const [universe, setUniverse] = useState('1')
  const [startAt, setStartAt] = useState('0.0')

  const start = async ()=>{
    if(!audio) return
    await fetch(API('/api/play'),{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({audio_name: audio, delay: parseFloat(delay||'0'), universe: parseInt(universe||'1'), start_at_sec: parseFloat(startAt||'0')})
    })
  }
  const cancel = async ()=>{
    await fetch(API('/api/cancel/play'),{method:'POST'})
  }

  return (
    <div>
      <div className="row"><label>Song name (without extension)</label>
        <input value={audio} onChange={e=>setAudio(e.target.value)} placeholder="e.g. everlong" />
      </div>
      <div className="row inline">
        <div style={{flex:1}}>
          <label>DMX delay seconds (-d)</label>
          <input value={delay} onChange={e=>setDelay(e.target.value)} />
        </div>
        <div style={{flex:1}}>
          <label>OLA universe (-u)</label>
          <input value={universe} onChange={e=>setUniverse(e.target.value)} />
        </div>
        <div style={{flex:1}}>
          <label>Start at seconds (-s)</label>
          <input value={startAt} onChange={e=>setStartAt(e.target.value)} />
        </div>
      </div>
      <div className="inline-buttons">
        <button className="btn" onClick={start} disabled={status.play_running}>Play show</button>
        <button className="btn secondary" onClick={cancel} disabled={!status.play_running}>Cancel</button>
      </div>
    </div>
  )
}
