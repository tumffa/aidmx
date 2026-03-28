import React, { useEffect, useRef, useState } from 'react'
import AnalyzeTab from './components/AnalyzeTab'
import PlayTab from './components/PlayTab'
import UniverseTab from './components/UniverseTab'

const API = (path) => `http://localhost:5000${path}`

export default function App(){
  const [active, setActive] = useState('Analyze')
  const [cursor, setCursor] = useState(0)
  const [lines, setLines] = useState([])
  const [status, setStatus] = useState({analyze_running:false, play_running:false})
  const logRef = useRef(null)

  useEffect(()=>{
    const id = setInterval(async ()=>{
      try{
        const s = await fetch(API('/api/status')).then(r=>r.json())
        setStatus(s)
      }catch{}
      try{
        const res = await fetch(API(`/api/logs?cursor=${cursor}`)).then(r=>r.json())
        if(res?.lines?.length){
          setLines(prev=>[...prev, ...res.lines])
          setCursor(res.next_cursor)
          if(logRef.current){
            logRef.current.scrollTop = logRef.current.scrollHeight
          }
        }
      }catch{}
    }, 800)
    return ()=> clearInterval(id)
  }, [cursor])

  return (
    <div className="app">
      <div className="log">
        <h2>Logs</h2>
        <pre ref={logRef}>{lines.map(l=>l.text).join('\n')}</pre>
      </div>
      <div className="content">
        <div className="topbar">
          <div className={`tab ${active==='Analyze'?'active':''}`} onClick={()=>setActive('Analyze')}>Analyze</div>
          <div className={`tab ${active==='Play'?'active':''}`} onClick={()=>setActive('Play')}>Play</div>
          <div className={`tab ${active==='Universe'?'active':''}`} onClick={()=>setActive('Universe')}>Universe</div>
        </div>
        <div className="panel">
          {active==='Analyze' && <AnalyzeTab status={status} />}
          {active==='Play' && <PlayTab status={status} />}
          {active==='Universe' && <UniverseTab />}
        </div>
      </div>
    </div>
  )
}
