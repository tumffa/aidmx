import React, { useEffect, useState } from 'react'

const API = (path) => `http://localhost:5000${path}`

function ScalarEditor({value, onChange}){
  const [text, setText] = useState(typeof value === 'string' ? value : JSON.stringify(value))
  useEffect(()=>{
    setText(typeof value === 'string' ? value : JSON.stringify(value))
  }, [value])
  return <input value={text} onChange={e=>{setText(e.target.value); onChange(e.target.value)}} />
}

function NodeEditor({obj, onChange}){
  const [open, setOpen] = useState({})
  const [newKey, setNewKey] = useState('')
  const [newVal, setNewVal] = useState('')

  const setPath = (key, updater) => {
    const next = Array.isArray(obj) ? [...obj] : {...obj}
    const cur = next[key]
    next[key] = updater(cur)
    onChange(next)
  }

  const addKey = () => {
    if(!newKey) return
    let val
    try { val = JSON.parse(newVal) } catch { val = newVal }
    const next = Array.isArray(obj) ? [...obj] : {...obj}
    next[newKey] = val
    onChange(next)
    setNewKey(''); setNewVal('')
  }

  return (
    <div className="tree">
      <div className="node">
        <div className="controls">
          <input placeholder="new key" value={newKey} onChange={e=>setNewKey(e.target.value)} />
          <input placeholder="value (json or text)" value={newVal} onChange={e=>setNewVal(e.target.value)} />
          <button className="btn" onClick={addKey}>Add</button>
        </div>
        {Object.entries(obj||{}).map(([k,v])=>{
          const isObj = v && typeof v === 'object'
          return (
            <div key={k} className="node">
              <div onClick={()=> setOpen(o=>({...o,[k]:!o[k]}))} style={{cursor:'pointer'}}>
                <span className="key">{isObj ? (open[k] ? '▼ ' : '▶ ') : ''}{k}</span>
              </div>
              {isObj ? (
                open[k] && <NodeEditor obj={v} onChange={(nv)=> setPath(k, ()=> nv)} />
              ) : (
                <div className="controls">
                  <ScalarEditor value={v} onChange={(txt)=>{
                    let parsed
                    try{ parsed = JSON.parse(txt) }catch{ parsed = txt }
                    setPath(k, ()=> parsed)
                  }} />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function UniverseTab(){
  const [universe, setUniverse] = useState(null)
  const [size, setSize] = useState('512')
  const [saving, setSaving] = useState(false)

  useEffect(()=>{
    (async ()=>{
      const res = await fetch(API('/api/universe')).then(r=>r.json())
      const uni = res.universe || {}
      setUniverse(Object.fromEntries(Object.entries(uni).filter(([k])=>k!== 'size')))
      setSize(String(uni.size ?? 512))
    })()
  }, [])

  const save = async ()=>{
    if(!universe) return
    setSaving(true)
    const payload = { size: parseInt(size||'512'), ...universe }
    const res = await fetch(API('/api/universe'),{
      method:'PUT', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({universe: payload})
    })
    setSaving(false)
  }

  const addGroup = ()=>{
    const name = prompt('New group name:')
    if(!name) return
    setUniverse(prev=> ({...(prev||{}), [name]: {}}))
  }

  const addFixture = (group)=>{
    setUniverse(prev=>{
      const next = {...prev}
      const fixtures = next[group] || {}
      const nums = Object.keys(fixtures).map(k=> parseInt(k)).filter(n=>!isNaN(n))
      const nid = (nums.length? Math.max(...nums):0) + 1
      fixtures[String(nid)] = {}
      next[group] = {...fixtures}
      return next
    })
  }

  if(universe == null){ return <div>Loading...</div> }

  return (
    <div>
      <div className="row inline">
        <div>
          <label>universe_size</label>
          <input value={size} onChange={e=>setSize(e.target.value)} />
        </div>
        <div className="inline-buttons">
          <button className="btn" onClick={save} disabled={saving}>{saving? 'Saving...' : 'Save'}</button>
          <button className="btn secondary" onClick={addGroup}>Add Group</button>
        </div>
      </div>

      {Object.entries(universe).map(([g, fixtures])=> (
        <div key={g} className="node">
          <div className="inline-buttons" style={{justifyContent:'space-between'}}>
            <strong>Group: {g}</strong>
            <button className="btn secondary" onClick={()=> addFixture(g)}>Add Fixture</button>
          </div>
          {Object.entries(fixtures||{}).map(([fk, fv])=> (
            <div key={fk} style={{marginLeft:16}}>
              <div>Fixture {fk}</div>
              <NodeEditor obj={fv} onChange={(nv)=> setUniverse(prev=> ({...prev, [g]: {...prev[g], [fk]: nv}}))} />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
