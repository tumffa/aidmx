import React, { useEffect, useState, useRef } from 'react'

const API = (path) => `http://localhost:5000${path}`

function ScalarEditor({value, onChange}){
  const [text, setText] = useState(typeof value === 'string' ? value : String(value))
  useEffect(()=>{
    setText(typeof value === 'string' ? value : String(value))
  }, [value])

  const coerce = (t) => {
    // Coerce pure integer strings to numbers, otherwise keep as string
    return /^[-]?\d+$/.test(t) ? parseInt(t, 10) : t
  }

  const onTextChange = (t) => {
    setText(t)
    onChange(coerce(t))
  }

  return (
    <input
      value={text}
      onChange={e=> onTextChange(e.target.value)}
      style={{padding:'6px 8px', borderRadius:4}}
    />
  )
}

function NodeEditor({obj, onChange, showAddMode = true}){
  const [open, setOpen] = useState({})
  const [newKey, setNewKey] = useState('')
  const [newVal, setNewVal] = useState('')
  const [addMode, setAddMode] = useState('text') // 'text' | 'json'
  const addModeNameRef = useRef(`add-mode-${Math.random().toString(36).slice(2)}`)

  const setPath = (key, updater) => {
    const next = Array.isArray(obj) ? [...obj] : {...obj}
    const cur = next[key]
    next[key] = updater(cur)
    onChange(next)
  }

  const addKey = () => {
    if(!newKey) return
    let val
    if(addMode === 'json'){
      // Add a new dictionary/object under this key
      val = {}
    } else {
      // string/int mode: coerce pure integers, else keep string
      if(/^[-]?\d+$/.test(newVal)) val = parseInt(newVal, 10)
      else val = newVal
    }
    const next = Array.isArray(obj) ? [...obj] : {...obj}
    next[newKey] = val
    onChange(next)
    setNewKey(''); setNewVal('')
  }

  return (
    <div className="tree">
      <div className="node">
        <div className="controls">
          <input
            placeholder="new key"
            value={newKey}
            onChange={e=>setNewKey(e.target.value)}
            style={{padding:'6px 8px', borderRadius:4}}
          />
          <input
            placeholder="value (string or int)"
            value={newVal}
            onChange={e=>setNewVal(e.target.value)}
            style={{padding:'6px 8px', borderRadius:4}}
          />
          {showAddMode && (
            <>
              <label style={{display:'flex', alignItems:'center', gap:4}}>
                <input type="radio" name={addModeNameRef.current} checked={addMode==='json'} onChange={()=> setAddMode('json')} /> json
              </label>
              <label style={{display:'flex', alignItems:'center', gap:4}}>
                <input type="radio" name={addModeNameRef.current} checked={addMode==='text'} onChange={()=> setAddMode('text')} /> string/int
              </label>
            </>
          )}
          <button className="btn" onClick={addKey}>Add</button>
        </div>
        {Object.entries(obj||{}).map(([k,v])=>{
          const isObj = v && typeof v === 'object'
          return (
            <div key={k} className="node">
              <div style={{display:'flex', alignItems:'center', gap:8}}>
                <span
                  className="key"
                  style={{cursor: isObj ? 'pointer' : 'default', color:'#fff'}}
                  onClick={isObj ? ()=> setOpen(o=>({...o,[k]:!o[k]})) : undefined}
                >
                  {isObj ? (open[k] ? '▼ ' : '▶ ') : ''}{k}
                </span>
                <button
                  className="btn secondary"
                  style={{background:'#e53935', color:'#fff', border:'none', borderRadius:'4px', padding:'2px 6px', fontSize:'12px'}}
                  onClick={(e)=>{
                    e.stopPropagation()
                    const next = Array.isArray(obj) ? [...obj] : {...obj}
                    if(Array.isArray(obj)){
                      const idx = parseInt(k, 10)
                      if(!isNaN(idx)) next.splice(idx, 1)
                    } else {
                      delete next[k]
                    }
                    onChange(next)
                  }}
                >
                  remove
                </button>
              </div>
              {isObj ? (
                open[k] && <NodeEditor obj={v} onChange={(nv)=> setPath(k, ()=> nv)} showAddMode={showAddMode} />
              ) : (
                <div className="controls">
                  <ScalarEditor value={v} onChange={(txt)=>{
                    let parsed
                    // set via ScalarEditor controlled parse; txt may already be coerced
                    parsed = txt
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
  const [openGroups, setOpenGroups] = useState({})
  const [openFixtures, setOpenFixtures] = useState({}) // key: `${group}/${fixtureId}` -> bool

  const fetchUniverse = async ()=>{
    const res = await fetch(API('/api/universe')).then(r=>r.json())
    const uni = res.universe || {}
    setUniverse(Object.fromEntries(Object.entries(uni).filter(([k])=>k!== 'size')))
    setSize(String(uni.size ?? 512))
    // keep current open states; do not reset
  }

  useEffect(()=>{
    // Ensure groups are initially collapsed
    fetchUniverse().then(()=>{
      setOpenGroups({})
    })
  }, [])

  const save = async ()=>{
    if(!universe) return
    setSaving(true)
    const payload = { size: parseInt(size||'512'), ...universe }
    await fetch(API('/api/universe'),{
      method:'PUT', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({universe: payload})
    })
    setSaving(false)
  }

  const restore = async ()=>{
    await fetchUniverse()
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

  const deleteGroup = (g)=>{
    if(!window.confirm(`Are you sure you want to delete group ${g}?`)) return
    setUniverse(prev=>{
      const next = {...prev}
      delete next[g]
      return next
    })
    // Clear open states for this group
    setOpenGroups(prev=>{
      const { [g]: _omit, ...rest } = prev
      return rest
    })
    setOpenFixtures(prev=>{
      const rest = {}
      Object.keys(prev).forEach(k=>{
        if(!k.startsWith(`${g}/`)) rest[k] = prev[k]
      })
      return rest
    })
  }

  const deleteFixture = (g, fk)=>{
    if(!window.confirm(`Are you sure you want to delete fixture ${fk} in group ${g}?`)) return
    setUniverse(prev=>{
      const next = {...prev}
      const fixtures = {...(next[g] || {})}
      delete fixtures[fk]
      next[g] = fixtures
      return next
    })
    const key = `${g}/${fk}`
    setOpenFixtures(prev=>{
      const { [key]: _omit, ...rest } = prev
      return rest
    })
  }

  const toggleGroup = (g)=>{
    setOpenGroups(prev=> ({...prev, [g]: !prev[g]}))
  }

  const toggleFixture = (g, fk)=>{
    const key = `${g}/${fk}`
    setOpenFixtures(prev=> ({...prev, [key]: !prev[key]}))
  }

  if(universe == null){ return <div>Loading...</div> }

  return (
    <div>
      {/* Top bar */}
      <div className="row inline" style={{alignItems:'center', gap:12, marginBottom:12}}>
        <button className="btn" onClick={save} disabled={saving}>{saving? 'Saving...' : 'Save'}</button>
        <button className="btn secondary" onClick={restore}>Restore</button>
        <button className="btn secondary" onClick={addGroup}>Add Group</button>
        <div style={{marginLeft:'auto', display:'flex', alignItems:'center', gap:6}}>
          <label htmlFor="uni-size">universe_size</label>
          <input id="uni-size" value={size} onChange={e=>setSize(e.target.value)} style={{width:60}} />
        </div>
      </div>

      {/* Groups */}
      {Object.entries(universe).map(([g, fixtures])=> {
        const gOpen = !!openGroups[g]
        return (
          <div key={g} className="node">
            <div
              className="inline-buttons"
              style={{justifyContent:'flex-start', alignItems:'center', gap:8}}
            >
              <span
                style={{cursor:'pointer', fontWeight:'bold'}}
                onClick={()=> toggleGroup(g)}
              >
                {gOpen ? '▼ ' : '▶ '}Group: {g}
              </span>
            </div>
            {gOpen && (
              <div style={{display:'flex', justifyContent:'flex-start', alignItems:'center', gap:8, margin:'6px 0'}}>
                <button
                  className="btn secondary"
                  onClick={()=> addFixture(g)}
                >
                  Add Fixture
                </button>
                <button
                  className="btn secondary"
                  style={{background:'#e53935', color:'#fff', border:'none'}}
                  onClick={()=> deleteGroup(g)}
                >
                  Delete Group
                </button>
              </div>
            )}
            {gOpen && Object.entries(fixtures||{}).map(([fk, fv])=> {
              const fkKey = `${g}/${fk}`
              const fOpen = !!openFixtures[fkKey]
              return (
                <div key={fk} style={{marginLeft:16}}>
                  <div style={{display:'flex', alignItems:'center', gap:8}}>
                    <span
                      style={{cursor:'pointer'}}
                      onClick={()=> toggleFixture(g, fk)}
                    >
                      {fOpen ? '▼ ' : '▶ '}Fixture {fk}
                    </span>
                    <button
                      className="btn secondary"
                      style={{background:'#e53935', color:'#fff', border:'none', borderRadius:'4px', padding:'2px 6px', fontSize:'12px'}}
                      onClick={(e)=>{ e.stopPropagation(); deleteFixture(g, fk) }}
                    >
                      remove
                    </button>
                  </div>
                  {fOpen && (
                    <NodeEditor
                      obj={fv}
                      onChange={(nv)=> setUniverse(prev=> ({...prev, [g]: {...prev[g], [fk]: nv}}))}
                      showAddMode={false}
                    />
                  )}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}