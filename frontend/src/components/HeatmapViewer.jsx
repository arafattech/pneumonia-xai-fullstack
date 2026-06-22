// src/components/HeatmapViewer.jsx

import { useState } from 'react'
import { Eye, Flame, Layers } from 'lucide-react'

const TABS = [
  { key: 'original_b64', label: 'Original',  Icon: Eye },
  { key: 'heatmap_b64',  label: 'Heatmap',   Icon: Flame },
  { key: 'overlay_b64',  label: 'Overlay',   Icon: Layers },
]

export default function HeatmapViewer({ title, data, explanation, type = 'gradcam' }) {
  const [active, setActive] = useState('overlay_b64')

  if (!data) return null

  const isPneumonia  = data.predicted_class === 'PNEUMONIA'
  const tagColor     = isPneumonia ? 'var(--pneumonia)' : 'var(--normal)'
  const tagBg        = isPneumonia ? '#3d1a1a' : '#1a3d2b'

  return (
    <div className="card">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h3 style={{ fontWeight: 700 }}>{title}</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ background: tagBg, color: tagColor, padding: '0.2rem 0.7rem', borderRadius: '999px', fontSize: '0.8rem', fontWeight: 700 }}>
            {data.predicted_class}
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            {(data.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Tab switcher (Grad-CAM only) */}
      {type === 'gradcam' && (
        <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1rem' }}>
          {TABS.map(({ key, label, Icon }) => (
            <button
              key={key}
              onClick={() => setActive(key)}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                padding: '0.4rem 0.9rem',
                background: active === key ? 'var(--accent)' : 'var(--bg3)',
                color: active === key ? 'white' : 'var(--text-muted)',
                borderRadius: 'var(--radius)',
                fontSize: '0.82rem',
                border: '1px solid var(--border)',
              }}
            >
              <Icon size={13} /> {label}
            </button>
          ))}
        </div>
      )}

      {/* Image display */}
      {type === 'gradcam' ? (
        <img
          src={`data:image/png;base64,${data[active]}`}
          alt={active}
          style={{ width: '100%', borderRadius: 'var(--radius)', maxHeight: '340px', objectFit: 'contain', background: '#000' }}
        />
      ) : (
        /* LIME: side by side */
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
          {[
            { key: 'all_b64',       label: 'Original' },
            { key: 'positive_b64',  label: 'Supporting regions' },
          ].map(({ key, label }) => data[key] && (
            <div key={key}>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.3rem', textAlign: 'center' }}>{label}</p>
              <img
                src={`data:image/png;base64,${data[key]}`}
                alt={label}
                style={{ width: '100%', borderRadius: 'var(--radius)', background: '#000' }}
              />
            </div>
          ))}
        </div>
      )}

      {/* Explanation text */}
      {explanation && (
        <div style={{
          marginTop: '1rem',
          padding: '0.8rem 1rem',
          background: 'rgba(79,142,247,0.08)',
          border: '1px solid rgba(79,142,247,0.2)',
          borderRadius: 'var(--radius)',
          fontSize: '0.85rem',
          color: 'var(--text-muted)',
          lineHeight: 1.6,
        }}>
          💡 {explanation}
        </div>
      )}

      {/* Heatmap color legend */}
      {type === 'gradcam' && active !== 'original_b64' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.8rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          <span>Low attention</span>
          <div style={{ flex: 1, height: '8px', borderRadius: '4px', background: 'linear-gradient(to right, #00008b, #0000ff, #00ffff, #00ff00, #ffff00, #ff8000, #ff0000)' }} />
          <span>High attention</span>
        </div>
      )}
    </div>
  )
}
