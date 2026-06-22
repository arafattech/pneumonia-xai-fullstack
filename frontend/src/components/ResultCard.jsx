// src/components/ResultCard.jsx

import { CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react'
import { RadialBarChart, RadialBar, ResponsiveContainer, Tooltip } from 'recharts'

export default function ResultCard({ prediction }) {
  if (!prediction) return null

  const isPneumonia = prediction.predicted_class === 'PNEUMONIA'
  const color       = isPneumonia ? 'var(--pneumonia)' : 'var(--normal)'
  const Icon        = isPneumonia ? AlertTriangle : CheckCircle
  const pct         = Math.round(prediction.confidence * 100)

  const radialData = [{
    name: prediction.predicted_class,
    value: pct,
    fill: color,
  }]

  return (
    <div className="card" style={{ borderLeft: `4px solid ${color}` }}>
      <h2 style={{ marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <TrendingUp size={20} color="var(--accent)" />
        Prediction Result
      </h2>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
        {/* Radial confidence gauge */}
        <div style={{ width: 120, height: 120 }}>
          <ResponsiveContainer>
            <RadialBarChart
              innerRadius="65%" outerRadius="100%"
              data={radialData} startAngle={90} endAngle={-270}
            >
              <RadialBar dataKey="value" cornerRadius={6} background={{ fill: 'var(--bg3)' }} />
              <Tooltip formatter={v => `${v}%`} />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        {/* Text result */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', marginBottom: '0.5rem' }}>
            <Icon size={28} color={color} />
            <span className={isPneumonia ? 'tag-pneumonia' : 'tag-normal'} style={{ fontSize: '1rem' }}>
              {prediction.predicted_class}
            </span>
          </div>
          <p style={{ fontSize: '2rem', fontWeight: 800, color, lineHeight: 1 }}>
            {pct}%
            <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
              confidence
            </span>
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.4rem' }}>
            Model: {prediction.model_name}
          </p>
        </div>
      </div>

      {/* Probability bars */}
      <div style={{ marginTop: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {Object.entries(prediction.probabilities).map(([cls, prob]) => {
          const barColor = cls === 'PNEUMONIA' ? 'var(--pneumonia)' : 'var(--normal)'
          return (
            <div key={cls}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                <span>{cls}</span>
                <span style={{ fontWeight: 600 }}>{(prob * 100).toFixed(1)}%</span>
              </div>
              <div style={{ background: 'var(--bg3)', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
                <div style={{
                  width: `${prob * 100}%`, height: '100%',
                  background: barColor, borderRadius: '4px',
                  transition: 'width 0.6s ease',
                }} />
              </div>
            </div>
          )
        })}
      </div>

      {isPneumonia && (
        <div style={{
          marginTop: '1rem',
          padding: '0.8rem 1rem',
          background: 'rgba(231,76,60,0.1)',
          border: '1px solid rgba(231,76,60,0.3)',
          borderRadius: 'var(--radius)',
          fontSize: '0.85rem',
          color: 'var(--text-muted)',
        }}>
          ⚠️ This is an AI screening tool only. Clinical diagnosis requires a qualified radiologist.
        </div>
      )}
    </div>
  )
}
