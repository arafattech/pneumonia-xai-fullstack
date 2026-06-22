// src/components/LoadingSpinner.jsx

export default function LoadingSpinner({ text = 'Processing...' }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: '1rem', padding: '2rem',
    }}>
      <div className="spinner" />
      <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>{text}</p>
    </div>
  )
}
