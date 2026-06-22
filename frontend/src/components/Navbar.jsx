// src/components/Navbar.jsx

import { Link, useLocation } from 'react-router-dom'
import { Activity } from 'lucide-react'

const links = [
  { to: '/',        label: 'Predict' },
  { to: '/explain', label: 'Explain' },
  { to: '/about',   label: 'About' },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav style={{
      background: 'var(--bg2)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      <div className="container" style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
        padding: '1rem 1.5rem',
      }}>
        {/* Logo */}
        <Link to="/" style={{
          display: 'flex', alignItems: 'center', gap: '0.6rem',
          color: 'var(--text)', fontWeight: 700, fontSize: '1.1rem',
        }}>
          <Activity size={22} color="var(--accent)" />
          Pneumonia<span style={{ color: 'var(--accent)' }}>XAI</span>
        </Link>

        {/* Nav links */}
        <div style={{ display: 'flex', gap: '0.3rem' }}>
          {links.map(({ to, label }) => (
            <Link key={to} to={to} style={{
              padding: '0.45rem 1rem',
              borderRadius: 'var(--radius)',
              fontSize: '0.9rem',
              fontWeight: pathname === to ? 600 : 400,
              color: pathname === to ? 'var(--accent)' : 'var(--text-muted)',
              background: pathname === to ? 'rgba(79,142,247,0.12)' : 'transparent',
              transition: 'all 0.2s',
            }}>
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
