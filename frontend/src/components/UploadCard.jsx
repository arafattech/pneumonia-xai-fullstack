// src/components/UploadCard.jsx

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X } from 'lucide-react'

export default function UploadCard({ preview, onSelect, onReset }) {
  const onDrop = useCallback(files => {
    if (files[0]) onSelect(files[0])
  }, [onSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [] },
    maxFiles: 1,
  })

  return (
    <div className="card">
      <h2 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <ImageIcon size={20} color="var(--accent)" />
        Upload X-Ray Image
      </h2>

      {!preview ? (
        <div {...getRootProps()} style={{
          border: `2px dashed ${isDragActive ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding: '2.5rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: isDragActive ? 'rgba(79,142,247,0.06)' : 'var(--bg3)',
          transition: 'all 0.2s',
        }}>
          <input {...getInputProps()} />
          <Upload size={36} color="var(--accent)" style={{ margin: '0 auto 1rem' }} />
          <p style={{ fontWeight: 600, marginBottom: '0.3rem' }}>
            {isDragActive ? 'Drop it here!' : 'Drag & drop your X-ray'}
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            or click to browse — JPEG / PNG, max 10 MB
          </p>
        </div>
      ) : (
        <div style={{ position: 'relative' }}>
          <img
            src={preview}
            alt="X-ray preview"
            style={{
              width: '100%', maxHeight: '320px',
              objectFit: 'contain',
              borderRadius: 'var(--radius)',
              background: '#000',
            }}
          />
          <button
            onClick={onReset}
            style={{
              position: 'absolute', top: '0.6rem', right: '0.6rem',
              background: 'rgba(0,0,0,0.7)',
              color: 'white',
              borderRadius: '50%',
              width: '32px', height: '32px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1px solid var(--border)',
            }}
          >
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  )
}
