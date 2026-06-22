// src/pages/Home.jsx  —  Upload + Predict page

import { Zap, AlertCircle } from 'lucide-react'
import { usePredict } from '../hooks/usePredict'
import UploadCard     from '../components/UploadCard'
import ResultCard     from '../components/ResultCard'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Home() {
  const {
    preview, prediction, loading, error,
    selectFile, runPredict, reset,
  } = usePredict()

  return (
    <main className="container" style={{ padding: '2rem 1.5rem' }}>
      {/* Hero */}
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>
          Pneumonia Detection
          <span style={{ color: 'var(--accent)' }}> AI</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', maxWidth: '520px', margin: '0 auto' }}>
          Upload a chest X-ray image and get an instant AI-powered prediction
          using DenseNet121 trained on 5,800+ labeled images.
        </p>
      </div>

      {/* Main layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Left: Upload */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <UploadCard preview={preview} onSelect={selectFile} onReset={reset} />

          <button
            className="btn-primary"
            disabled={!preview || loading.predict}
            onClick={runPredict}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', width: '100%' }}
          >
            {loading.predict
              ? <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Analyzing...</>
              : <><Zap size={18} /> Run Prediction</>}
          </button>

          {error && (
            <div className="error-box" style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
              <AlertCircle size={16} style={{ marginTop: '0.1rem', flexShrink: 0 }} />
              {error}
            </div>
          )}

          {/* Instructions */}
          {!preview && (
            <div className="card" style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              <p style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '0.5rem' }}>How it works</p>
              <ol style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                <li>Upload a chest X-ray (JPEG or PNG)</li>
                <li>Click <strong style={{ color: 'var(--text)' }}>Run Prediction</strong></li>
                <li>View result: NORMAL or PNEUMONIA</li>
                <li>Go to <strong style={{ color: 'var(--text)' }}>Explain</strong> tab for AI heatmaps</li>
              </ol>
            </div>
          )}
        </div>

        {/* Right: Result */}
        <div>
          {loading.predict
            ? <LoadingSpinner text="Running inference..." />
            : prediction
              ? <ResultCard prediction={prediction} />
              : (
                <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '3rem' }}>
                  <p style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>🫁</p>
                  <p>Upload an X-ray image and click<br /><strong style={{ color: 'var(--text)' }}>Run Prediction</strong> to see results</p>
                </div>
              )
          }
        </div>
      </div>

      {/* Disclaimer */}
      <p style={{
        marginTop: '2rem', textAlign: 'center',
        color: 'var(--text-muted)', fontSize: '0.78rem',
      }}>
        ⚠️ Research tool only. Not for clinical use without radiologist review.
        Dataset: Kermany et al. 2018 | Model: DenseNet121
      </p>
    </main>
  )
}
