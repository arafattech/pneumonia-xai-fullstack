// src/pages/Explain.jsx  —  XAI heatmap explanations

import { Flame, Grid, AlertCircle } from 'lucide-react'
import { usePredict }    from '../hooks/usePredict'
import UploadCard        from '../components/UploadCard'
import HeatmapViewer     from '../components/HeatmapViewer'
import LoadingSpinner    from '../components/LoadingSpinner'

export default function Explain() {
  const {
    preview, gradcam, lime, loading, error,
    selectFile, runGradCAM, runLIME, reset,
  } = usePredict()

  return (
    <main className="container" style={{ padding: '2rem 1.5rem' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '1.8rem', fontWeight: 800, marginBottom: '0.4rem' }}>
          AI <span style={{ color: 'var(--accent)' }}>Explainability</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', maxWidth: '540px', margin: '0 auto' }}>
          Understand <em>why</em> the model made its prediction using Grad-CAM and LIME visual explanations.
        </p>
      </div>

      {/* Upload + buttons */}
      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '1.5rem', alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          <UploadCard preview={preview} onSelect={selectFile} onReset={reset} />

          <button
            className="btn-primary"
            disabled={!preview || loading.gradcam}
            onClick={runGradCAM}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
          >
            {loading.gradcam
              ? <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Generating...</>
              : <><Flame size={18} /> Generate Grad-CAM</>}
          </button>

          <button
            className="btn-secondary"
            disabled={!preview || loading.lime}
            onClick={runLIME}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
          >
            {loading.lime
              ? <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Running LIME...</>
              : <><Grid size={18} /> Generate LIME</>}
          </button>

          {error && (
            <div className="error-box" style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
              <AlertCircle size={16} style={{ marginTop: '0.1rem', flexShrink: 0 }} />
              {error}
            </div>
          )}

          {/* XAI method explainers */}
          <div className="card" style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            <p style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '0.5rem' }}>
              🔥 Grad-CAM
            </p>
            <p style={{ marginBottom: '0.8rem' }}>
              Highlights regions the CNN attended to by tracing gradients back to the last convolutional layer.
              Red = high attention, blue = low.
            </p>
            <p style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '0.5rem' }}>
              🟡 LIME
            </p>
            <p>
              Perturbs image superpixels to find which regions most changed the prediction.
              Yellow boundaries = regions supporting the diagnosis.
            </p>
          </div>
        </div>

        {/* Right: heatmap results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {!preview && (
            <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '4rem 2rem' }}>
              <p style={{ fontSize: '2.5rem', marginBottom: '0.8rem' }}>🔬</p>
              <p>Upload an X-ray, then click<br /><strong style={{ color: 'var(--text)' }}>Generate Grad-CAM</strong> or <strong style={{ color: 'var(--text)' }}>Generate LIME</strong></p>
            </div>
          )}

          {loading.gradcam && <LoadingSpinner text="Computing Grad-CAM heatmap..." />}
          {!loading.gradcam && gradcam && (
            <HeatmapViewer
              title="Grad-CAM Explanation"
              data={gradcam}
              explanation={gradcam.explanation}
              type="gradcam"
            />
          )}

          {loading.lime && <LoadingSpinner text="Running LIME (takes ~20s)..." />}
          {!loading.lime && lime && (
            lime.available
              ? <HeatmapViewer
                  title="LIME Explanation"
                  data={lime}
                  explanation={lime.explanation}
                  type="lime"
                />
              : <div className="error-box">LIME not available: {lime.error}</div>
          )}
        </div>
      </div>
    </main>
  )
}
