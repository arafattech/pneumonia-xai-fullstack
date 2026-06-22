// src/pages/About.jsx

import { Github, Database, BookOpen, AlertTriangle } from 'lucide-react'

const Section = ({ icon: Icon, title, children }) => (
  <div className="card" style={{ marginBottom: '1rem' }}>
    <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.8rem', fontSize: '1.1rem' }}>
      <Icon size={18} color="var(--accent)" /> {title}
    </h2>
    {children}
  </div>
)

const Row = ({ label, value }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border)', fontSize: '0.9rem' }}>
    <span style={{ color: 'var(--text-muted)' }}>{label}</span>
    <span style={{ fontWeight: 500 }}>{value}</span>
  </div>
)

export default function About() {
  return (
    <main className="container" style={{ padding: '2rem 1.5rem', maxWidth: '780px' }}>
      <h1 style={{ fontSize: '1.8rem', fontWeight: 800, marginBottom: '0.4rem' }}>About This Project</h1>
      <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
        Explainable AI-based Pneumonia Detection from Chest X-ray Images using Deep Learning and Grad-CAM
      </p>

      <Section icon={BookOpen} title="Project Overview">
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.7 }}>
          This project implements a complete pipeline for automated pneumonia detection from chest X-ray images.
          The system uses transfer learning with DenseNet121 pretrained on ImageNet, fine-tuned on 5,856 pediatric
          chest X-ray images. Explainability is provided through Grad-CAM heatmaps, LIME superpixel explanations,
          and SHAP attributions.
        </p>
      </Section>

      <Section icon={Database} title="Model & Dataset">
        <Row label="Model"            value="DenseNet121 (ImageNet pretrained)" />
        <Row label="Dataset"          value="Kermany et al. 2018 — Kaggle" />
        <Row label="Train images"     value="5,216" />
        <Row label="Test images"      value="624" />
        <Row label="Classes"          value="NORMAL vs PNEUMONIA" />
        <Row label="Image size"       value="224 × 224 px" />
        <Row label="XAI methods"      value="Grad-CAM, LIME, SHAP" />
        <Row label="Backend"          value="FastAPI + PyTorch" />
        <Row label="Frontend"         value="React + Vite" />
      </Section>

      <Section icon={AlertTriangle} title="Limitations">
        <ul style={{ color: 'var(--text-muted)', fontSize: '0.88rem', paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <li>Dataset is pediatric (age 1–5) — may not generalise to adult populations</li>
          <li>Binary classification only — does not distinguish bacterial vs viral pneumonia</li>
          <li>Grad-CAM highlights correlation, not clinical causation</li>
          <li>Not validated by radiologists — do not use for clinical decisions</li>
          <li>Training imbalance (~3:1 PNEUMONIA:NORMAL) may affect NORMAL recall</li>
        </ul>
      </Section>

      <Section icon={Github} title="References">
        <ul style={{ color: 'var(--text-muted)', fontSize: '0.85rem', paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <li>Kermany et al. (2018). Cell. doi:10.1016/j.cell.2018.02.010</li>
          <li>Rajpurkar et al. (2017). CheXNet. arXiv:1711.05225</li>
          <li>Selvaraju et al. (2017). Grad-CAM. ICCV. arXiv:1610.02391</li>
          <li>Ribeiro et al. (2016). LIME. KDD. arXiv:1602.04938</li>
          <li>Lundberg & Lee (2017). SHAP. NeurIPS. arXiv:1705.07874</li>
          <li>Huang et al. (2017). DenseNet. CVPR. arXiv:1608.06993</li>
        </ul>
      </Section>

      <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '2rem' }}>
        For research and educational purposes only.
      </p>
    </main>
  )
}
