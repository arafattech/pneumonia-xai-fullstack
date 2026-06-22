// src/utils/api.js  —  Axios API client

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
})

// Grad-CAM + LIME slow on CPU — use longer timeout
const slowApi = axios.create({
  baseURL: BASE_URL,
  timeout: 300000,  // 5 min
})

// ── Health ────────────────────────────────────────────────────
export const checkHealth = () =>
  api.get('/health').then(r => r.data)

// ── Prediction ────────────────────────────────────────────────
export const predictImage = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }).then(r => r.data)
}

// ── Grad-CAM ──────────────────────────────────────────────────
export const getGradCAM = (file) => {
  const form = new FormData()
  form.append('file', file)
  return slowApi.post('/explain/gradcam', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }).then(r => r.data)
}

// ── LIME ──────────────────────────────────────────────────────
export const getLIME = (file) => {
  const form = new FormData()
  form.append('file', file)
  return slowApi.post('/explain/lime', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }).then(r => r.data)
}
