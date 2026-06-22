// src/hooks/usePredict.js  —  API call state management

import { useState, useCallback } from 'react'
import { predictImage, getGradCAM, getLIME } from '../utils/api'

export function usePredict() {
  const [file,        setFile]        = useState(null)
  const [preview,     setPreview]     = useState(null)
  const [prediction,  setPrediction]  = useState(null)
  const [gradcam,     setGradcam]     = useState(null)
  const [lime,        setLime]        = useState(null)
  const [loading,     setLoading]     = useState({ predict: false, gradcam: false, lime: false })
  const [error,       setError]       = useState(null)

  const selectFile = useCallback((f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setPrediction(null)
    setGradcam(null)
    setLime(null)
    setError(null)
  }, [])

  const runPredict = useCallback(async () => {
    if (!file) return
    setLoading(l => ({ ...l, predict: true }))
    setError(null)
    try {
      const result = await predictImage(file)
      setPrediction(result)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(l => ({ ...l, predict: false }))
    }
  }, [file])

  const runGradCAM = useCallback(async () => {
    if (!file) return
    setLoading(l => ({ ...l, gradcam: true }))
    setError(null)
    try {
      const result = await getGradCAM(file)
      setGradcam(result)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(l => ({ ...l, gradcam: false }))
    }
  }, [file])

  const runLIME = useCallback(async () => {
    if (!file) return
    setLoading(l => ({ ...l, lime: true }))
    setError(null)
    try {
      const result = await getLIME(file)
      setLime(result)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(l => ({ ...l, lime: false }))
    }
  }, [file])

  const reset = useCallback(() => {
    setFile(null); setPreview(null); setPrediction(null)
    setGradcam(null); setLime(null); setError(null)
  }, [])

  return {
    file, preview, prediction, gradcam, lime,
    loading, error,
    selectFile, runPredict, runGradCAM, runLIME, reset
  }
}
