import React, { useState } from 'react'
import {
  Zap,
  Activity,
  CheckCircle2,
  Clock,
  Sparkles,
  Sliders,
  Loader2,
  Cpu,
  Layers
} from 'lucide-react'
import { runPrediction } from '../services/api'
import { ModelVersion, PredictionResult } from '../types'

interface PredictLabViewProps {
  models: ModelVersion[]
}

export const PredictLabView: React.FC<PredictLabViewProps> = ({ models }) => {
  const [inputType, setInputType] = useState<'tabular' | 'sequence'>('tabular')
  const [selectedModelId, setSelectedModelId] = useState<number | undefined>(undefined)
  const [features, setFeatures] = useState<number[]>([
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ])
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)

  const featureLabels = [
    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
    'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
    'Radius Error', 'Texture Error', 'Perimeter Error', 'Area Error', 'Smoothness Error',
    'Compactness Error', 'Concavity Error', 'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error',
    'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
    'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
  ]

  const handleRunInference = async () => {
    setIsLoading(true)
    try {
      if (inputType === 'tabular') {
        const res = await runPrediction({
          features: features,
          model_id: selectedModelId,
        })
        setPredictionResult(res)
      } else {
        // Generate (10, 6) sequence
        const seq = Array.from({ length: 10 }).map((_, t) =>
          Array.from({ length: 6 }).map((_, f) => +(Math.sin(t * 0.5 + f) * 1.5 + 2.0).toFixed(3))
        )
        const res = await runPrediction({
          sequence: seq,
          model_id: selectedModelId,
        })
        setPredictionResult(res)
      }
    } catch (err: any) {
      console.error('Inference error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const loadPreset = (type: 'malignant' | 'benign') => {
    if (type === 'malignant') {
      setFeatures([
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
        0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
      ])
    } else {
      setFeatures([
        13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
        0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146,
        0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2,
        0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
      ])
    }
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-display font-bold text-white tracking-tight">
              Real-time Inference Lab
            </h2>
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300 text-xs font-mono">
              &lt;15ms Latency Audit
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Execute sub-millisecond diagnostic predictions with confidence distributions and database telemetry logging
          </p>
        </div>

        {/* Execute Button */}
        <button
          onClick={handleRunInference}
          disabled={isLoading}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer disabled:opacity-50 shrink-0"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Evaluating Model...</span>
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 fill-current" />
              <span>Run Live Inference</span>
            </>
          )}
        </button>
      </div>

      {/* Mode & Preset Controls */}
      <div className="glass-panel p-4 flex flex-wrap items-center justify-between gap-4">
        {/* Input Selector */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setInputType('tabular')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
              inputType === 'tabular'
                ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                : 'bg-slate-900 text-slate-400 hover:text-white'
            }`}
          >
            Tabular (30 Features)
          </button>
          <button
            type="button"
            onClick={() => setInputType('sequence')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
              inputType === 'sequence'
                ? 'bg-purple-500 text-white shadow-md shadow-purple-500/20'
                : 'bg-slate-900 text-slate-400 hover:text-white'
            }`}
          >
            Time-Series Sequence (10×6)
          </button>
        </div>

        {/* Model Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-500">Model:</span>
          <select
            value={selectedModelId ?? ''}
            onChange={(e) => setSelectedModelId(e.target.value ? Number(e.target.value) : undefined)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="">Auto (Production)</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.model_name} {m.is_production ? '[PROD]' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Diagnostic Presets */}
        {inputType === 'tabular' && (
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-slate-500">Presets:</span>
            <button
              onClick={() => loadPreset('malignant')}
              className="px-2.5 py-1 rounded bg-rose-950/80 border border-rose-800 text-rose-300 hover:bg-rose-900/60 transition-colors"
            >
              Sample A (Malignant)
            </button>
            <button
              onClick={() => loadPreset('benign')}
              className="px-2.5 py-1 rounded bg-emerald-950/80 border border-emerald-800 text-emerald-300 hover:bg-emerald-900/60 transition-colors"
            >
              Sample B (Benign)
            </button>
          </div>
        )}
      </div>

      {/* Main Grid: Inputs vs Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Features Panel */}
        <div className="lg:col-span-2 glass-panel p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <h3 className="text-base font-display font-bold text-white">
                {inputType === 'tabular' ? 'Input Diagnostic Feature Vector' : 'Multi-Channel Sensor Time-Series'}
              </h3>
            </div>
            <span className="text-xs font-mono text-slate-400">
              {inputType === 'tabular' ? 'Wisconsin Diagnostic Dataset' : '10 Timesteps × 6 Channels'}
            </span>
          </div>

          {inputType === 'tabular' ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-[380px] overflow-y-auto pr-2">
              {features.map((val, idx) => (
                <div key={idx} className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                  <div className="text-[10px] font-mono text-slate-400 truncate" title={featureLabels[idx] || `F${idx + 1}`}>
                    {featureLabels[idx] || `Feature ${idx + 1}`}
                  </div>
                  <input
                    type="number"
                    step="any"
                    value={val}
                    onChange={(e) => {
                      const updated = [...features]
                      updated[idx] = parseFloat(e.target.value) || 0
                      setFeatures(updated)
                    }}
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-cyan-300 font-mono focus:outline-none focus:border-cyan-500"
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3 font-mono text-xs">
              <div className="flex items-center gap-2 text-purple-400">
                <Layers className="w-4 h-4" />
                <span className="font-bold">Generated 10-Timestep Temporal Sensor Batch</span>
              </div>
              <p className="text-slate-400 text-xs">
                Sensors stream real-time vibrational, telemetry, and acoustic time-series input into <code className="text-purple-300">TimeSeriesTransformerNN</code>.
              </p>
              <div className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] text-purple-300 overflow-x-auto">
                {`Sequence Tensor shape: (1, 10, 6) | dtype: torch.float32`}
              </div>
            </div>
          )}
        </div>

        {/* Prediction Results & Latency Card */}
        <div className="glass-panel p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-display font-bold text-white">Inference Assessment</h3>
            <span className="font-mono text-xs text-emerald-400 flex items-center gap-1">
              <Activity className="w-3.5 h-3.5" />
              <span>Live Engine</span>
            </span>
          </div>

          {predictionResult ? (
            <div className="space-y-5 animate-fadeIn">
              {/* Classification Outcome */}
              <div
                className={`p-4 rounded-xl border text-center space-y-1.5 ${
                  predictionResult.prediction === 1
                    ? 'bg-rose-950/40 border-rose-500/50 text-rose-300'
                    : 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
                }`}
              >
                <div className="text-xs uppercase font-mono tracking-wider font-semibold">Predicted Outcome</div>
                <div className="text-2xl font-display font-bold text-white">{predictionResult.predicted_label}</div>
                <div className="text-xs font-mono">
                  Confidence: {(predictionResult.confidence_score * 100).toFixed(1)}%
                </div>
              </div>

              {/* Probabilities Distribution */}
              <div className="space-y-2 text-xs font-mono">
                <div className="text-slate-400 uppercase font-semibold">Probability Distribution</div>
                <div className="space-y-1.5">
                  <div className="flex justify-between text-slate-300">
                    <span>Benign / Normal:</span>
                    <span className="text-emerald-400 font-bold">
                      {((predictionResult.probabilities[0] || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className="h-full bg-emerald-400 rounded-full"
                      style={{ width: `${(predictionResult.probabilities[0] || 0) * 100}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-slate-300 pt-1">
                    <span>Malignant / Anomaly:</span>
                    <span className="text-rose-400 font-bold">
                      {((predictionResult.probabilities[1] || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className="h-full bg-rose-400 rounded-full"
                      style={{ width: `${(predictionResult.probabilities[1] || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Latency & DB Audit Telemetry */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2 text-xs font-mono">
                <div className="flex items-center justify-between text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Inference Latency:</span>
                  </span>
                  <span className="text-cyan-400 font-bold">{predictionResult.latency_ms} ms</span>
                </div>
                <div className="flex items-center justify-between text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-purple-400" />
                    <span>Model Version:</span>
                  </span>
                  <span className="text-purple-300 font-bold">ID #{predictionResult.model_version_id}</span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-emerald-400 text-[11px]">
                  <span className="flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Logged to DB</span>
                  </span>
                  <span className="text-slate-500">table: predictions</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center gap-3 text-slate-500 text-center p-4">
              <Zap className="w-8 h-8 text-slate-600" />
              <div className="text-xs">Select features and click <span className="text-cyan-400">"Run Live Inference"</span> to test model classification.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
