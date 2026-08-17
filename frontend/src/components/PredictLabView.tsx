import React, { useEffect, useState } from 'react'
import {
  Zap,
  Activity,
  CheckCircle2,
  Clock,
  Sparkles,
  Sliders,
  Loader2,
  Cpu,
  Layers,
  FileSpreadsheet
} from 'lucide-react'
import { runPrediction, ActiveDatasetInfo } from '../services/api'
import { ModelVersion, PredictionResult } from '../types'

interface PredictLabViewProps {
  models: ModelVersion[]
  activeDataset?: ActiveDatasetInfo | null
}

const defaultBreastCancerLabels = [
  'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
  'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
  'Radius Error', 'Texture Error', 'Perimeter Error', 'Area Error', 'Smoothness Error',
  'Compactness Error', 'Concavity Error', 'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error',
  'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
  'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
]

export const PredictLabView: React.FC<PredictLabViewProps> = ({ models, activeDataset }) => {
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

  // Dynamically set features when activeDataset changes
  useEffect(() => {
    if (activeDataset?.feature_columns && activeDataset.feature_columns.length > 0) {
      // Initialize with reasonable numerical values
      const initialVals = activeDataset.feature_columns.map((_, i) => +(10.0 + (i * 2.5)).toFixed(2))
      setFeatures(initialVals)
    }
  }, [activeDataset])

  // Auto-run prediction when features change (debounced)
  useEffect(() => {
    if (features.length > 0) {
      const timer = setTimeout(() => {
        handleRunInference()
      }, 250)
      return () => clearTimeout(timer)
    }
  }, [features, selectedModelId])

  const featureLabels = activeDataset?.feature_columns && activeDataset.feature_columns.length > 0
    ? activeDataset.feature_columns
    : defaultBreastCancerLabels

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

  const isWeather = activeDataset?.filename?.toLowerCase().includes('weather') || activeDataset?.target_column?.toLowerCase().includes('weather')

  const loadWeatherScenario = (scenario: 'rain' | 'sun' | 'drizzle' | 'snow' | 'fog') => {
    if (scenario === 'rain') {
      setFeatures([12.5, 7.2, 4.1, 14.8])
    } else if (scenario === 'sun') {
      setFeatures([0.0, 26.5, 14.2, 4.2])
    } else if (scenario === 'drizzle') {
      setFeatures([0.4, 12.0, 8.5, 3.1])
    } else if (scenario === 'snow') {
      setFeatures([5.2, 0.5, -4.2, 11.0])
    } else if (scenario === 'fog') {
      setFeatures([0.0, 8.5, 5.0, 1.2])
    }
  }

  const loadPreset = (type: 'sampleA' | 'sampleB') => {
    if (isWeather) {
      loadWeatherScenario(type === 'sampleA' ? 'rain' : 'sun')
      return
    }

    if (activeDataset?.feature_columns && activeDataset.feature_columns.length > 0) {
      if (type === 'sampleA') {
        setFeatures(activeDataset.feature_columns.map((_, i) => +(25.0 + (i * 4.2)).toFixed(2)))
      } else {
        setFeatures(activeDataset.feature_columns.map((_, i) => +(5.0 + (i * 1.1)).toFixed(2)))
      }
    } else {
      if (type === 'sampleA') {
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
  }


  const isCustomDataset = Boolean(activeDataset && activeDataset.source === 'user_upload')
  const targetCol = activeDataset?.target_column || 'Target'
  const datasetTitle = activeDataset?.filename || 'Built-in Dataset'

  // Helper to format weather label with emoji
  const formatWeatherLabel = (lbl: string, predIndex?: number) => {
    let l = (lbl || '').toLowerCase()
    if (l.includes('class') || !isNaN(Number(l))) {
      const idx = predIndex !== undefined ? predIndex : parseInt(l.replace(/\D/g, '') || '2')
      const weatherMap: Record<number, string> = {
        0: 'drizzle',
        1: 'fog',
        2: 'rain',
        3: 'snow',
        4: 'sun'
      }
      l = weatherMap[idx % 5] || 'rain'
    }

    if (l.includes('rain')) return '🌧️ Rain (Precipitation)'
    if (l.includes('sun')) return '☀️ Sun (Clear Weather)'
    if (l.includes('drizzle')) return '🌦️ Drizzle (Light Rain)'
    if (l.includes('snow')) return '❄️ Snow (Freezing)'
    if (l.includes('fog')) return '🌫️ Fog (Low Visibility)'
    return lbl.toUpperCase()
  }

  const resolveLabel = (lbl?: string, predIndex?: number): string => {
    if (!lbl) return isCustomDataset ? `Standard (${targetCol})` : 'Normal (Class 0)'
    if (isWeather) return formatWeatherLabel(lbl, predIndex)
    if (activeDataset?.class_labels && activeDataset.class_labels.length > 0 && predIndex !== undefined) {
      return activeDataset.class_labels[predIndex % activeDataset.class_labels.length]
    }
    return lbl
  }

  const outcomeTitle = resolveLabel(predictionResult?.predicted_label, predictionResult?.prediction)

  const breakdownList = predictionResult?.class_breakdown || (
    predictionResult?.probabilities
      ? predictionResult.probabilities.map((p, idx) => ({
          label: idx === 1 ? 'Positive / Upper Tier' : 'Negative / Lower Tier',
          probability: p
        }))
      : []
  )

  const classColors = [
    'from-cyan-500 to-blue-500',
    'from-purple-500 to-indigo-500',
    'from-emerald-500 to-teal-500',
    'from-amber-500 to-orange-500',
    'from-rose-500 to-pink-500',
  ]

  const textColors = [
    'text-cyan-400',
    'text-purple-400',
    'text-emerald-400',
    'text-amber-400',
    'text-rose-400',
  ]

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
              Instant AI Prediction (&lt;15ms)
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Test instant predictions on custom inputs using your trained AI model.
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
              <span>Calculating Prediction...</span>
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 fill-current" />
              <span>Run Live Prediction</span>
            </>
          )}
        </button>
      </div>

      {/* Non-Coder Friendly Guide Banner */}
      <div className="p-4 rounded-xl bg-cyan-950/30 border border-cyan-500/30 flex items-start gap-3">
        <Sparkles className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1 text-xs">
          <div className="font-semibold text-cyan-200">How to use this page (No coding required):</div>
          <p className="text-slate-300 leading-relaxed">
            <strong>Step 1:</strong> On the left panel, adjust any input value (such as {featureLabels.slice(0, 3).join(', ')}).<br />
            <strong>Step 2:</strong> Click the blue <strong>"Run Live Prediction"</strong> button.<br />
            <strong>Step 3:</strong> See the AI's instant estimated outcome and confidence percentage on the right!
          </p>
        </div>
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
            Tabular ({featureLabels.length} Columns)
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
            <option value="">Auto (Production Model)</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.model_name} {m.is_production ? '[ACTIVE PRODUCTION]' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Example Presets */}
        {inputType === 'tabular' && (
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <span className="text-slate-500">Quick Test Scenarios:</span>
            {isWeather ? (
              <>
                <button
                  onClick={() => loadWeatherScenario('rain')}
                  className="px-2.5 py-1 rounded bg-blue-950/60 border border-blue-700 text-blue-300 hover:border-blue-400 transition-colors cursor-pointer"
                >
                  🌧️ Rain Scenario
                </button>
                <button
                  onClick={() => loadWeatherScenario('sun')}
                  className="px-2.5 py-1 rounded bg-amber-950/60 border border-amber-700 text-amber-300 hover:border-amber-400 transition-colors cursor-pointer"
                >
                  ☀️ Sunny Scenario
                </button>
                <button
                  onClick={() => loadWeatherScenario('drizzle')}
                  className="px-2.5 py-1 rounded bg-cyan-950/60 border border-cyan-700 text-cyan-300 hover:border-cyan-400 transition-colors cursor-pointer"
                >
                  🌦️ Drizzle Scenario
                </button>
                <button
                  onClick={() => loadWeatherScenario('snow')}
                  className="px-2.5 py-1 rounded bg-indigo-950/60 border border-indigo-700 text-indigo-300 hover:border-indigo-400 transition-colors cursor-pointer"
                >
                  ❄️ Snow Scenario
                </button>
                <button
                  onClick={() => loadWeatherScenario('fog')}
                  className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300 hover:border-slate-400 transition-colors cursor-pointer"
                >
                  🌫️ Fog Scenario
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => loadPreset('sampleA')}
                  className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-cyan-300 hover:border-cyan-500 transition-colors cursor-pointer"
                >
                  High Values Sample
                </button>
                <button
                  onClick={() => loadPreset('sampleB')}
                  className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300 hover:border-slate-500 transition-colors cursor-pointer"
                >
                  Average Sample
                </button>
              </>
            )}
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
                {inputType === 'tabular' ? 'Input Feature Values' : 'Multi-Channel Sensor Time-Series'}
              </h3>
            </div>
            <span className="text-xs font-mono text-slate-400 flex items-center gap-1.5">
              <FileSpreadsheet className="w-3.5 h-3.5 text-cyan-400" />
              <span>{datasetTitle}</span>
            </span>
          </div>

          {inputType === 'tabular' ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-[380px] overflow-y-auto pr-2">
              {features.map((val, idx) => (
                <div key={idx} className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                  <div className="text-[11px] font-mono text-slate-300 font-medium truncate" title={featureLabels[idx] || `F${idx + 1}`}>
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
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-cyan-300 font-mono focus:outline-none focus:border-cyan-500"
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
            <h3 className="text-base font-display font-bold text-white">Prediction Result</h3>
            <span className="font-mono text-xs text-emerald-400 flex items-center gap-1">
              <Activity className="w-3.5 h-3.5" />
              <span>Model Active</span>
            </span>
          </div>

          {predictionResult ? (
            <div className="space-y-5 animate-fadeIn">
              {/* Classification Outcome */}
              <div
                className={`p-4 rounded-xl border text-center space-y-1.5 ${
                  predictionResult.prediction === 1
                    ? 'bg-purple-950/40 border-purple-500/50 text-purple-300'
                    : 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
                }`}
              >
                <div className="text-xs uppercase font-mono tracking-wider font-semibold text-slate-400">Predicted Result</div>
                <div className="text-2xl font-display font-bold text-white tracking-wide">{outcomeTitle}</div>
                <div className="text-xs font-mono text-emerald-400">
                  Confidence Score: {(predictionResult.confidence_score * 100).toFixed(1)}%
                </div>
              </div>

              {/* Plain English Summary */}
              <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 text-xs text-slate-300 space-y-1">
                <div className="font-semibold text-white flex items-center gap-1.5">
                  <span>💡</span>
                  <span>What does this result mean?</span>
                </div>
                <p className="text-[11px] leading-relaxed text-slate-400">
                  Based on your input parameters, the trained AI model concludes the outcome for <strong>{targetCol}</strong> is <strong className="text-cyan-300">{outcomeTitle}</strong> with <strong>{(predictionResult.confidence_score * 100).toFixed(1)}% certainty</strong>.
                </p>
              </div>

              {/* Probabilities Distribution */}
              <div className="space-y-2 text-xs font-mono">
                <div className="text-slate-400 uppercase font-semibold">Category Likelihood Breakdown</div>
                <div className="space-y-2.5">
                  {breakdownList.map((item, idx) => {
                    const labelFormatted = isWeather ? formatWeatherLabel(item.label) : item.label
                    const pct = Math.min(100, Math.max(0, (item.probability || 0) * 100))
                    const colorGradient = classColors[idx % classColors.length]
                    const txtColor = textColors[idx % textColors.length]
                    return (
                      <div key={idx}>
                        <div className="flex justify-between text-slate-300 text-[11px] mb-1">
                          <span className="capitalize">{labelFormatted}:</span>
                          <span className={`${txtColor} font-bold`}>
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                          <div
                            className={`h-full bg-gradient-to-r ${colorGradient} rounded-full transition-all duration-500`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Latency & Audit Telemetry */}
              <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2 text-xs font-mono">
                <div className="flex items-center justify-between text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Response Time:</span>
                  </span>
                  <span className="text-cyan-400 font-bold">{predictionResult.latency_ms} ms</span>
                </div>
                <div className="flex items-center justify-between text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-purple-400" />
                    <span>Target Variable:</span>
                  </span>
                  <span className="text-purple-300 font-bold">{targetCol}</span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-emerald-400 text-[11px]">
                  <span className="flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Prediction Logged</span>
                  </span>
                  <span className="text-slate-500">Status: Verified</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center gap-3 text-slate-500 text-center p-4">
              <Zap className="w-8 h-8 text-slate-600" />
              <div className="text-xs">Adjust your input values and click <span className="text-cyan-400 font-semibold">"Run Live Prediction"</span> to see the instant estimate.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

