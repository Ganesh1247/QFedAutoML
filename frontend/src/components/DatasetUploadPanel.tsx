import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Upload,
  Database,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Loader2,
  AlertTriangle,
  ChevronDown,
  FileSpreadsheet,
  Trash2
} from 'lucide-react'
import {
  peekCsvHeaders,
  uploadDataset,
  fetchActiveDataset,
  resetDataset,
  ActiveDatasetInfo
} from '../services/api'

interface DatasetUploadPanelProps {
  onDatasetChange?: () => void
}

export const DatasetUploadPanel: React.FC<DatasetUploadPanelProps> = ({ onDatasetChange }) => {
  const [activeDataset, setActiveDataset] = useState<ActiveDatasetInfo | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [columns, setColumns] = useState<string[]>([])
  const [targetColumn, setTargetColumn] = useState<string>('')
  const [peekLoading, setPeekLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)
  const [resetting, setResetting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Load active dataset on mount
  useEffect(() => {
    fetchActiveDataset()
      .then(setActiveDataset)
      .catch(() => setActiveDataset(null))
  }, [])

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMsg('Only .csv files are supported.')
      return
    }
    setErrorMsg(null)
    setSuccessMsg(null)
    setSelectedFile(file)
    setColumns([])
    setTargetColumn('')
    setPeekLoading(true)
    try {
      const { columns: cols } = await peekCsvHeaders(file)
      setColumns(cols)
      setTargetColumn(cols[cols.length - 1] || '')
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail || 'Failed to read CSV headers.')
    } finally {
      setPeekLoading(false)
    }
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }, [handleFileSelect])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }

  const handleUpload = async () => {
    if (!selectedFile || !targetColumn) return
    setUploading(true)
    setUploadPct(0)
    setErrorMsg(null)
    setSuccessMsg(null)
    try {
      const result = await uploadDataset(selectedFile, targetColumn, setUploadPct)
      setActiveDataset(result.dataset as ActiveDatasetInfo)
      setSuccessMsg(result.message)
      setSelectedFile(null)
      setColumns([])
      setTargetColumn('')
      if (fileInputRef.current) fileInputRef.current.value = ''
      onDatasetChange?.()
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail || 'Upload failed. Check your CSV format.')
    } finally {
      setUploading(false)
    }
  }

  const handleReset = async () => {
    setResetting(true)
    setErrorMsg(null)
    setSuccessMsg(null)
    try {
      await resetDataset()
      const info = await fetchActiveDataset()
      setActiveDataset(info)
      setSuccessMsg('Reset to built-in Wisconsin Breast Cancer dataset.')
      onDatasetChange?.()
    } catch {
      setErrorMsg('Reset failed.')
    } finally {
      setResetting(false)
    }
  }

  const isUserDataset = activeDataset?.source === 'user_upload'

  return (
    <div className="glass-panel p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-cyan-400" />
          <h3 className="text-base font-display font-bold text-white">
            Dataset Management
          </h3>
        </div>
        <span
          className={`text-[10px] font-mono px-2.5 py-0.5 rounded-full border font-semibold ${
            isUserDataset
              ? 'bg-purple-950 border-purple-700 text-purple-300'
              : 'bg-slate-800 border-slate-700 text-slate-400'
          }`}
        >
          {isUserDataset ? '🔷 Custom Dataset Active' : '🔵 Built-in Dataset'}
        </span>
      </div>

      {/* Active Dataset Info */}
      {activeDataset && (
        <div className={`p-4 rounded-xl border space-y-3 ${
          isUserDataset
            ? 'bg-purple-950/20 border-purple-700/50'
            : 'bg-slate-900/60 border-slate-800'
        }`}>
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1 min-w-0">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className={`w-3.5 h-3.5 shrink-0 ${isUserDataset ? 'text-purple-400' : 'text-cyan-400'}`} />
                <span className="text-xs font-semibold text-white truncate">
                  {activeDataset.filename}
                </span>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-slate-400">
                <span>{activeDataset.num_samples.toLocaleString()} rows</span>
                <span>{activeDataset.num_features} features</span>
                <span>{activeDataset.num_classes} classes</span>
                <span>Target: <code className="text-cyan-400">{activeDataset.target_column}</code></span>
              </div>
            </div>
            {isUserDataset && (
              <button
                onClick={handleReset}
                disabled={resetting}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-slate-400 hover:text-rose-300 hover:border-rose-700 text-xs font-mono transition-all shrink-0 cursor-pointer disabled:opacity-50"
              >
                {resetting
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <RotateCcw className="w-3.5 h-3.5" />
                }
                <span>Reset to Default</span>
              </button>
            )}
          </div>

          {/* Feature pills */}
          <div className="flex flex-wrap gap-1.5">
            {activeDataset.feature_columns.slice(0, 10).map((f) => (
              <span key={f} className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-400">
                {f}
              </span>
            ))}
            {activeDataset.feature_columns.length > 10 && (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-500">
                +{activeDataset.feature_columns.length - 10} more
              </span>
            )}
          </div>

          {/* Class distribution */}
          <div className="flex gap-2 flex-wrap">
            {Object.entries(activeDataset.class_distribution).map(([cls, count]) => (
              <div key={cls} className="flex items-center gap-1.5 text-[11px] font-mono">
                <span className="w-2 h-2 rounded-full bg-cyan-400 opacity-70"></span>
                <span className="text-slate-400">Class {cls}:</span>
                <span className="text-white font-semibold">{count}</span>
              </div>
            ))}
          </div>

          {/* Dropped columns warning */}
          {activeDataset.dropped_non_numeric && activeDataset.dropped_non_numeric.length > 0 && (
            <div className="flex items-center gap-2 text-[11px] font-mono text-amber-400">
              <AlertTriangle className="w-3 h-3 shrink-0" />
              <span>Non-numeric columns auto-dropped: {activeDataset.dropped_non_numeric.join(', ')}</span>
            </div>
          )}
        </div>
      )}

      {/* Upload Zone */}
      <div className="space-y-3">
        <p className="text-xs text-slate-400">
          Upload your own <span className="text-cyan-400 font-mono">.csv</span> dataset — it will replace the active dataset across <span className="text-white">all platform modules</span> (AutoML, SHAP/LIME, Predict Lab, Federated Training).
        </p>

        {/* Drag & Drop Zone */}
        <div
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => fileInputRef.current?.click()}
          className={`relative flex flex-col items-center justify-center gap-3 p-6 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
            dragOver
              ? 'border-cyan-400 bg-cyan-500/10 scale-[1.01]'
              : selectedFile
              ? 'border-emerald-600 bg-emerald-950/20'
              : 'border-slate-700 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={onInputChange}
          />
          {selectedFile ? (
            <>
              <CheckCircle2 className="w-8 h-8 text-emerald-400" />
              <div className="text-center">
                <p className="text-sm font-semibold text-white">{selectedFile.name}</p>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  {(selectedFile.size / 1024).toFixed(1)} KB — click to change
                </p>
              </div>
            </>
          ) : (
            <>
              <Upload className={`w-8 h-8 transition-colors ${dragOver ? 'text-cyan-400' : 'text-slate-600'}`} />
              <div className="text-center">
                <p className="text-sm font-semibold text-slate-300">
                  {dragOver ? 'Drop your CSV here' : 'Drag & drop your CSV file'}
                </p>
                <p className="text-xs text-slate-500 mt-0.5">or click to browse — max 50 MB</p>
              </div>
            </>
          )}
        </div>

        {/* Column Headers Peek Loading */}
        {peekLoading && (
          <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
            <span>Reading CSV headers...</span>
          </div>
        )}

        {/* Target Column Selector */}
        {columns.length > 0 && !peekLoading && (
          <div className="space-y-2">
            <label className="text-xs font-mono text-slate-400 uppercase font-semibold">
              Select Target / Label Column
            </label>
            <div className="relative">
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full appearance-none bg-slate-900 border border-slate-700 rounded-lg px-3 py-2.5 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono pr-8"
              >
                {columns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
            </div>
            <p className="text-[11px] text-slate-500 font-mono">
              All other columns will be used as features. Non-numeric columns are auto-dropped.
            </p>

            {/* Upload Progress */}
            {uploading && (
              <div className="space-y-1.5">
                <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
                    style={{ width: `${uploadPct}%` }}
                  />
                </div>
                <p className="text-[11px] font-mono text-cyan-400">{uploadPct}% Uploading & Validating...</p>
              </div>
            )}

            {/* Upload Button */}
            <button
              onClick={handleUpload}
              disabled={uploading || !targetColumn}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50 cursor-pointer"
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Activating Dataset...</span>
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  <span>Upload & Activate Dataset</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Status Messages */}
        {successMsg && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-950/40 border border-emerald-700/50 text-emerald-300 text-xs font-mono">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{successMsg}</span>
          </div>
        )}
        {errorMsg && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-950/40 border border-rose-700/50 text-rose-300 text-xs font-mono">
            <XCircle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>
    </div>
  )
}
