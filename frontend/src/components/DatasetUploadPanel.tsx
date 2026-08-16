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
      const res = await peekCsvHeaders(file)
      const cols = Array.isArray(res?.columns) ? res.columns : []
      setColumns(cols)
      setTargetColumn(cols.length > 0 ? cols[cols.length - 1] : '')
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to read CSV headers.')
    } finally {
      setPeekLoading(false)
    }
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
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
      if (result && result.dataset) {
        setActiveDataset(result.dataset as ActiveDatasetInfo)
        setSuccessMsg(result.message || 'Dataset activated successfully!')
        setSelectedFile(null)
        setColumns([])
        setTargetColumn('')
        if (fileInputRef.current) fileInputRef.current.value = ''
        onDatasetChange?.()
      } else {
        throw new Error('Invalid dataset response')
      }
    } catch (err: any) {
      setErrorMsg(err?.message || 'Upload failed. Check your CSV format.')
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
          <Database className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-base font-display font-bold text-white">
              Dataset Management & Active Data Status
            </h3>
            <p className="text-xs text-slate-400">
              The dataset active below is used across all training, AutoML, and prediction modules
            </p>
          </div>
        </div>
        <span
          className={`text-[11px] font-mono px-3 py-1 rounded-full border font-bold ${
            isUserDataset
              ? 'bg-purple-950 border-purple-500 text-purple-300 shadow-sm shadow-purple-500/20'
              : 'bg-slate-800 border-slate-700 text-slate-300'
          }`}
        >
          {isUserDataset ? '🔷 Active Custom Dataset' : '🔵 Built-in Sample Dataset'}
        </span>
      </div>

      {/* Active Dataset Clear Profile Card */}
      {activeDataset ? (
        <div className={`p-5 rounded-2xl border space-y-4 ${
          isUserDataset
            ? 'bg-gradient-to-r from-purple-950/40 via-slate-900 to-slate-900 border-purple-600/60 shadow-lg shadow-purple-950/40'
            : 'bg-slate-900/80 border-slate-800'
        }`}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className={`p-3 rounded-xl ${isUserDataset ? 'bg-purple-500/20 text-purple-300' : 'bg-cyan-500/20 text-cyan-300'}`}>
                <FileSpreadsheet className="w-6 h-6" />
              </div>
              <div className="min-w-0">
                <div className="text-sm font-bold text-white truncate flex items-center gap-2">
                  <span>{activeDataset.filename}</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-cyan-300">
                    Active in Memory
                  </span>
                </div>
                <div className="text-xs text-slate-300 mt-0.5">
                  Goal: Predicting <strong className="text-cyan-300 font-mono">{activeDataset.target_column || 'target'}</strong> based on {activeDataset.num_features ?? 0} factors
                </div>
              </div>
            </div>

            {isUserDataset && (
              <button
                onClick={handleReset}
                disabled={resetting}
                className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:text-rose-300 hover:border-rose-700 text-xs font-semibold transition-all shrink-0 cursor-pointer disabled:opacity-50"
              >
                {resetting
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  : <RotateCcw className="w-3.5 h-3.5" />
                }
                <span>Switch to Built-in Sample</span>
              </button>
            )}
          </div>

          {/* Dataset Statistics Matrix */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase">Total Rows</span>
              <div className="text-base font-bold text-white mt-0.5">{(activeDataset.num_samples ?? 0).toLocaleString()}</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase">Input Factors</span>
              <div className="text-base font-bold text-cyan-300 mt-0.5">{activeDataset.num_features ?? 0} Columns</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase">Predicted Target</span>
              <div className="text-xs font-bold text-purple-300 mt-1 truncate" title={activeDataset.target_column || 'target'}>
                {activeDataset.target_column || 'target'}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase">Partition Status</span>
              <div className="text-xs font-bold text-emerald-400 mt-1 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                <span>3 Edge Nodes</span>
              </div>
            </div>
          </div>

          {/* Feature columns badges */}
          {activeDataset.feature_columns && activeDataset.feature_columns.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <div className="text-[11px] font-mono text-slate-400 font-semibold uppercase">
                Active Factors (Input Features):
              </div>
              <div className="flex flex-wrap gap-1.5">
                {activeDataset.feature_columns.map((f) => (
                  <span key={f} className="px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-950/90 border border-slate-800 text-slate-300">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Dropped columns warning if any */}
          {activeDataset.dropped_non_numeric && activeDataset.dropped_non_numeric.length > 0 && (
            <div className="flex items-center gap-2 p-2.5 rounded-lg bg-amber-950/30 border border-amber-800/40 text-xs font-mono text-amber-300">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>Note: Non-numeric columns (e.g. text/date strings) were excluded: {activeDataset.dropped_non_numeric.join(', ')}</span>
            </div>
          )}
        </div>
      ) : null}

      {/* Upload Zone */}
      <div className="space-y-3 pt-2">
        <div className="text-xs text-slate-300 font-semibold flex items-center gap-2">
          <Upload className="w-4 h-4 text-cyan-400" />
          <span>Upload a New Dataset (.csv):</span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Upload any CSV spreadsheet (such as housing data, sales figures, or medical metrics). It will immediately update all modules across the platform.
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
