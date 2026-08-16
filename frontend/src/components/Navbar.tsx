import React from 'react'
import { Cpu, ShieldCheck, Activity, Database, Sparkles, FileSpreadsheet, HelpCircle } from 'lucide-react'
import { ActiveDatasetInfo } from '../services/api'

interface NavbarProps {
  systemStatus: {
    status?: string
    database?: string
    version?: string
    quantum_backend?: string
  }
  activeDataset?: ActiveDatasetInfo | null
  onOpenGuide?: () => void
}

export const Navbar: React.FC<NavbarProps> = ({ systemStatus, activeDataset, onOpenGuide }) => {
  const isCustomDataset = activeDataset?.source === 'user_upload'
  const datasetName = activeDataset?.filename || 'Wisconsin Breast Cancer'
  const targetCol = activeDataset?.target_column || 'diagnosis'

  return (
    <header className="h-16 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Brand Title */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 p-[1px] shadow-lg shadow-cyan-500/20">
          <div className="w-full h-full bg-slate-950 rounded-xl flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-cyan-400 animate-pulse-slow" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-display font-bold text-lg text-white tracking-tight">
              QFed<span className="bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-transparent">AutoML</span>
            </h1>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 font-semibold">
              v{systemStatus.version || '0.1.0'} Capstone
            </span>
          </div>
          <p className="text-xs text-slate-400">Quantum-Enhanced Federated AutoML Platform</p>
        </div>
      </div>

      {/* Center / Right: Active Dataset Banner & Guide */}
      <div className="flex items-center gap-3">
        {/* Active Dataset Pill */}
        <div className={`hidden md:flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-mono transition-all ${
          isCustomDataset
            ? 'bg-purple-950/50 border-purple-500/40 text-purple-200 shadow-sm shadow-purple-500/10'
            : 'bg-slate-900/80 border-slate-800 text-slate-300'
        }`}>
          <FileSpreadsheet className={`w-3.5 h-3.5 ${isCustomDataset ? 'text-purple-400' : 'text-cyan-400'}`} />
          <span className="font-semibold text-white truncate max-w-[150px]" title={datasetName}>{datasetName}</span>
          <span className="text-slate-500">|</span>
          <span className="text-[11px] text-cyan-300">Target: <strong className="text-white">{targetCol}</strong></span>
        </div>

        {/* Beginner Guide Button */}
        <button
          onClick={onOpenGuide}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-950/80 hover:bg-cyan-900/80 border border-cyan-500/30 text-cyan-300 hover:text-cyan-200 text-xs font-medium transition-all cursor-pointer shadow-sm shadow-cyan-500/10"
        >
          <HelpCircle className="w-3.5 h-3.5 text-cyan-400" />
          <span className="hidden sm:inline">How to Use</span>
        </button>

        {/* Quantum Simulator Aer Status */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-950/40 border border-purple-800/50 text-xs text-purple-200">
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span className="font-mono">{systemStatus.quantum_backend || 'Qiskit Aer QAOA'}</span>
          <span className="px-1.5 py-0.2 text-[9px] font-mono bg-purple-900/60 rounded text-purple-300 font-bold">≤16 Qubits</span>
        </div>

        {/* Database Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          <span className="font-mono">SQLite</span>
          <Activity className="w-3 h-3 text-emerald-400" />
        </div>
      </div>
    </header>
  )
}
