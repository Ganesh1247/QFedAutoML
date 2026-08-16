import React from 'react'
import { Cpu, ShieldCheck, Activity, Database, Sparkles } from 'lucide-react'

interface NavbarProps {
  systemStatus: {
    status?: string
    database?: string
    version?: string
    quantum_backend?: string
  }
}

export const Navbar: React.FC<NavbarProps> = ({ systemStatus }) => {
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

      {/* Live Hardware & Platform Telemetry Indicators */}
      <div className="flex items-center gap-4">
        {/* Classical PyTorch Indicator */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span>PyTorch Classical FL</span>
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        </div>

        {/* Quantum Simulator Aer Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-950/40 border border-purple-800/50 text-xs text-purple-200">
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span className="font-mono">{systemStatus.quantum_backend || 'Qiskit Aer QAOA'}</span>
          <span className="px-1.5 py-0.2 text-[9px] font-mono bg-purple-900/60 rounded text-purple-300 font-bold">≤16 Qubits</span>
        </div>

        {/* Privacy & DP Guard */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-xs text-emerald-300">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>DP-SGD Guard Active</span>
        </div>

        {/* Database Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          <span className="font-mono">SQLite / PG</span>
          <Activity className="w-3 h-3 text-emerald-400" />
        </div>
      </div>
    </header>
  )
}
