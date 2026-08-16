import React, { useState } from 'react'
import {
  ShieldAlert,
  Lock,
  Sliders,
  ShieldCheck,
  AlertTriangle,
  FileText
} from 'lucide-react'
import { EdgeClient } from '../types'

interface PrivacySecurityViewProps {
  clients: EdgeClient[]
}

export const PrivacySecurityView: React.FC<PrivacySecurityViewProps> = ({ clients }) => {
  const [clippingC, setClippingC] = useState<number>(1.0)
  const [noiseSigma, setNoiseSigma] = useState<number>(0.01)
  const [maxEpsilon, setMaxEpsilon] = useState<number>(5.0)

  const securityLogs = [
    {
      id: 101,
      type: 'BYZANTINE_ANOMALY_BLOCKED',
      severity: 'high',
      clientId: 'node_untrusted_99',
      details: 'Gradient L2 norm 4.82 exceeds 3.5x median norm threshold (1.12). Model poisoning attempt dropped.',
      timestamp: '2 mins ago',
    },
    {
      id: 102,
      type: 'DP_NOISE_INJECTED',
      severity: 'low',
      clientId: 'node_alpha_hosp',
      details: 'Client weights clipped to L2 norm 1.0; Gaussian noise calibrated to σ=0.01 added.',
      timestamp: '15 mins ago',
    },
    {
      id: 103,
      type: 'PRIVACY_ACCOUNTANT_STEP',
      severity: 'low',
      clientId: 'all_nodes',
      details: 'Moments Accountant recorded round step: cumulative ε=1.42 (budget: 5.0, δ=1e-5).',
      timestamp: '35 mins ago',
    },
    {
      id: 104,
      type: 'COSINE_DIVERGENCE_FLAGGED',
      severity: 'medium',
      clientId: 'node_test_lab',
      details: 'Cosine alignment cos(θ)=-0.14 against server median. Filtered from global FedAvg weight update.',
      timestamp: '1 hour ago',
    },
  ]

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-display font-bold text-white tracking-tight">
              Privacy & Security Center
            </h2>
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs font-mono">
              DP-SGD + Moments Accountant + Byzantine Filter
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Enforce formal Differential Privacy guarantees and defend federated aggregation against adversarial model poisoning
          </p>
        </div>
      </div>

      {/* Top Privacy Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Privacy Budget Gauge */}
        <div className="glass-panel p-6 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400 uppercase font-semibold">Cumulative Privacy (ε, δ)</span>
            <Lock className="w-4 h-4 text-emerald-400" />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between items-baseline">
              <span className="text-3xl font-display font-bold text-white">ε = 1.42</span>
              <span className="text-xs font-mono text-slate-400">Max Budget: {maxEpsilon}.0</span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full transition-all"
                style={{ width: `${(1.42 / maxEpsilon) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] font-mono text-slate-400 pt-1">
              <span>{((1.42 / maxEpsilon) * 100).toFixed(1)}% Spent</span>
              <span className="text-emerald-400">δ = 1.0e-5 (Preserved)</span>
            </div>
          </div>
        </div>

        {/* DP Mechanism Settings */}
        <div className="glass-panel p-6 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400 uppercase font-semibold">DP Gradient Clipping</span>
            <Sliders className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-slate-400">L2 Clip Bound (C):</span>
              <span className="text-white font-bold">{clippingC.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={5.0}
              step={0.1}
              value={clippingC}
              onChange={(e) => setClippingC(Number(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-900 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-slate-400 pt-1">
              <span>Noise Multiplier (σ):</span>
              <span className="text-white font-bold">{noiseSigma}</span>
            </div>
          </div>
        </div>

        {/* Byzantine Threat Status */}
        <div className="glass-panel p-6 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400 uppercase font-semibold">Byzantine Defense Shield</span>
            <ShieldCheck className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex items-center gap-2 text-emerald-400 font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span>Filter Mode: Norm + Cosine Cos(θ) &lt; -0.1</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Updates deviating in directional cosine angle or exceeding 3.5x median norm are automatically discarded before FedAvg aggregation.
            </p>
          </div>
        </div>
      </div>

      {/* Security Events Audit Log */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            <h3 className="text-base font-display font-bold text-white">Security Events & Audit Log (Database Synced)</h3>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Table: <code className="text-cyan-400">security_events</code>
          </span>
        </div>

        <div className="space-y-3">
          {securityLogs.map((log) => (
            <div
              key={log.id}
              className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex items-start justify-between gap-4"
            >
              <div className="flex items-start gap-3">
                {log.severity === 'high' ? (
                  <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 shrink-0">
                    <AlertTriangle className="w-4 h-4" />
                  </div>
                ) : (
                  <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 shrink-0">
                    <FileText className="w-4 h-4" />
                  </div>
                )}
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-xs text-white">{log.type}</span>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.2 rounded uppercase font-semibold ${
                        log.severity === 'high'
                          ? 'bg-rose-950 text-rose-300 border border-rose-800'
                          : log.severity === 'medium'
                          ? 'bg-amber-950 text-amber-300 border border-amber-800'
                          : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                      }`}
                    >
                      {log.severity}
                    </span>
                    <span className="text-xs font-mono text-slate-500">[{log.clientId}]</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{log.details}</p>
                </div>
              </div>
              <span className="text-[11px] font-mono text-slate-500 whitespace-nowrap">{log.timestamp}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
