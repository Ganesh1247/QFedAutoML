import React from 'react'
import {
  LayoutDashboard,
  Network,
  Atom,
  Boxes,
  Eye,
  ShieldAlert,
  Zap
} from 'lucide-react'

export type NavTab =
  | 'overview'
  | 'federated'
  | 'automl-quantum'
  | 'model-registry'
  | 'explainability'
  | 'privacy-security'
  | 'predict-lab'

interface SidebarProps {
  activeTab: NavTab
  onSelectTab: (tab: NavTab) => void
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab }) => {
  const navItems: { id: NavTab; label: string; icon: React.ComponentType<{ className?: string }>; tag?: string }[] = [
    { id: 'overview', label: 'Executive Overview', icon: LayoutDashboard },
    { id: 'federated', label: 'Federated Studio', icon: Network, tag: 'FL Core' },
    { id: 'automl-quantum', label: 'AutoML & Quantum', icon: Atom, tag: 'QAOA' },
    { id: 'model-registry', label: 'Model Registry', icon: Boxes, tag: 'Staging' },
    { id: 'explainability', label: 'Explainability & Trust', icon: Eye, tag: 'SHAP/LIME' },
    { id: 'privacy-security', label: 'Privacy & Security', icon: ShieldAlert, tag: 'DP/Byz' },
    { id: 'predict-lab', label: 'Inference Lab', icon: Zap, tag: 'Live API' },
  ]

  return (
    <aside className="w-64 border-r border-slate-800/80 bg-slate-950/60 backdrop-blur-xl p-4 flex flex-col justify-between shrink-0">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[11px] font-mono tracking-wider text-slate-500 uppercase font-semibold">
          Platform Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.id

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-cyan-500/15 via-indigo-500/10 to-transparent text-cyan-300 border-l-4 border-cyan-400 font-semibold shadow-inner'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </div>
              {item.tag && (
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    isActive
                      ? 'bg-cyan-950 text-cyan-300 border border-cyan-800/50'
                      : 'bg-slate-900 text-slate-500'
                  }`}
                >
                  {item.tag}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Capstone Architecture Info Box */}
      <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/70 text-xs space-y-2">
        <div className="flex items-center justify-between text-slate-300 font-medium">
          <span>Non-Negotiable Rule</span>
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Neural network training strictly executes on classical CPU/GPU. Quantum QAOA solves only combinatorial QUBO subproblems.
        </p>
      </div>
    </aside>
  )
}
