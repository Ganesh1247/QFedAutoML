import React from 'react'
import {
  X,
  Sparkles,
  Database,
  Users,
  Atom,
  Sliders,
  Eye,
  Shield,
  Zap,
  CheckCircle2,
  HelpCircle
} from 'lucide-react'

interface SimpleGuideModalProps {
  isOpen: boolean
  onClose: () => void
}

export const SimpleGuideModal: React.FC<SimpleGuideModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full max-h-[85vh] overflow-y-auto shadow-2xl flex flex-col">
        {/* Modal Header */}
        <div className="p-6 border-b border-slate-800 flex items-center justify-between sticky top-0 bg-slate-900/95 backdrop-blur z-10">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <HelpCircle className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-display font-bold text-white">
                Beginner's Guide to QFedAutoML
              </h2>
              <p className="text-xs text-slate-400">
                Understand what every section does without any programming knowledge
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 text-xs text-slate-300">
          {/* Main Idea */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-950/40 via-purple-950/30 to-slate-900 border border-cyan-500/30 space-y-2">
            <div className="font-semibold text-cyan-200 text-sm flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>What is this platform in simple words?</span>
            </div>
            <p className="leading-relaxed text-slate-300">
              <strong>QFedAutoML</strong> is an automated Artificial Intelligence platform. It allows multiple organizations (like hospitals or companies) to <strong>train AI models together without sharing their private files</strong>, uses <strong>Quantum algorithms</strong> to find the best data patterns in seconds, and gives you <strong>instant predictions with clear visual explanations</strong>.
            </p>
          </div>

          {/* 6 Core Modules Explained */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
              The 6 Pages Explained:
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {/* 1. Executive Overview */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-cyan-300 flex items-center gap-2">
                  <Database className="w-4 h-4 text-cyan-400" />
                  <span>1. Executive Overview</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Your mission control. Upload any spreadsheet (CSV) here, see which dataset is active, check connected edge computers, and view overall accuracy scores.
                </p>
              </div>

              {/* 2. Federated Studio */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-blue-300 flex items-center gap-2">
                  <Users className="w-4 h-4 text-blue-400" />
                  <span>2. Federated Studio</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Decentralized team training. Edge computers train on their own local data and send only learned mathematical updates to make a shared master model without leaking raw data.
                </p>
              </div>

              {/* 3. AutoML & Quantum */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-purple-300 flex items-center gap-2">
                  <Atom className="w-4 h-4 text-purple-400" />
                  <span>3. AutoML & Quantum</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Automated AI search. Quantum QAOA formulas find the most important columns in seconds and automatically test multiple algorithms to rank them on a Leaderboard.
                </p>
              </div>

              {/* 4. Model Registry */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-amber-300 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-amber-400" />
                  <span>4. Model Registry</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Model warehouse. View all saved versions of your trained models and click "Promote to Production" to activate the winning AI for live predictions.
                </p>
              </div>

              {/* 5. Explainability & Trust */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-emerald-300 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-emerald-400" />
                  <span>5. Explainability & Trust</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Transparent AI audit. Visual charts (SHAP & LIME) show you exactly WHY the model made its decision, ranking which columns increased or decreased the estimate.
                </p>
              </div>

              {/* 6. Inference Lab */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1.5">
                <div className="font-semibold text-cyan-300 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-cyan-400" />
                  <span>6. Real-Time Inference Lab</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Live prediction tester. Type in test numbers (e.g. 3 bedrooms, 2,500 sqft) and click "Run Live Prediction" to get an instant outcome in &lt;15 milliseconds.
                </p>
              </div>
            </div>
          </div>

          {/* Privacy & Security */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
            <div className="font-semibold text-slate-200 flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" />
              <span>Built-in Privacy & Security Shields:</span>
            </div>
            <ul className="space-y-1.5 text-slate-400 list-disc list-inside leading-relaxed">
              <li><strong className="text-slate-300">Differential Privacy:</strong> Blurs data mathematically so individual records can never be reconstructed or stolen.</li>
              <li><strong className="text-slate-300">Byzantine Filter:</strong> Automatically detects and discards corrupted or fake updates from malicious computers.</li>
            </ul>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/90 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-all cursor-pointer"
          >
            Got It! Close Guide
          </button>
        </div>
      </div>
    </div>
  )
}
