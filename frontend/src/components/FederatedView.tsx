import React, { useState } from 'react'
import {
  Play,
  CheckCircle,
  Network,
  Activity,
  Layers,
  ArrowDownUp,
  ShieldAlert,
  Loader2
} from 'lucide-react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line, Bar } from 'react-chartjs-2'
import { EdgeClient } from '../types'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface FederatedViewProps {
  clients: EdgeClient[]
}

export const FederatedView: React.FC<FederatedViewProps> = ({ clients }) => {
  const [numRounds, setNumRounds] = useState<number>(5)
  const [modelType, setModelType] = useState<string>('xgboost')
  const [clientSelector, setClientSelector] = useState<string>('quantum_qaoa')
  const [isTraining, setIsTraining] = useState<boolean>(false)
  const [currentRound, setCurrentRound] = useState<number>(3)
  const [roundsHistory, setRoundsHistory] = useState([
    { round: 1, loss: 0.624, accuracy: 0.884, roc_auc: 0.912, uplinkMB: 1.24, downlinkMB: 2.15 },
    { round: 2, loss: 0.412, accuracy: 0.926, roc_auc: 0.954, uplinkMB: 1.25, downlinkMB: 2.14 },
    { round: 3, loss: 0.285, accuracy: 0.958, roc_auc: 0.982, uplinkMB: 1.23, downlinkMB: 2.16 },
  ])

  const handleStartTraining = async () => {
    setIsTraining(true)
    // Simulate interactive rounds progression
    for (let r = currentRound + 1; r <= currentRound + 2; r++) {
      await new Promise((res) => setTimeout(res, 800))
      setRoundsHistory((prev) => [
        ...prev,
        {
          round: r,
          loss: Math.max(0.12, +(prev[prev.length - 1].loss * 0.82).toFixed(3)),
          accuracy: Math.min(0.985, +(prev[prev.length - 1].accuracy + 0.012).toFixed(3)),
          roc_auc: Math.min(0.995, +(prev[prev.length - 1].roc_auc + 0.008).toFixed(3)),
          uplinkMB: +(1.22 + Math.random() * 0.05).toFixed(2),
          downlinkMB: +(2.14 + Math.random() * 0.05).toFixed(2),
        },
      ])
      setCurrentRound(r)
    }
    setIsTraining(false)
  }

  // Chart 1: Accuracy & Loss Convergence
  const accuracyLossChartData = {
    labels: roundsHistory.map((r) => `Round ${r.round}`),
    datasets: [
      {
        label: 'Validation Accuracy',
        data: roundsHistory.map((r) => r.accuracy * 100),
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#06b6d4',
        pointRadius: 4,
      },
      {
        label: 'ROC-AUC',
        data: roundsHistory.map((r) => r.roc_auc * 100),
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#8b5cf6',
        pointRadius: 4,
      },
      {
        label: 'Training Loss (scaled)',
        data: roundsHistory.map((r) => r.loss * 100),
        borderColor: '#f43f5e',
        backgroundColor: 'transparent',
        borderDash: [5, 5],
        tension: 0.4,
        pointBackgroundColor: '#f43f5e',
        pointRadius: 3,
      },
    ],
  }

  // Chart 2: Telemetry Communication Volume
  const networkVolumeData = {
    labels: roundsHistory.map((r) => `R${r.round}`),
    datasets: [
      {
        label: 'Client Uplink (MB)',
        data: roundsHistory.map((r) => r.uplinkMB),
        backgroundColor: 'rgba(59, 130, 246, 0.7)',
        borderRadius: 6,
      },
      {
        label: 'Server Downlink (MB)',
        data: roundsHistory.map((r) => r.downlinkMB),
        backgroundColor: 'rgba(147, 51, 234, 0.7)',
        borderRadius: 6,
      },
    ],
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(51, 65, 85, 0.3)' },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
      },
      y: {
        grid: { color: 'rgba(51, 65, 85, 0.3)' },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
      },
    },
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-display font-bold text-white tracking-tight">
              Federated Training Studio
            </h2>
            <span className="px-2.5 py-0.5 rounded-full bg-blue-950/80 border border-blue-800 text-blue-300 text-xs font-mono">
              FedAvg + DP-SGD
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Orchestrate decentralized model parameter aggregation without transmitting raw patient data
          </p>
        </div>

        {/* Action Button */}
        <button
          onClick={handleStartTraining}
          disabled={isTraining}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-xs flex items-center gap-2 shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50 cursor-pointer shrink-0"
        >
          {isTraining ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Aggregating Round {currentRound + 1}...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Execute Next FL Round</span>
            </>
          )}
        </button>
      </div>

      {/* Control Panel & Config */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 space-y-2">
          <label className="text-xs font-mono text-slate-400 uppercase font-semibold">Model Architecture</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="xgboost">XGBoost Decision Trees</option>
            <option value="random_forest">Random Forest Ensemble</option>
            <option value="transformer">TimeSeriesTransformerNN</option>
          </select>
        </div>

        <div className="glass-panel p-4 space-y-2">
          <label className="text-xs font-mono text-slate-400 uppercase font-semibold">Client Selection Method</label>
          <select
            value={clientSelector}
            onChange={(e) => setClientSelector(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="quantum_qaoa">Quantum QAOA QUBO (Ising)</option>
            <option value="classical_heuristic">Classical Quality Heuristic</option>
          </select>
        </div>

        <div className="glass-panel p-4 space-y-2">
          <label className="text-xs font-mono text-slate-400 uppercase font-semibold">Target Rounds</label>
          <input
            type="number"
            value={numRounds}
            min={1}
            max={20}
            onChange={(e) => setNumRounds(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
          />
        </div>

        <div className="glass-panel p-4 space-y-2 flex flex-col justify-center">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Byzantine Threat Filter</span>
            <span className="text-emerald-400 font-mono">ACTIVE</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-300">
            <ShieldAlert className="w-3.5 h-3.5 text-emerald-400" />
            <span>Cosine & L2 Norm Clip</span>
          </div>
        </div>
      </div>

      {/* Real-time Convergence Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart: Accuracy & ROC-AUC */}
        <div className="lg:col-span-2 glass-panel p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-display font-bold text-white">Federated Convergence Curves</h3>
              <p className="text-xs text-slate-400">Validation metric progression across decentralized aggregation rounds</p>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="flex items-center gap-1 text-cyan-400">
                <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                Acc: {(roundsHistory[roundsHistory.length - 1].accuracy * 100).toFixed(1)}%
              </span>
              <span className="flex items-center gap-1 text-purple-400">
                <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                AUC: {(roundsHistory[roundsHistory.length - 1].roc_auc * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="h-64">
            <Line data={accuracyLossChartData} options={chartOptions} />
          </div>
        </div>

        {/* Network Overhead Chart */}
        <div className="glass-panel p-6 space-y-4">
          <div>
            <h3 className="text-base font-display font-bold text-white">Communication Volume</h3>
            <p className="text-xs text-slate-400">Uplink & Downlink payload bandwidth in MB</p>
          </div>
          <div className="h-64">
            <Bar data={networkVolumeData} options={chartOptions} />
          </div>
        </div>
      </div>

      {/* Participating Edge Clients */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-display font-bold text-white">Client Participants in Round {currentRound}</h3>
            <p className="text-xs text-slate-400">Weights aggregated via Flower NumPyClient interface</p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/80 border border-cyan-800 px-2.5 py-1 rounded-lg">
            3 Clients Aggregated
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {clients.map((c, idx) => (
            <div key={c.id} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white">{c.name}</span>
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              </div>
              <div className="space-y-1 text-xs font-mono text-slate-400">
                <div className="flex justify-between">
                  <span>Samples:</span>
                  <span className="text-white">{c.data_samples_count || 300}</span>
                </div>
                <div className="flex justify-between">
                  <span>Quality Score:</span>
                  <span className="text-cyan-400">{c.data_quality_score || 0.95}</span>
                </div>
                <div className="flex justify-between">
                  <span>DP Noise:</span>
                  <span className="text-indigo-400">σ = 0.01 (L2=1.0)</span>
                </div>
              </div>
              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-emerald-400">
                <span className="flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  <span>Update Verified</span>
                </span>
                <span className="font-mono text-slate-500">Node #{idx + 1}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
