import React, { useEffect, useState } from 'react'
import { Navbar } from './components/Navbar'
import { Sidebar, NavTab } from './components/Sidebar'
import { OverviewView } from './components/OverviewView'
import { FederatedView } from './components/FederatedView'
import { AutoMLQuantumView } from './components/AutoMLQuantumView'
import { ModelRegistryView } from './components/ModelRegistryView'
import { ExplainabilityView } from './components/ExplainabilityView'
import { PrivacySecurityView } from './components/PrivacySecurityView'
import { PredictLabView } from './components/PredictLabView'
import { ErrorBoundary } from './components/ErrorBoundary'
import {
  fetchClients,
  fetchModels,
  fetchLeaderboard,
  fetchSystemHealth,
  fetchActiveDataset,
  ActiveDatasetInfo
} from './services/api'
import { EdgeClient, ModelVersion, LeaderboardCandidate } from './types'

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('overview')
  const [clients, setClients] = useState<EdgeClient[]>([])
  const [models, setModels] = useState<ModelVersion[]>([])
  const [activeDataset, setActiveDataset] = useState<ActiveDatasetInfo | null>(null)
  const [systemStatus, setSystemStatus] = useState<any>({
    status: 'healthy',
    version: '0.1.0',
    quantum_backend: 'qiskit_aer'
  })
  const [leaderboardData, setLeaderboardData] = useState<{
    total_candidates: number
    best_candidate: any
    leaderboard: LeaderboardCandidate[]
  }>({
    total_candidates: 0,
    best_candidate: null,
    leaderboard: []
  })

  const loadAllData = async () => {
    try {
      const [cRes, mRes, lRes, sRes] = await Promise.all([
        fetchClients(),
        fetchModels(),
        fetchLeaderboard(),
        fetchSystemHealth(),
      ])
      setClients(cRes)
      setModels(mRes)
      setLeaderboardData(lRes)
      setSystemStatus(sRes)
    } catch (err) {
      console.error('Data initialization error:', err)
    }

    try {
      const dRes = await fetchActiveDataset()
      setActiveDataset(dRes)
    } catch {
      setActiveDataset(null)
    }
  }

  useEffect(() => {
    loadAllData()
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Navbar */}
      <Navbar systemStatus={systemStatus} />

      {/* Main Body with Sidebar + Tab Content */}
      <div className="flex-1 flex overflow-hidden">
        <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

        <main className="flex-1 overflow-y-auto p-8 max-w-7xl mx-auto w-full">
          <ErrorBoundary fallbackName="Studio View">
            {activeTab === 'overview' && (
              <OverviewView
                clients={clients}
                models={models}
                onNavigate={(tab: NavTab) => setActiveTab(tab)}
                onDatasetChange={loadAllData}
              />
            )}

            {activeTab === 'federated' && (
              <FederatedView clients={clients} />
            )}

            {activeTab === 'automl-quantum' && (
              <AutoMLQuantumView
                leaderboardData={leaderboardData}
                onRefreshLeaderboard={loadAllData}
                activeDataset={activeDataset}
              />
            )}

            {activeTab === 'model-registry' && (
              <ModelRegistryView
                models={models}
                onRefreshModels={loadAllData}
              />
            )}

            {activeTab === 'explainability' && (
              <ExplainabilityView
                models={models}
                activeDataset={activeDataset}
              />
            )}

            {activeTab === 'privacy-security' && (
              <PrivacySecurityView clients={clients} />
            )}

            {activeTab === 'predict-lab' && (
              <PredictLabView
                models={models}
                activeDataset={activeDataset}
              />
            )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}

