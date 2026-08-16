export interface EdgeClient {
  id: string
  name: string
  status: 'online' | 'training' | 'offline' | 'registered'
  device_info: {
    cpu?: string
    ram_gb?: number
    os?: string
    [key: string]: any
  }
  data_samples_count: number
  data_quality_score: number
  registered_at: string
  last_seen_at: string
  privacy_status?: {
    epsilon_spent: number
    max_epsilon: number
    delta: number
    exhausted: boolean
  }
}

export interface ModelVersion {
  id: number
  model_name: string
  version: string
  architecture_type: string
  model_binary_path?: string | null
  hyperparameters: Record<string, any>
  validation_metrics: {
    accuracy?: number
    f1?: number
    roc_auc?: number
    precision?: number
    recall?: number
    [key: string]: any
  }
  is_production: boolean
  created_at: string
}

export interface LeaderboardCandidate {
  rank?: number
  model_name: string
  search_method: string
  feature_set: string
  accuracy: number
  f1: number
  roc_auc: number
  hyperparameters: Record<string, any>
  execution_time_s: number
}

export interface TrainingRoundMetrics {
  round_num: number
  loss: number
  accuracy: number
  f1: number
  roc_auc: number
  uplink_bytes: number
  downlink_bytes: number
  participating_clients: number
  byzantine_filtered_count: number
}

export interface PredictionResult {
  prediction: number
  predicted_label: string
  probabilities: number[]
  confidence_score: number
  latency_ms: number
  model_version_id: number
  model_name: string
  architecture: string
}

export interface SHAPFeatureImportance {
  feature_name: string
  importance: number
  rank: number
}

export interface LIMEFeatureAttribution {
  feature_name: string
  weight: number
  feature_value: number
}

export interface SecurityEventItem {
  id: number
  event_type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  client_id?: string
  details: Record<string, any>
  timestamp: string
}
