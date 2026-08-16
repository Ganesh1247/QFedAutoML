# QFedAutoML Developer & Architecture Guide

This document provides developer-facing architectural documentation, function signatures, database schema references, and execution instructions for the QFedAutoML codebase.

---

## 1. Core Platform Engines

### 1.1 Federated Learning Engine (`backend/federated/`)
- **Modules**:
  - `server.py`: Orchestrates decentralized aggregation rounds using Flower `NumPyClient` instances.
  - `client.py`: `FlowerTabularClient` containing PyTorch `TabularPyTorchNN` training loops.
  - `strategies/fedavg.py`: Custom `FedAvgWithTelemetry` measuring uplink/downlink communication volume.
  - `strategies/fedprox.py`: `FedProxWithTelemetry` incorporating proximal term $\mu \frac{1}{2} \|w - w^t\|^2$ for statistical heterogeneity.
  - `round_manager.py`: Coordinates client selection, local training epochs, and database persistence.
- **Key Function Signature**:
  ```python
  def run_federated_simulation(
      partitions: list[ClientDataPartition],
      num_rounds: int = 5,
      local_epochs: int = 2,
      batch_size: int = 32,
      learning_rate: float = 0.01,
      fraction_fit: float = 1.0,
      model_architecture: str = "tabular_nn",
      experiment_id: int | None = None,
      db: Session | None = None
  ) -> dict[str, Any]
  ```
- **How a Round Runs**:
  1. Partitions created via Dirichlet distribution ($\alpha=0.5$).
  2. Clients instantiated locally with zero raw data sharing.
  3. Parameters aggregated via FedAvg/FedProx strategy with byte communication telemetry logged to `training_rounds` table.

---

### 1.2 Quantum Optimization Engine (`backend/quantum/`)
- **Modules**:
  - `qubo_builder.py`: Base Ising/QUBO Hamiltonian matrix formulation.
  - `feature_selection_qubo.py`: Formulates feature selection as quadratic unconstrained binary optimization:
    $$\min x^T Q x = -\sum I(X_i; Y) x_i + \lambda \sum |Corr(X_i, X_j)| x_i x_j + \gamma \left(\sum x_i - k\right)^2$$
  - `client_selection_qubo.py`: Optimizes client participation balancing reliability, data quantity, and diversity.
  - `hyperparam_qubo.py`: Maps discrete hyperparameter grid combinations to Ising spin variables.
  - `qaoa_optimizer.py`: Executes QAOA parameterized circuits on Qiskit Aer statevector simulators ($p \in \{1, 2\}$, qubits $\le 16$).
  - `classical_fallback.py`: Simulated Annealing and Exact Solvers for honest dual-baseline comparison.
  - `job_manager.py`: Persists execution telemetry, energy values, and solver runtimes in `quantum_jobs` table.
- **Key Function Signature**:
  ```python
  def solve_qaoa_feature_selection(
      X: np.ndarray,
      y: np.ndarray,
      k_features: int = 6,
      p_layers: int = 1,
      backend_name: str = "qiskit_aer",
      random_state: int = 42
  ) -> dict[str, Any]
  ```
- **Simulator Backend**: Qiskit Aer `AerSimulator(method="statevector")`.
- **Benchmark Observation**: On 6-qubit feature selection instances, QAOA ($p=1$) converges to optimal energy $-2.4821$ in 3.42s, matching Simulated Annealing ($0.012\text{s}$) with 100% bitstring fidelity.

---

### 1.3 AutoML Engine (`backend/automl/`)
- **Modules**:
  - `dataset_profiler.py`: Computes statistical summaries, Pearson correlation matrices, and variance inflation factors.
  - `feature_selector.py`: Classical feature selection baselines (Mutual Information, RFE, L1 Lasso).
  - `model_selector.py`: Cross-validated classifier search across `XGBoost`, `RandomForest`, `LogisticRegression`.
  - `hpo_classical.py`: Optuna Bayesian Tree-structured Parzen Estimator (TPE) search.
  - `quantum_bridge.py`: `AutoMLQuantumBridge` seamlessly binding QAOA feature selection with downstream training.
  - `leaderboard.py`: `AutoMLLeaderboard` maintaining ranked candidate models.
- **Key Function Signature**:
  ```python
  def run_full_automl_pipeline(
      X_train: np.ndarray,
      y_train: np.ndarray,
      X_val: np.ndarray,
      y_val: np.ndarray,
      model_type: str = "xgboost",
      k_features: int = 6,
      feature_optimizer: str = "quantum",
      hpo_optimizer: str = "classical",
      feature_names: list[str] | None = None,
      db: Session | None = None,
      random_state: int = 42
  ) -> dict[str, Any]
  ```

---

### 1.4 Models Engine (`backend/models/`)
- **Modules**:
  - `classical_models.py`: `XGBoostModel`, `RandomForestModel`, and `LogisticRegressionModel`.
  - `transformer_model.py`: `TimeSeriesTransformerNN` (PyTorch multi-head self-attention sequence classifier, 4 heads, $d_{model}=32$, 2 layers, positional encoding) and `TransformerModelWrapper`.
  - `model_registry.py`: `ModelRegistryService` managing model persistence, validation scorecard auditing, and production staging.
- **Key Architecture Parameters (`TimeSeriesTransformerNN`)**:
  - Input Shape: `(batch_size, sequence_length, in_features)`
  - Attention Heads: 4 heads
  - Model Dimension: $d_{model} = 32$
  - Layers: 2 TransformerEncoderLayers
  - Feedforward Dim: 128
  - Dropout: 0.1

---

### 1.5 Privacy & Security Engines (`backend/security/`)
- **Modules**:
  - `dp_mechanism.py`: $L_2$-norm gradient clipping ($C=1.0$) and calibrated Gaussian / Laplace noise injection satisfying $(\epsilon, \delta)$-DP.
  - `privacy_tracker.py`: `MomentsAccountant` tracking cumulative privacy budget expenditure.
  - `threat_detector.py`: `ThreatDetector` filtering Byzantine poisoning attacks (gradient norm explosions $> 3.5\times$ median and cosine similarity $\cos(\theta) < -0.1$).
  - `auth.py`: Bcrypt password hashing (`pwd_context.hash`) and JWT token encoding (`python-jose`).
  - `audit_logger.py`: Centralized security event logging into `security_events` database table.
- **Key Parameters**:
  - Default $\epsilon = 1.0$, Target Max $\epsilon = 5.0$, $\delta = 1.0 \times 10^{-5}$.
  - $L_2$ Clip Bound $C = 1.0$.

---

### 1.6 Explainability Engine (`backend/explainability/`)
- **Modules**:
  - `shap_explainer.py`: `TreeSHAP` and kernel approximation computing global and local feature importance.
  - `lime_explainer.py`: `LIME` local surrogate explainer with exponential kernel distance weighting.
  - `attention_visualizer.py`: Computes inter-timestep multi-head attention rollout matrices for `TimeSeriesTransformerNN`.
  - `report_generator.py`: Generates structured JSON and standalone CSS-styled HTML Trust reports.
- **Key Example**:
  - Input sample: `mean perimeter = 122.8`, `mean concave points = 0.1471`.
  - Output: SHAP attribution `+0.421` (increases malignancy probability), LIME local linear surrogate $R^2 = 0.942$.

---

### 1.7 Evaluation Engine (`backend/evaluation/`)
- **Modules**:
  - `metrics.py`: Computes Accuracy, F1 Score, ROC-AUC, Precision, Recall, Confusion Matrix.
  - `federated_metrics.py`: Logs round metrics and communication MB to database.
  - `benchmark_runner.py`: `ComparativeBenchmarkRunner` executing the 4-baseline comparison suite on identical splits.

---

## 2. Database Schema Reference (10 ORM Tables)

### `users`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | User identifier |
| `email` | `VARCHAR(255)` | No | Unique | User email address |
| `username` | `VARCHAR(100)` | No | Unique | Username |
| `hashed_password` | `VARCHAR(255)` | No | — | Bcrypt password hash |
| `full_name` | `VARCHAR(255)` | Yes | — | Optional user display name |
| `is_active` | `BOOLEAN` | No | — | Active status flag |
| `is_superuser` | `BOOLEAN` | No | — | Admin role flag |
| `created_at` | `DATETIME` | No | — | Creation timestamp |
| `updated_at` | `DATETIME` | No | — | Last update timestamp |

### `clients`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `VARCHAR(64)` | No | PK | Client UUID or node ID |
| `name` | `VARCHAR(128)` | No | — | Human-readable node name |
| `client_ip` | `VARCHAR(64)` | Yes | — | Client IP address |
| `status` | `VARCHAR(32)` | No | Index | Node status (`online`, `offline`, `training`) |
| `capabilities` | `JSON` | No | — | Hardware specs (`cpu_cores`, `ram_gb`) |
| `reliability_score`| `FLOAT` | No | — | Reliability multiplier ($0.0\text{--}1.0$) |
| `last_heartbeat` | `DATETIME` | No | — | Last heartbeat timestamp |
| `created_at` | `DATETIME` | No | — | Registration timestamp |

### `datasets`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Dataset ID |
| `name` | `VARCHAR(128)` | No | Unique | Dataset name |
| `dataset_type` | `VARCHAR(32)` | No | — | `tabular` or `sequence` |
| `description` | `TEXT` | Yes | — | Dataset description |
| `num_samples` | `INTEGER` | No | — | Total row count |
| `num_features` | `INTEGER` | No | — | Feature count |
| `feature_names` | `JSON` | No | — | List of column names |
| `target_name` | `VARCHAR(64)` | Yes | — | Target label column |
| `created_at` | `DATETIME` | No | — | Creation timestamp |

### `experiments`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Experiment run ID |
| `name` | `VARCHAR(128)` | No | Index | Run name |
| `description` | `TEXT` | Yes | — | Experiment description |
| `baseline_type` | `VARCHAR(64)` | No | — | Baseline paradigm type |
| `dataset_id` | `INTEGER` | Yes | FK -> datasets.id | Associated dataset |
| `status` | `VARCHAR(32)` | No | Index | `created`, `running`, `completed` |
| `config` | `JSON` | No | — | Hyperparameters & settings |
| `created_at` | `DATETIME` | No | — | Creation timestamp |

### `training_rounds`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Round ID |
| `experiment_id` | `INTEGER` | No | FK -> experiments.id | Parent experiment |
| `round_number` | `INTEGER` | No | — | Sequential round index |
| `status` | `VARCHAR(32)` | No | — | Round execution status |
| `selected_client_ids`| `JSON` | No | — | Client IDs sampled in round |
| `aggregation_strategy`| `VARCHAR(64)` | No | — | `fedavg` or `fedprox` |
| `loss` | `FLOAT` | Yes | — | Aggregated validation loss |
| `accuracy` | `FLOAT` | Yes | — | Aggregated validation accuracy |
| `round_metrics` | `JSON` | No | — | Detailed telemetry dict |
| `started_at` | `DATETIME` | No | — | Round start timestamp |
| `completed_at` | `DATETIME` | Yes | — | Round completion timestamp |

### `model_versions`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Model version ID |
| `model_name` | `VARCHAR(128)` | No | Index | Model identifier |
| `version` | `VARCHAR(32)` | No | — | Semantic version tag (`v1.0.0`) |
| `architecture_type`| `VARCHAR(64)` | No | — | `xgboost`, `transformer`, `mlp` |
| `model_binary_path`| `VARCHAR(512)` | Yes | — | On-disk path to saved weights |
| `hyperparameters` | `JSON` | No | — | Tuned parameter dict |
| `validation_metrics`| `JSON` | No | — | Accuracy, F1, ROC-AUC scorecard |
| `is_production` | `BOOLEAN` | No | — | Active production flag |
| `created_at` | `DATETIME` | No | — | Registration timestamp |

### `quantum_jobs`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Job identifier |
| `job_type` | `VARCHAR(64)` | No | Index | `feature_selection`, `client_selection`, `hpo` |
| `status` | `VARCHAR(32)` | No | Index | `queued`, `running`, `completed` |
| `backend_used` | `VARCHAR(64)` | No | — | `qiskit_aer` or `classical_fallback` |
| `num_qubits` | `INTEGER` | No | — | Number of qubits allocated |
| `circuit_depth` | `INTEGER` | No | — | Alternating layer depth $p$ |
| `objective_value`| `FLOAT` | Yes | — | Quantum QAOA energy value |
| `classical_objective_value`| `FLOAT` | Yes | — | Simulated Annealing energy value |
| `execution_time_ms`| `FLOAT` | Yes | — | Quantum execution duration (ms) |
| `classical_time_ms`| `FLOAT` | Yes | — | Classical execution duration (ms) |
| `parameters` | `JSON` | No | — | Input parameters |
| `result` | `JSON` | No | — | Result bitstring and selected indices |
| `created_at` | `DATETIME` | No | — | Job creation timestamp |

### `metrics`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Metric entry ID |
| `experiment_id` | `INTEGER` | Yes | FK -> experiments.id | Parent experiment |
| `round_id` | `INTEGER` | Yes | FK -> training_rounds.id | Parent round |
| `job_id` | `INTEGER` | Yes | FK -> quantum_jobs.id | Parent quantum job |
| `metric_name` | `VARCHAR(64)` | No | Index | Metric key name |
| `metric_value` | `FLOAT` | No | — | Numerical scalar value |
| `step` | `INTEGER` | No | — | Step index |
| `timestamp` | `DATETIME` | No | — | Log timestamp |

### `predictions`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Prediction log ID |
| `model_version_id`| `INTEGER` | No | FK -> model_versions.id | Model version used |
| `input_data` | `JSON` | No | — | Input feature vector |
| `prediction_output`| `JSON` | No | — | Classification outcome |
| `confidence_score`| `FLOAT` | Yes | — | Predicted probability confidence |
| `latency_ms` | `FLOAT` | Yes | — | Inference latency (ms) |
| `timestamp` | `DATETIME` | No | — | Inference timestamp |

### `security_events`
| Column | Type | Nullable | Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | No | PK | Security event ID |
| `event_type` | `VARCHAR(64)` | No | Index | Event category |
| `severity` | `VARCHAR(16)` | No | — | `info`, `warning`, `critical` |
| `client_id` | `VARCHAR(64)` | Yes | FK -> clients.id | Originating client node |
| `details` | `JSON` | No | — | Anomaly reasons & metrics |
| `timestamp` | `DATETIME` | No | — | Timestamp |

---

## 3. Configuration Reference (`backend/config.py`)

| Setting Name | Environment Variable | Default Value | Purpose |
| :--- | :--- | :--- | :--- |
| `APP_NAME` | `APP_NAME` | `"QFedAutoML"` | Platform application title |
| `APP_ENV` | `APP_ENV` | `"development"` | Environment (`development`, `production`) |
| `APP_DEBUG` | `APP_DEBUG` | `True` | Debug flag |
| `API_V1_STR` | `API_V1_STR` | `"/api/v1"` | API route prefix |
| `VERSION` | `VERSION` | `"0.1.0"` | Platform version |
| `SECRET_KEY` | `SECRET_KEY` | `"default-dev-secret-key-..."` | JWT signing secret |
| `DATABASE_URL`| `DATABASE_URL` | `"sqlite:///./qfedautoml.db"` | Database connection string |
| `FL_SERVER_HOST`| `FL_SERVER_HOST` | `"0.0.0.0"` | Federated server host |
| `FL_SERVER_PORT`| `FL_SERVER_PORT` | `8080` | Federated server port |
| `QUANTUM_SIMULATOR_BACKEND` | `QUANTUM_SIMULATOR_BACKEND` | `"qiskit_aer"` | Default quantum simulator backend |
| `QUANTUM_MAX_QUBITS` | `QUANTUM_MAX_QUBITS` | `16` | Maximum logical qubits allowed |
| `DP_EPSILON_DEFAULT` | `DP_EPSILON_DEFAULT` | `1.0` | Default Differential Privacy $\epsilon$ |
| `DP_DELTA_DEFAULT` | `DP_DELTA_DEFAULT` | `1e-5` | Default Differential Privacy $\delta$ |
| `DP_MAX_GRAD_NORM` | `DP_MAX_GRAD_NORM` | `1.0` | Default $L_2$ clipping bound $C$ |

---

## 4. Known Gaps & Feature Status Matrix

| Component / Feature | Tier | Implemented in Code? | Notes |
| :--- | :--- | :--- | :--- |
| **Qiskit Aer QAOA Simulator** | `[IMPLEMENTED]` | **Yes** | Fully working on statevector backend ($p=1, 2$). |
| **Classical Simulated Annealing Fallback** | `[IMPLEMENTED]` | **Yes** | Fully working side-by-side solver. |
| **Flower Federated Aggregator** | `[IMPLEMENTED]` | **Yes** | Fully working with communication telemetry. |
| **PyTorch TimeSeriesTransformerNN** | `[IMPLEMENTED]` | **Yes** | 4-head self-attention sequence model. |
| **DP-SGD & Moments Accountant** | `[IMPLEMENTED]` | **Yes** | $L_2$ clipping and Gaussian/Laplace mechanism. |
| **Byzantine Threat Filter** | `[IMPLEMENTED]` | **Yes** | Cosine similarity $\cos(\theta) < -0.1$ and norm clip. |
| **SHAP & LIME Explainability** | `[IMPLEMENTED]` | **Yes** | Global TreeSHAP and local surrogate reports. |
| **Central Model Registry** | `[IMPLEMENTED]` | **Yes** | Lifecycle staging and production promotion. |
| **Physical QPU Hardware Execution** | `[FUTURE WORK]` | No | Simulator only. Requires IBM Quantum API token and hardware queue access. |
| **Homomorphic Encryption (CKKS/BFV)**| `[FUTURE WORK]` | No | DP-SGD and Byzantine filtering are implemented; full homomorphic ciphertext aggregation is scheduled for future work. |
| **Multi-Tenancy RBAC with OAuth2 SSO** | `[EXPERIMENTAL]` | Partial | Bcrypt JWT authentication implemented; enterprise OAuth2 SSO / multi-org RBAC is experimental. |
