# QFedAutoML REST API Reference & Specification

Base URL: `http://localhost:8000/api/v1`  
Interactive Swagger UI: `http://localhost:8000/docs`  
OpenAPI Specification: `http://localhost:8000/openapi.json`

---

## 1. System & Health Router (`/system`)

### Health Check
**Purpose:** Returns real-time health telemetry of the FastAPI server, database connection, and Qiskit Aer simulator.  
**Where it lives:** [backend/api/routes_system.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_system.py)  
**How to access it:** `GET /api/v1/system/health`  
**Auth requirement:** None (Public)  
**Inputs / Parameters:** None  
**Example Request:**
```bash
curl -X GET http://127.0.0.1:8000/api/v1/system/health
```
**Example Response:**
```json
{
  "status": "healthy",
  "app_name": "QFedAutoML",
  "version": "0.1.0",
  "environment": "development",
  "quantum_simulator": "qiskit_aer",
  "timestamp": "2026-08-16T01:33:03.821798+00:00"
}
```
**Errors:** Returns HTTP 500 if database or simulator environment check fails.  
**Depends on / Feeds into:** Consumed by Navbar status badge.

---

## 2. Authentication Router (`/auth`)

### Register User
**Purpose:** Creates a new user account with bcrypt salted password hashing.  
**Where it lives:** [backend/api/routes_auth.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_auth.py)  
**How to access it:** `POST /api/v1/auth/register`  
**Auth requirement:** None  
**Request Schema:**

| Field | Type | Required? | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `email` | `string` | Yes | — | Unique user email address |
| `username` | `string` | Yes | — | Unique username (3-50 chars) |
| `password` | `string` | Yes | — | Plaintext password (min 8 chars) |
| `full_name` | `string` | No | `null` | Optional full name |

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "researcher@qfed.ai", "username": "dr_ganesh", "password": "SecurePassword123!"}'
```
**Example Response:**
```json
{
  "id": 1,
  "email": "researcher@qfed.ai",
  "username": "dr_ganesh",
  "full_name": null,
  "is_active": true,
  "is_superuser": false,
  "created_at": "2026-08-16T01:00:00Z"
}
```
**Errors:** HTTP 400 if email or username already exists.

---

### Login (JSON Body)
**Purpose:** Authenticates user credentials and returns a signed JWT bearer token.  
**Where it lives:** [backend/api/routes_auth.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_auth.py)  
**How to access it:** `POST /api/v1/auth/login-json`  
**Auth requirement:** None  
**Request Schema:**

| Field | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `username` | `string` | Yes | Username or email |
| `password` | `string` | Yes | Plaintext password |

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login-json \
  -H "Content-Type: application/json" \
  -d '{"username": "dr_ganesh", "password": "SecurePassword123!"}'
```
**Example Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in_minutes": 60,
  "user": {
    "id": 1,
    "email": "researcher@qfed.ai",
    "username": "dr_ganesh"
  }
}
```
**Errors:** HTTP 401 for invalid username or password.

---

## 3. Edge Clients Mesh Router (`/clients`)

### Register Edge Client
**Purpose:** Registers an edge computing device (hospital server, mobile edge node) in the network.  
**Where it lives:** [backend/api/routes_clients.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_clients.py)  
**How to access it:** `POST /api/v1/clients/register`  
**Auth requirement:** None  
**Request Schema:**

| Field | Type | Required? | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `string` | Yes | — | Unique client ID (UUID or node identifier) |
| `name` | `string` | Yes | — | Human-readable node name |
| `client_ip` | `string` | No | `null` | Optional IP address |
| `capabilities` | `object` | No | `{}` | Hardware specs (`{"cpu_cores": 8, "ram_gb": 32}`) |
| `reliability_score`| `float` | No | `1.0` | Initial client reliability score |

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/clients/register \
  -H "Content-Type: application/json" \
  -d '{"id": "node_apollo_01", "name": "Apollo Hospital Node", "capabilities": {"cpu": "M2 Pro", "ram_gb": 32}}'
```
**Example Response:**
```json
{
  "id": "node_apollo_01",
  "name": "Apollo Hospital Node",
  "status": "registered",
  "reliability_score": 1.0,
  "capabilities": { "cpu": "M2 Pro", "ram_gb": 32 },
  "created_at": "2026-08-16T01:10:00Z"
}
```

---

### List Edge Clients
**Purpose:** Retrieves all registered client nodes and their telemetry status.  
**Where it lives:** [backend/api/routes_clients.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_clients.py)  
**How to access it:** `GET /api/v1/clients`  
**Auth requirement:** None  
**Example Response:**
```json
[
  {
    "id": "node_apollo_01",
    "name": "Apollo Hospital Node",
    "status": "online",
    "reliability_score": 1.0,
    "capabilities": { "cpu": "M2 Pro", "ram_gb": 32 },
    "last_heartbeat": "2026-08-16T01:30:00Z",
    "created_at": "2026-08-16T01:10:00Z"
  }
]
```

---

## 4. Federated Training Router (`/training`)

### Start Training Simulation
**Purpose:** Triggers a decentralized federated learning round simulation across edge partitions with communication telemetry logging.  
**Where it lives:** [backend/api/routes_training.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_training.py)  
**How to access it:** `POST /api/v1/training/start`  
**Auth requirement:** None  
**Request Schema:**

| Field | Type | Required? | Default | Valid Values | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `name` | `string` | Yes | — | — | Experiment run name |
| `model_architecture`| `string` | No | `tabular_nn` | `tabular_nn`, `transformer` | Target model architecture |
| `num_clients` | `integer`| No | `3` | `2` to `50` | Number of simulated clients |
| `num_rounds` | `integer`| No | `3` | `1` to `50` | Number of federated rounds |
| `partition_mode` | `string` | No | `non_iid` | `iid`, `non_iid` | Client data partitioning mode |
| `fraction_fit` | `float` | No | `1.0` | `0.1` to `1.0` | Fraction of clients sampled per round |
| `local_epochs` | `integer`| No | `2` | `1` to `10` | Local epochs per client |
| `batch_size` | `integer`| No | `32` | `8` to `256` | Local training batch size |

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"name": "Fed-XGBoost-Round1", "num_clients": 3, "num_rounds": 2, "partition_mode": "non_iid"}'
```
**Example Response:**
```json
{
  "experiment_id": 1,
  "status": "completed",
  "num_rounds": 2,
  "num_clients": 3,
  "final_train_loss": 0.285,
  "final_train_accuracy": 0.942,
  "final_val_accuracy": 0.958,
  "final_val_f1": 0.962,
  "total_comm_mb": 7.5,
  "round_history": [
    { "round": 1, "train_loss": 0.512, "train_accuracy": 0.892, "total_comm_mb": 3.75 },
    { "round": 2, "train_loss": 0.285, "train_accuracy": 0.942, "total_comm_mb": 3.75 }
  ]
}
```

---

## 5. Quantum Optimization Router (`/quantum`)

### Submit Quantum QUBO Job
**Purpose:** Formulates a combinatorial subproblem (feature, client, or hyperparameter selection) into an Ising Hamiltonian and solves it on Qiskit Aer QAOA simulator and Classical Simulated Annealing.  
**Where it lives:** [backend/api/routes_quantum.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_quantum.py)  
**How to access it:** `POST /api/v1/quantum/optimize`  
**Auth requirement:** None  
**Request Schema:**

| Field | Type | Required? | Default | Valid Values | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `job_type` | `string` | Yes | — | `feature_selection`, `client_selection`, `hpo` | Subproblem type |
| `num_qubits` | `integer`| No | `6` | `2` to `16` | Number of qubits (max 16 logical) |
| `circuit_depth`| `integer`| No | `1` | `1`, `2` | QAOA alternating layers $p$ |
| `backend` | `string` | No | `qiskit_aer` | `qiskit_aer`, `classical_fallback` | Execution backend |
| `parameters` | `object` | No | `{}` | — | Additional QUBO penalty multipliers |

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/quantum/optimize \
  -H "Content-Type: application/json" \
  -d '{"job_type": "feature_selection", "num_qubits": 6, "circuit_depth": 1, "backend": "qiskit_aer"}'
```
**Example Response:**
```json
{
  "job_id": 1,
  "job_type": "feature_selection",
  "status": "completed",
  "backend_used": "qiskit_aer",
  "num_qubits": 6,
  "circuit_depth": 1,
  "quantum_objective_value": -2.4821,
  "classical_objective_value": -2.4821,
  "quantum_execution_time_ms": 342.1,
  "classical_execution_time_ms": 12.4,
  "best_bitstring": "110101",
  "selected_indices": [0, 1, 3, 5]
}
```

---

## 6. AutoML Pipeline Router (`/automl`)

### Run AutoML Pipeline
**Purpose:** Triggers end-to-end Classical vs Quantum AutoML pipeline with feature selection, model selection, and Optuna HPO.  
**Where it lives:** [backend/api/routes_automl.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_automl.py)  
**How to access it:** `POST /api/v1/automl/run`  
**Request Schema:**

| Field | Type | Required? | Default | Valid Values | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `model_type` | `string` | No | `xgboost` | `xgboost`, `random_forest`, `logistic_regression` | Downstream classifier |
| `feature_optimizer`| `string` | No | `quantum` | `quantum` (QAOA), `classical` (Mutual Info) | Feature selection solver |
| `hpo_optimizer` | `string` | No | `classical` | `classical` (Optuna TPE), `quantum` | Hyperparameter optimizer |
| `k_features` | `integer`| No | `6` | `2` to `12` | Target features |

**Example Response:**
```json
{
  "status": "completed",
  "model_type": "xgboost",
  "feature_optimizer": "quantum",
  "hpo_optimizer": "classical",
  "feature_selection": { "k": 6, "selected_features": ["mean radius", "mean texture", "mean perimeter", "mean area", "worst radius", "worst perimeter"] },
  "validation_metrics": { "accuracy": 0.9785, "f1": 0.9821, "roc_auc": 0.9945, "precision": 0.9750, "recall": 0.9890 },
  "registered_model_version_id": 4
}
```

---

## 7. Central Model Registry Router (`/models`)

### List Models & Lifecycle Staging
- `GET /api/v1/models`: Retrieves all registered models.
- `GET /api/v1/models/active/production`: Returns the single model version marked `is_production=True`.
- `PUT /api/v1/models/{model_id}/stage`: Promotes model to production stage.

**Example Promotion Request:**
```bash
curl -X PUT http://127.0.0.1:8000/api/v1/models/1/stage \
  -H "Content-Type: application/json" \
  -d '{"is_production": true}'
```
**Example Response:**
```json
{
  "id": 1,
  "model_name": "Wisconsin-Diagnostic-XGBoost",
  "version": "v2.1.0",
  "architecture_type": "xgboost",
  "is_production": true,
  "validation_metrics": { "accuracy": 0.978, "f1": 0.982, "roc_auc": 0.994 }
}
```

---

## 8. Real-time Inference Router (`/predict`)

### Execute Live Inference
**Purpose:** Executes sub-millisecond predictions against the active production model and audits latency and confidence in the `predictions` table.  
**Where it lives:** [backend/api/routes_predict.py](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/api/routes_predict.py)  
**How to access it:** `POST /api/v1/predict`  
**Request Schema:**

| Field | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `features` | `array[float]` | No* | 1D list of continuous numerical features (30 elements for Breast Cancer dataset) |
| `sequence` | `array[array[float]]` | No* | 2D list of shape `(timesteps, channels)` for sensor models |
| `model_id` | `integer` | No | Optional target model ID (defaults to active production model) |

*\* Exactly one of `features` or `sequence` must be provided.*

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```
**Example Response:**
```json
{
  "prediction": 1,
  "predicted_label": "Malignant",
  "probabilities": [0.016, 0.984],
  "confidence_score": 0.984,
  "latency_ms": 0.42,
  "model_version_id": 1,
  "model_name": "Wisconsin-Diagnostic-XGBoost",
  "architecture": "xgboost"
}
```

---

## 9. Explainability & Governance Router (`/explain`)

- `GET /api/v1/explain/shap/{model_id}`: Returns ranked global SHAP feature importance.
- `POST /api/v1/explain/lime/{model_id}`: Returns local LIME surrogate attribution weights for an input vector.
- `GET /api/v1/explain/report/{model_id}/json`: Exports structured JSON Trust report.
- `GET /api/v1/explain/report/{model_id}/html`: Returns standalone styled HTML Trust report.

---

## 10. File & Dataset Upload Endpoints

| Upload Endpoint | Method | Format Accepted | Size Limits | Storage Destination | Partitioning Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/api/v1/automl/profile` | `POST` (JSON/Multipart) | CSV, JSON | Max 50 MB | Stored in `datasets` table or memory buffer | Partitioned across edge clients using Dirichlet ($\alpha=0.5$) or uniform IID via `clients_simulation/data_partitioner.py`. |
| `/api/v1/models/{id}/stage` | `PUT` | JSON | N/A | Updates `is_production` column in `model_versions` table | Demotes prior version to prevent dual-active production states. |
