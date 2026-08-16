# QFedAutoML Data, Files & Storage Specification

This document details the exact ingestion pipelines, data partitioning mechanisms, storage destinations, and artifact export capabilities within the QFedAutoML platform.

---

## 1. Summary Reference Matrix

| Aspect | Implementation Details | Exact Path / Table Reference |
| :--- | :--- | :--- |
| **New Dataset Upload** | UI: AutoML & Quantum Studio<br>API: `POST /api/v1/automl/profile` | `backend/database/models_orm.py` (`datasets` table) |
| **Accepted Data Formats** | Tabular CSV, JSON record arrays, NumPy arrays | Validated via `backend/automl/preprocessing.py` |
| **Validation Rules** | Rejects NaNs, requires continuous/categorical splits, checks target column existence | Raises `HTTP 422 Unprocessable Entity` on invalid format |
| **Partitioning Strategy** | Dirichlet Non-IID ($\alpha=0.5$) or Uniform IID | `clients_simulation/data_partitioner.py` (`partition_data_non_iid_dirichlet`) |
| **Model Artifact Storage** | Serialized binary `.joblib` / `.pt` and JSON metadata | `backend/database/models_orm.py` (`model_versions` table) |
| **Quantum Results Persistence** | Execution times, QAOA objective energies, optimal bitstrings | `backend/database/models_orm.py` (`quantum_jobs` table) |
| **Security & Audit Logs** | Anomaly reasons, client IDs, severity levels | `backend/database/models_orm.py` (`security_events` table) |
| **Explanation Report Export**| Standalone styled HTML / JSON API | `GET /api/v1/explain/report/{id}/html` |

---

## 2. Dataset Ingestion & Partitioning Workflow

### 2.1 Tabular & Sensor Datasets
1. **Wisconsin Breast Cancer Diagnostic Dataset**:
   - **Dimensions**: 569 samples $\times$ 30 continuous numerical features.
   - **Preloaded in**: `backend/automl/preprocessing.py` (`load_starter_tabular_dataset`).
   - **Standardization**: `StandardScaler()` applied with stratified 80/20 train-test split.
2. **Multi-Channel Temporal Sensor Dataset**:
   - **Dimensions**: $N$ samples $\times 10$ timesteps $\times 6$ sensor channels.
   - **Preloaded in**: `backend/automl/preprocessing.py` (`load_sensor_timeseries_dataset`).

### 2.2 Client Data Partitioning
Partitions are created using Dirichlet distributions to simulate non-IID statistical heterogeneity across hospital edge nodes:
```python
# clients_simulation/data_partitioner.py
def partition_data_non_iid_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 3,
    alpha: float = 0.5,
    random_state: int = 42
) -> list[ClientDataPartition]
```
- **Locality Guarantee**: Raw partitions are held in memory by `ClientDataPartition` objects on edge clients. The central server receives only flattened gradient parameter arrays.

---

## 3. Model Artifacts & Lifecycle Versioning

1. **Storage Destination**:
   - Metadata, hyperparameters, and validation scores are written to the `model_versions` table.
   - Binary weights are assigned a unique semantic version tag (e.g. `v1.0.0`, `v2.1.0`) and saved in the application data directory.
2. **Production Staging**:
   - Only ONE model can hold `is_production = True` at any given time.
   - Promoting a new model version via `PUT /api/v1/models/{id}/stage` automatically demotes previously active versions to prevent inference routing collisions.

---

## 4. Quantum Optimization Persistence (`quantum_jobs` Table)

Every QAOA or Simulated Annealing run is recorded in the `quantum_jobs` table with the following schema:
- `id`: Unique Job ID.
- `job_type`: `feature_selection`, `client_selection`, or `hpo`.
- `status`: `completed`.
- `backend_used`: `qiskit_aer` (or `classical_fallback`).
- `num_qubits`: Number of qubits simulated ($\le 16$).
- `circuit_depth`: QAOA layer count $p$.
- `objective_value`: Quantum energy minimum.
- `classical_objective_value`: Classical Simulated Annealing minimum energy.
- `execution_time_ms`: Quantum circuit evaluation latency.
- `classical_time_ms`: Classical baseline evaluation latency.
- `result`: JSON payload containing the optimal binary bitstring and selected feature column names.

---

## 5. Artifact Downloads & Governance Reports

1. **HTML Trust & Governance Audit Report**:
   - **URL**: `http://localhost:8000/api/v1/explain/report/{model_id}/html`
   - **UI Access**: Click **Export HTML Trust Report** on the Explainability & Trust page.
   - **Format**: Standalone HTML document containing clinical fidelity scores ($R^2$), global TreeSHAP rankings, local LIME surrogate weights, and differential privacy guarantees for regulatory compliance.
2. **JSON Metric Audit**:
   - **URL**: `http://localhost:8000/api/v1/explain/report/{model_id}/json`
   - **Format**: Structured JSON suitable for CI/CD model deployment pipelines.
