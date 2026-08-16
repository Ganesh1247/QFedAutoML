# QFedAutoML Web Dashboard User Guide

This guide explains how to operate and navigate the 7 interactive studios within the QFedAutoML web dashboard ([http://localhost:3000](http://localhost:3000)).

---

## Global Navigation & Layout Architecture

The user interface follows a modern dark glassmorphism aesthetic (`#020617` background with `rgba(15, 23, 42, 0.75)` backdrop blur panels).

- **Top Navbar** ([frontend/src/components/Navbar.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/Navbar.tsx)):
  - **Platform Brand**: Version badge (`v0.1.0 Capstone`).
  - **PyTorch Classical FL Indicator**: Confirms classical gradient computation is active.
  - **Quantum Aer Simulator Status**: Displays simulator backend (`Qiskit Aer QAOA ≤16 Qubits`).
  - **DP-SGD Guard**: Real-time privacy guard status.
  - **Database Sync**: Live indicator for SQLite / PostgreSQL connectivity.
- **Left Sidebar** ([frontend/src/components/Sidebar.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/Sidebar.tsx)):
  - Provides 1-click tab switching between all 7 modules.
  - Architecture Callout: Reminds operators that neural network training runs strictly on classical hardware.

---

## 1. Executive Overview Studio

### Executive Overview
**Purpose:** Provides an executive summary of platform operations, comparing all 4 baseline paradigms side-by-side, monitoring real-time production model accuracy, edge node fleet health, and privacy budget consumption.
**Where it lives:** [frontend/src/components/OverviewView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/OverviewView.tsx)
**How to access it:** Click **Executive Overview** in the left sidebar or navigate to default route at `http://localhost:3000`.
**Inputs / Parameters:**

| Parameter / Action | Type | Required? | Default | What it controls |
| :--- | :--- | :--- | :--- | :--- |
| **Launch Federated Studio** | Button | No | — | Navigates directly to the Federated Training Studio tab. |
| **Run Quantum QUBO Solver** | Button | No | — | Navigates directly to the AutoML & Quantum Studio tab. |
| **Manage Nodes** | Button | No | — | Navigates directly to the Federated Studio client participants list. |

**Example:**
On initial mount, the dashboard loads system health and registered client telemetry:
- Production Model ROC-AUC: **99.4%**
- Active Edge Nodes: **3 / 3 Online**
- Quantum Simulator: **Qiskit Aer QAOA ($p=1, 2$)**
- Cumulative Differential Privacy Budget: **$\epsilon = 1.42 / 5.0$**

**Output / Rendering:**
1. **Hero Banner**: High-level platform mission callout with quick action launch buttons.
2. **4 KPI Scorecards**:
   - Active Production Score (ROC-AUC / Model Name).
   - Edge Nodes In Mesh (Online count + 100% Heartbeat Health indicator).
   - Quantum Optimizer (Qiskit Aer QAOA + Ising Fallback Verified).
   - Privacy Budget ($\epsilon=1.42 / 5.0$, $\delta=10^{-5}$, Moments Accountant).
3. **4-Baseline Comparison Matrix**:
   - Baseline 1: Centralized ML (`XGBoostModel` — 96.5% Accuracy).
   - Baseline 2: Classical FL (`FedAvg` — 95.8% Accuracy).
   - Baseline 3: Federated Transformer (`TimeSeriesTransformerNN` — 98.1% ROC-AUC).
   - Baseline 4: Proposed QFedAutoML (`QAOA QUBO + Optuna` — 99.4% ROC-AUC).
4. **Registered Edge Clients Table**: Node ID, Name, Status badge, CPU/RAM specs, Samples count, and Privacy $\epsilon$ spent.
5. **Trust & Compliance Checklist**: Displays status of Data Sovereignty, Classical Neural Execution, Byzantine Threat Filter, and Dual Solver Verification.

**File/data upload details:** N/A on overview.
**Errors:** If backend API is unreachable on initial mount, falls back gracefully to cached mock client and model structures while displaying offline status.
**Depends on / Feeds into:** Fetches data via `fetchClients()`, `fetchModels()`, and `fetchSystemHealth()` in [frontend/src/services/api.ts](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/services/api.ts).

---

## 2. Federated Training Studio

### Federated Training Studio
**Purpose:** Allows ML engineers to configure, launch, and monitor decentralized federated learning rounds with live loss and accuracy convergence tracking, communication volume auditing, and client node aggregation.
**Where it lives:** [frontend/src/components/FederatedView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/FederatedView.tsx)
**How to access it:** Click **Federated Studio** in the left sidebar.
**Inputs / Parameters:**

| Parameter | Type | Required? | Default | Valid Range | What it controls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | Dropdown | Yes | `xgboost` | `xgboost`, `random_forest`, `transformer` | Target architecture to train in decentralized rounds. |
| **Client Selection Method**| Dropdown | Yes | `quantum_qaoa` | `quantum_qaoa`, `classical_heuristic` | Algorithm used to select client subset for round. |
| **Target Rounds** | Number input | Yes | `5` | `1` to `20` | Number of federated aggregation rounds. |
| **Execute Next FL Round** | Button | No | — | — | Starts local/federated simulation round step. |

**Example:**
Clicking **Execute Next FL Round** triggers a simulated federated aggregation round, advancing round count from Round 3 to Round 4:
- Accuracy increases from $95.8\%$ to $97.0\%$.
- ROC-AUC advances to $99.0\%$.
- Client uplink / server downlink traffic is audited ($1.24\text{ MB} / 2.15\text{ MB}$).

**Output / Rendering:**
1. **Chart.js Line Chart (Federated Convergence Curves)**:
   - X-Axis: Federated Round Number (`Round 1`, `Round 2`, `Round 3`...).
   - Y-Axis: Metric Value ($0\text{--}100\%$).
   - Datasets: Validation Accuracy (Cyan line `#06b6d4`), ROC-AUC (Purple line `#8b5cf6`), Training Loss (Red dashed line `#f43f5e`).
2. **Chart.js Bar Chart (Communication Volume)**:
   - X-Axis: Round (`R1`, `R2`, `R3`...).
   - Y-Axis: Megabytes (MB).
   - Datasets: Client Uplink (Blue bar `rgba(59, 130, 246, 0.7)`), Server Downlink (Purple bar `rgba(147, 51, 234, 0.7)`).
3. **Participating Edge Clients Grid**: Displays 3 active participant cards with sample count, quality score, and DP noise configuration ($\sigma=0.01, L_2=1.0$).

**File/data upload details:** Client training consumes localized private data partitions from `clients_simulation/data_partitioner.py`.
**Errors:** Disables training button and shows spinner (`Aggregating Round N...`) while simulation is executing.
**Depends on / Feeds into:** Feeds metrics into `backend/database/models_orm.py` (`training_rounds` and `metrics` tables).

---

## 3. AutoML & Quantum Optimization Studio

### AutoML & Quantum Optimization Studio
**Purpose:** Formulates combinatorial feature selection and hyperparameter optimization as Ising/QUBO Hamiltonian minimization subproblems, executes QAOA quantum circuits on Qiskit Aer simulators, and maintains the ranked candidate leaderboard.
**Where it lives:** [frontend/src/components/AutoMLQuantumView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/AutoMLQuantumView.tsx)
**How to access it:** Click **AutoML & Quantum** in the left sidebar.
**Inputs / Parameters:**

| Parameter | Type | Required? | Default | Valid Range | What it controls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Feature Optimizer** | Button Toggle | Yes | `quantum` | `quantum` (QAOA), `classical` (Mutual Info) | Solver used to solve the feature selection QUBO. |
| **Target Features ($k$)** | Slider | Yes | `6` | `2` to `12` | Number of qubits / features to select. |
| **Classifier Architecture**| Dropdown | Yes | `xgboost` | `xgboost`, `random_forest`, `logistic_regression` | Downstream classifier evaluated with selected features. |
| **QAOA Circuit Depth ($p$)**| Dropdown | Yes | `1` | `1` (Fast Statevector), `2` (Deep Variational) | Number of alternating Hamiltonian layers in QAOA. |
| **Execute AutoML Pipeline**| Button | No | — | — | Submits job to `POST /api/v1/automl/run`. |

**Example:**
Clicking **Execute AutoML Pipeline** with Quantum QAOA ($k=6, p=1$):
- Runs Qiskit Aer QAOA statevector circuit.
- Selects top 6 features based on minimum Ising energy.
- Runs Optuna Bayesian TPE hyperparameter optimization.
- Generates validation scorecard: **Accuracy: 97.8%**, **ROC-AUC: 99.4%**.
- Automatically registers new model artifact as ID #4 in Model Registry.

**Output / Rendering:**
1. **Mathematical QUBO Hamiltonian Card**: Renders the exact cost formula:
   $$\min x^T Q x = -\sum I(X_i; Y) x_i + \lambda \sum |Corr(X_i, X_j)| x_i x_j + \gamma \left(\sum x_i - k\right)^2$$
2. **Latest Execution Summary Card**: Appears upon job completion showing selected feature count, accuracy, ROC-AUC, and assigned model version ID.
3. **AutoML Ranked Candidate Leaderboard Table**: Ranked list of evaluated models with rank badge (#1 Gold Flame), model name, search method, feature set, accuracy, ROC-AUC, and runtime in seconds.

**File/data upload details:** Datasets profiled via `backend/automl/dataset_profiler.py`.
**Errors:** Displays loading indicator (`Simulating QAOA Circuit (p=1)...`) during quantum execution.
**Depends on / Feeds into:** Calls `POST /api/v1/automl/run` and `GET /api/v1/automl/leaderboard`.

---

## 4. Model Registry & Staging Studio

### Model Registry & Staging Studio
**Purpose:** Serves as the central repository for trained model artifacts, scorecards, and hyperparameter logs, enabling 1-click promotion to active production.
**Where it lives:** [frontend/src/components/ModelRegistryView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/ModelRegistryView.tsx)
**How to access it:** Click **Model Registry** in the left sidebar.
**Inputs / Parameters:**

| Parameter / Action | Type | Required? | Default | What it controls |
| :--- | :--- | :--- | :--- | :--- |
| **Select Model Card** | Click | Yes | Model #1 | Selects a model version to inspect in detail. |
| **Promote to Production** | Button | No | — | Calls `PUT /api/v1/models/{id}/stage` to make the selected model active for live inference. |

**Example:**
Selecting `Wisconsin-Diagnostic-XGBoost (v2.1.0)`:
- Accuracy: **97.8%**, F1: **98.2%**, ROC-AUC: **99.4%**, Precision: **97.5%**.
- Clicking **Promote to Production Stage** automatically demotes any previous production model and sets `is_production: true` on this version.

**Output / Rendering:**
1. **Model Versions List**: Cards showing model name, semantic version tag (`v2.1.0`), architecture (`XGBOOST`), creation timestamp, ROC-AUC, and `ACTIVE PRODUCTION` badge.
2. **Model Artifact Inspector Panel**:
   - Production Staging Banner: Shows whether the model is currently receiving 100% of live traffic via `/api/v1/predict`.
   - Validation Scorecard: Accuracy, F1 Score, ROC-AUC, Precision.
   - Tuned Hyperparameters JSON Viewer: Syntax-highlighted JSON block with parameters (e.g. `n_estimators`, `max_depth`, `learning_rate`).

**File/data upload details:** Binary model weights stored in `backend/database/models_orm.py` (`model_versions` table and local disk).
**Errors:** Disables button during promotion request.
**Depends on / Feeds into:** Directly controls which model handles requests at `POST /api/v1/predict`.

---

## 5. Explainability & Trust Center

### Explainability & Trust Center
**Purpose:** Delivers transparent model interpretability via global TreeSHAP rankings, local LIME surrogate attributions, Transformer multi-head self-attention rollouts, and downloadable compliance reports.
**Where it lives:** [frontend/src/components/ExplainabilityView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/ExplainabilityView.tsx)
**How to access it:** Click **Explainability & Trust** in the left sidebar.
**Inputs / Parameters:**

| Parameter | Type | Required? | Default | Valid Range | What it controls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inspecting Model** | Dropdown | Yes | Active Production Model | All registered models | Selects model version to explain. |
| **Export HTML Trust Report**| Link/Button| No | — | — | Opens `/api/v1/explain/report/{id}/html` in new browser tab. |

**Example:**
Selecting model ID #1 generates global SHAP rankings:
1. `mean perimeter` (Impact: 0.384)
2. `mean concave points` (Impact: 0.342)
3. `worst radius` (Impact: 0.289)

**Output / Rendering:**
1. **Global SHAP Feature Importance (Chart.js Bar Chart)**:
   - X-Axis: Feature Name (`mean perimeter`, `mean concave points`...).
   - Y-Axis: Mean |SHAP Value| magnitude.
2. **Local LIME Surrogate Attribution (Chart.js Bar Chart)**:
   - Positive Risk Features: Rendered in green bars (`rgba(16, 185, 129, 0.75)`).
   - Negative Risk Features: Rendered in red bars (`rgba(244, 63, 94, 0.75)`).
3. **Transformer Multi-Head Self-Attention Rollout (4 Heatmaps)**:
   - Renders 4 separate 6x6 attention heatmaps for Heads 1 to 4 visualizing temporal inter-timestep attention intensity across sequence windows.
4. **HTML Trust Report**: Generates an audit report for clinical governance.

**File/data upload details:** Explanations computed on live memory models via `backend/explainability/`.
**Errors:** Shows animated spinner (`Computing SHAP & LIME Attributions...`) while computing.
**Depends on / Feeds into:** Calls `GET /api/v1/explain/shap/{id}` and `POST /api/v1/explain/lime/{id}`.

---

## 6. Privacy & Security Center

### Privacy & Security Center
**Purpose:** Tracks Differential Privacy budget consumption ($\epsilon, \delta$) via the Moments Accountant, calibrates DP-SGD gradient noise/clipping bounds, and audits Byzantine threat mitigations.
**Where it lives:** [frontend/src/components/PrivacySecurityView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/PrivacySecurityView.tsx)
**How to access it:** Click **Privacy & Security** in the left sidebar.
**Inputs / Parameters:**

| Parameter | Type | Required? | Default | Valid Range | What it controls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L2 Clip Bound ($C$)** | Slider | Yes | `1.0` | `0.5` to `5.0` (step `0.1`) | Maximum $L_2$ norm for client parameter updates. |
| **Noise Multiplier ($\sigma$)**| Display | — | `0.01` | Fixed calibrated | Gaussian noise standard deviation. |
| **Max Epsilon ($\epsilon_{max}$)**| Display | — | `5.0` | Target limit | Privacy budget threshold. |

**Example:**
The Moments Accountant tracks cumulative privacy:
- $\epsilon = 1.42 / 5.0$ (28.4% budget consumed).
- $\delta = 1.0 \times 10^{-5}$ maintained.

**Output / Rendering:**
1. **Privacy Budget Gauge**: Gradient progress bar showing $\epsilon$ expenditure.
2. **DP Mechanism Slider**: Interactive slider controlling gradient clipping threshold $C$.
3. **Byzantine Defense Status Card**: Displays active cosine similarity filter rule ($\cos(\theta) < -0.1$ rejection).
4. **Security Events & Audit Log**: Table of real-time security events persisted in the database (`BYZANTINE_ANOMALY_BLOCKED`, `DP_NOISE_INJECTED`, `PRIVACY_ACCOUNTANT_STEP`, `COSINE_DIVERGENCE_FLAGGED`) with severity badges (`HIGH`, `MEDIUM`, `LOW`).

**File/data upload details:** Security events persisted in `security_events` table.
**Errors:** Highlights critical Byzantine anomaly events in red alert boxes.
**Depends on / Feeds into:** Connected to `backend/security/privacy_tracker.py` and `backend/security/threat_detector.py`.

---

## 7. Real-time Inference Lab

### Real-time Inference Lab
**Purpose:** Allows clinicians and operators to test real-time predictions against production models using 30-feature tabular inputs or 10-timestep multi-sensor sequences, measuring sub-millisecond latency and confidence distributions.
**Where it lives:** [frontend/src/components/PredictLabView.tsx](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/frontend/src/components/PredictLabView.tsx)
**How to access it:** Click **Inference Lab** in the left sidebar.
**Inputs / Parameters:**

| Parameter | Type | Required? | Default | Valid Range | What it controls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Mode** | Button Toggle | Yes | `Tabular` | `Tabular` (30 features), `Sequence` (10x6) | Input modality. |
| **Presets** | Buttons | No | — | `Sample A (Malignant)`, `Sample B (Benign)` | Loads diagnostic test vectors. |
| **Feature Inputs** | 30 Numeric Fields | Yes | Pre-loaded | Continuous floats | Feature values fed into model. |
| **Run Live Inference** | Button | No | — | — | Executes `POST /api/v1/predict`. |

**Example:**
Loading `Sample A (Malignant)` and clicking **Run Live Inference**:
- Inference latency: **0.42 ms**
- Classification: **Malignant**
- Confidence score: **98.4%**
- Probabilities: Benign: **1.6%**, Malignant: **98.4%**
- Database confirmation: Logged to `predictions` table with assigned record ID.

**Output / Rendering:**
1. **Input Features Panel**: 30 labeled feature cards with real-time numeric inputs and scrollable viewport.
2. **Outcome Card**: Large banner indicating predicted class (`Malignant` in red `#f43f5e` or `Benign` in emerald `#10b981`).
3. **Probability Distribution Progress Meters**: Visual breakdown for Benign vs Malignant probabilities.
4. **Latency & Audit Telemetry Box**: Displays execution latency in ms, active Model Version ID, and database audit confirmation.

**File/data upload details:** All predictions logged to `predictions` database table.
**Errors:** Validates feature count and sequence dimensions before submission.
**Depends on / Feeds into:** Calls `POST /api/v1/predict` and reads active production model from `backend/models/model_registry.py`.
