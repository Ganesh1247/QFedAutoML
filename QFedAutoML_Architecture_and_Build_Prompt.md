# QFedAutoML — System Architecture, Codebase Structure & AI-IDE Build Prompt

> Scope note: You already have the full 33-section conceptual spec for QFedAutoML. This document delivers the three things you asked for last: (1) the complete, internally-consistent system architecture, (2) the exact order/structure of files in the codebase, and (3) a ready-to-paste master prompt for building it with an AI coding agent (Antigravity, Cursor, Claude Code, Windsurf, etc.). Everything here respects the accuracy rules from your spec: no invented quantum advantage, implemented vs. future features kept separate, classical fallbacks named everywhere a quantum method is proposed.

---

## 1. Complete System Architecture

### 1.1 Design principle

QFedAutoML is a **classical federated-learning + AutoML platform with an optional, isolated quantum-optimization microservice**. The neural network training (Transformer / classical models) always runs classically, on each client. Quantum computing is invoked **only** for small combinatorial subproblems (feature selection, hyperparameter search, client selection) that can be expressed as QUBO/Ising problems and solved via simulators (Qiskit Aer / PennyLane) with a classical solver (e.g., simulated annealing or Bayesian optimization) always running in parallel as a baseline. This keeps the system implementable, testable without real quantum hardware, and scientifically honest.

### 1.2 Layered architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 6: PRESENTATION                                                      │
│   React + Tailwind Dashboard  |  REST/GraphQL consumers  |  CLI            │
└───────────────────────────────────────────────────────────────────────────┘
                                   │  HTTPS/JWT
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 5: API GATEWAY                                                       │
│   FastAPI  →  Auth (JWT)  →  Rate limiting  →  Request routing             │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 4: ORCHESTRATION                                                     │
│   Federated Learning Controller  |  AutoML Controller  |  Experiment Mgr   │
│   (decides WHAT to run: which model, which clients, which HPO strategy)    │
└───────────────────────────────────────────────────────────────────────────┘
              │                    │                       │
              ▼                    ▼                       ▼
┌───────────────────┐  ┌────────────────────────┐  ┌───────────────────────┐
│ Layer 3a:          │  │ Layer 3b:               │  │ Layer 3c:              │
│ FEDERATED ENGINE   │  │ QUANTUM OPT. ENGINE     │  │ AUTOML ENGINE          │
│ (Flower-based)     │  │ (Qiskit/PennyLane, sim) │  │ (search + eval loop)   │
│ - client registry  │  │ - QUBO builder          │  │ - dataset profiler     │
│ - client selection │  │ - QAOA / annealing      │  │ - model selector       │
│ - FedAvg/FedProx    │  │ - classical fallback    │  │ - HPO orchestrator     │
│ - secure aggregation│ │ - result validator       │  │ - leaderboard          │
└───────────────────┘  └────────────────────────┘  └───────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 2: CLIENT / EDGE LAYER  (simulated processes or real edge nodes)     │
│   ┌──────────────┬──────────────┬──────────────┐                          │
│   │  Client A     │  Client B     │  Client C ...│                          │
│   │  local data   │  local data   │  local data  │  ← never leaves client   │
│   │  preprocessing│  preprocessing│  preprocessing│                          │
│   │  Transformer/ │  Transformer/ │  Transformer/│                          │
│   │  ML model     │  ML model     │  ML model    │                          │
│   │  local trainer│  local trainer│  local trainer│                          │
│   │  DP noise     │  DP noise     │  DP noise    │                          │
│   │  secure comms │  secure comms │  secure comms│                          │
│   └──────────────┴──────────────┴──────────────┘                          │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 1: DATA SOURCES (Hospitals / IoT devices / Banks / Vehicles / Plants)│
│   Raw data physically stays local. Only model updates leave the client.    │
└───────────────────────────────────────────────────────────────────────────┘

Cross-cutting layers (touch every layer above):
┌───────────────────────────────────────────────────────────────────────────┐
│ SECURITY & PRIVACY  (TLS, JWT, DP-SGD, secure aggregation, threat logging) │
│ EXPLAINABILITY      (SHAP, LIME, attention maps, post-hoc reports)        │
│ MONITORING/MLOps    (metrics store, model registry, logging, tracing)     │
│ PERSISTENCE         (PostgreSQL: users, clients, rounds, models, jobs)    │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Detailed ASCII deployment diagram

```
                                 ┌────────────┐
                                 │   Users    │
                                 └─────┬──────┘
                                       ▼
                          ┌────────────────────────┐
                          │   React Web Dashboard   │
                          └───────────┬────────────┘
                                       ▼ HTTPS
                          ┌────────────────────────┐
                          │  FastAPI API Gateway     │
                          │  (JWT auth, routing)     │
                          └───────────┬────────────┘
                                       ▼
                    ┌──────────────────────────────────┐
                    │  Federated Learning Controller     │
                    │  (round manager, orchestration)    │
                    └───────┬───────────────┬───────────┘
                            ▼               ▼
             ┌─────────────────────┐  ┌──────────────────────┐
             │  AutoML Engine       │  │  Quantum Optimizer    │
             │  - model search      │  │  QAOA / annealing sim │
             │  - HPO (classical)   │  │  + classical fallback │
             └───────────┬──────────┘  └──────────┬────────────┘
                          └────────────┬───────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │  Client Selection Module    │
                        └─────────────┬─────────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │   Federated Training Round  │
                        └─────────────┬─────────────┘
             ┌─────────────────────────┼─────────────────────────┐
             ▼                         ▼                         ▼
   ┌──────────────────┐     ┌──────────────────┐      ┌──────────────────┐
   │     Client A       │     │     Client B       │      │     Client C       │
   │ Transformer/ML     │     │ Transformer/ML     │      │ Transformer/ML     │
   │ Local Training      │     │ Local Training      │      │ Local Training      │
   │ DP + Privacy Module │     │ DP + Privacy Module │      │ DP + Privacy Module │
   └─────────┬──────────┘     └─────────┬──────────┘      └─────────┬──────────┘
             └─────────────────────────┼─────────────────────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │    Secure Aggregation       │
                        │    (FedAvg / FedProx)       │
                        └─────────────┬─────────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │       Global Model          │
                        └─────────────┬─────────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │     Explainable AI Layer     │
                        │  (SHAP / LIME / attention)   │
                        └─────────────┬─────────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │       Model Registry         │
                        └─────────────┬─────────────┘
                                       ▼
                        ┌───────────────────────────┐
                        │      Prediction API           │
                        └───────────────────────────┘
```

**Change from your draft:** I split "Client Selection / AutoML" into two parallel siblings (AutoML Engine and Quantum Optimizer) feeding into Client Selection, since in your spec the quantum engine also optimizes feature selection and hyperparameters — not only client selection. This keeps the diagram consistent with Section 7 of your spec.

---

## 2. Complete Codebase File Structure

```
qfedautoml/
│
├── README.md
├── LICENSE
├── .env.example
├── docker-compose.yml
├── pyproject.toml / requirements.txt
│
├── backend/
│   ├── main.py                          # FastAPI app entrypoint
│   ├── config.py                        # env config, settings via pydantic
│   ├── dependencies.py                  # shared DI (db session, auth)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_auth.py               # /auth/register, /auth/login
│   │   ├── routes_clients.py            # /clients/*
│   │   ├── routes_training.py           # /training/*
│   │   ├── routes_quantum.py            # /quantum/*
│   │   ├── routes_predict.py            # /predict, /explain
│   │   ├── routes_models.py             # /models/*
│   │   └── routes_system.py             # /system/health
│   │
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── server.py                    # Flower-based FL server wrapper
│   │   ├── client.py                    # Flower client wrapper (simulated node)
│   │   ├── strategies/
│   │   │   ├── fedavg.py
│   │   │   ├── fedprox.py
│   │   │   └── robust_aggregation.py
│   │   ├── client_selector.py           # feeds candidates to quantum/automl
│   │   └── round_manager.py             # orchestrates one FL round end-to-end
│   │
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── qubo_builder.py              # translates ML problems → QUBO
│   │   ├── qaoa_optimizer.py            # Qiskit/PennyLane QAOA implementation
│   │   ├── quantum_annealing_sim.py     # simulated-annealing style fallback
│   │   ├── classical_fallback.py        # classical solver used for comparison
│   │   ├── feature_selection_qubo.py
│   │   ├── hyperparam_qubo.py
│   │   ├── client_selection_qubo.py
│   │   └── job_manager.py               # tracks async quantum jobs, status
│   │
│   ├── automl/
│   │   ├── __init__.py
│   │   ├── dataset_profiler.py          # analyzes dataset stats, dtype, size
│   │   ├── preprocessing.py
│   │   ├── feature_selector.py          # classical (mutual info, RFE, etc.)
│   │   ├── model_selector.py            # chooses among candidate architectures
│   │   ├── hpo_classical.py             # grid/random/Bayesian search
│   │   ├── hpo_quantum_bridge.py        # calls quantum/ engine, compares results
│   │   └── leaderboard.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── classical_models.py          # sklearn/XGBoost wrappers
│   │   ├── transformer_model.py         # sequence model for time-series clients
│   │   └── model_registry.py            # versioning, save/load
│   │
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── differential_privacy.py      # DP-SGD, epsilon accounting
│   │   ├── secure_aggregation.py
│   │   └── privacy_budget_tracker.py
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── auth.py                      # JWT issuing/verification
│   │   ├── tls_config.py
│   │   ├── threat_detection.py          # basic anomaly/poisoning checks
│   │   └── audit_log.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── attention_visualizer.py
│   │   └── report_generator.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                   # accuracy, F1, ROC-AUC, MAE/RMSE
│   │   ├── federated_metrics.py         # rounds, comm cost, convergence
│   │   ├── quantum_metrics.py           # qubits, depth, iterations, obj value
│   │   └── benchmark_runner.py          # runs baselines 1–4 from Section 19
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models_orm.py                # SQLAlchemy ORM (see schema, Sec 3)
│   │   ├── migrations/                  # Alembic migrations
│   │   └── repositories/
│   │       ├── user_repo.py
│   │       ├── client_repo.py
│   │       ├── training_repo.py
│   │       ├── quantum_job_repo.py
│   │       └── model_repo.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics_exporter.py          # Prometheus-style counters
│   │   └── tracing.py
│   │
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── e2e/
│
├── clients_simulation/
│   ├── run_client.py                    # spins up N simulated edge clients
│   └── data_partitioner.py              # IID / Non-IID splitting utilities
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── api/                         # typed API client
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Clients.tsx
│   │   │   ├── TrainingRounds.tsx
│   │   │   ├── QuantumJobs.tsx
│   │   │   ├── ModelRegistry.tsx
│   │   │   ├── Explainability.tsx
│   │   │   └── Login.tsx
│   │   ├── components/
│   │   │   ├── charts/                  # Chart.js wrappers
│   │   │   ├── tables/
│   │   │   └── layout/
│   │   └── store/                       # state management
│   └── public/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_classical_baseline.ipynb
│   ├── 03_federated_baseline.ipynb
│   ├── 04_quantum_vs_classical_hpo.ipynb
│   └── 05_privacy_accuracy_tradeoff.ipynb
│
├── docs/
│   ├── architecture.md                  # this document, trimmed for repo
│   ├── api_reference.md
│   ├── setup_guide.md
│   └── research/
│       ├── literature_review.md
│       └── experiment_results.md
│
└── infra/
    ├── docker/
    │   ├── backend.Dockerfile
    │   ├── frontend.Dockerfile
    │   └── client.Dockerfile
    └── ci/
        └── github_actions.yml
```

---

## 3. Prerequisites — Install Before You Start the AI Agent

Do these yourself, in a terminal, **before** handing the master prompt to the AI IDE. Letting the agent guess versions is the single biggest source of avoidable errors in a stack this wide (Qiskit/PennyLane/PyTorch/Flower all pin specific compatible versions).

### 3.1 System-level tools

| Tool | Why | Check / Install |
|---|---|---|
| Python 3.11 | Backend + ML + quantum libs; 3.12+ breaks some Qiskit/PennyLane versions as of early 2026 | `python3 --version` → install via [python.org](https://www.python.org) or `pyenv install 3.11.9` |
| Node.js 18 or 20 LTS | React frontend build tooling | `node --version` → install via [nodejs.org](https://nodejs.org) or `nvm install 20` |
| PostgreSQL 15+ | Primary database | `psql --version` → install via your OS package manager, or skip and just use the Dockerized version in `docker-compose.yml` |
| Docker + Docker Compose | Containerized dev/deploy | `docker --version && docker compose version` |
| Git | Version control | `git --version` |

### 3.2 Python environment (backend, ML, federated, quantum)

```bash
# From the qfedautoml/ repo root
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip

# --- Backend / API ---
pip install fastapi uvicorn[standard] pydantic pydantic-settings \
            sqlalchemy alembic psycopg2-binary python-jose[cryptography] \
            passlib[bcrypt] python-dotenv httpx

# --- Classical ML ---
pip install torch --index-url https://download.pytorch.org/whl/cpu   # use the CUDA wheel instead if you have a GPU
pip install scikit-learn xgboost pandas numpy

# --- Federated learning ---
pip install flwr

# --- Quantum (simulators only) ---
pip install qiskit qiskit-aer pennylane

# --- AutoML / HPO ---
pip install optuna

# --- Privacy ---
pip install opacus

# --- Explainability ---
pip install shap lime

# --- Testing / dev quality ---
pip install pytest pytest-cov ruff black
```

Freeze this once it's working: `pip freeze > requirements.txt` — commit that file, since Qiskit/PennyLane/Flower APIs shift between minor versions and an unpinned install six months from now may not match what the agent wrote code against.

### 3.3 Frontend environment

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install chart.js react-chartjs-2 axios react-router-dom
```

### 3.4 IDE / editor extensions

These apply whether you're in VS Code, Antigravity, Cursor, or Windsurf — they're all VS Code–based forks and use the same extension marketplace:

| Extension | Purpose |
|---|---|
| Python (Microsoft) | Python language support, debugging |
| Pylance | Type checking, import resolution — catches a large share of errors before runtime |
| Ruff | Fast Python linting, matches the `ruff` installed above |
| Docker | Manage/inspect containers from the editor |
| ESLint | Frontend lint |
| Prettier | Consistent formatting (JS/TS/JSON/MD) |
| Tailwind CSS IntelliSense | Autocomplete for utility classes |
| SQLTools (+ PostgreSQL driver) | Inspect the Postgres schema without leaving the editor |
| GitLens | Diff/blame while reviewing agent-generated commits |
| Thunder Client or REST Client | Manually test FastAPI endpoints as each phase completes |

I don't have verified, current specifics on Antigravity's own built-in feature set beyond it being a VS Code–based AI IDE, so if it has a proprietary extension panel, check its own marketplace for equivalents of the above rather than assuming these exact names are listed there.

### 3.5 Verify before starting Phase 1

```bash
python -c "import torch, sklearn, flwr, qiskit, pennylane, shap; print('OK')"
node -v && npm -v
docker compose version
```

If any import fails, resolve it now — don't let the agent debug a broken environment for you as its first task; that burns a large share of the context budget on installation noise instead of actual project code.

---

## 4. Master Prompt for an AI Coding IDE (Antigravity / Cursor / Claude Code / Windsurf)

Copy everything in the box below as the **first message / project instructions** in your AI coding tool. It is written so the agent builds incrementally, phase by phase, and never overclaims quantum results.

```
PROJECT: QFedAutoML — Quantum-Enhanced Federated AutoML Platform for
Privacy-Preserving Intelligent Systems

ROLE: You are acting as a senior full-stack + ML systems engineer building a
final-year B.Tech CS/AI&DS capstone project. Work incrementally, phase by
phase, and confirm each phase runs and is tested before starting the next.

NON-NEGOTIABLE ACCURACY RULES (apply to code, comments, docs, and any text
you generate):
1. Neural network / Transformer training always runs on CLASSICAL hardware
   (PyTorch, CPU/GPU). Quantum code NEVER trains the neural network itself.
2. Quantum computing is used ONLY for small combinatorial optimization
   subproblems: feature selection, hyperparameter search, client selection.
   Formulate these as QUBO/Ising problems.
3. Every quantum optimizer module MUST have a classical counterpart
   (random/grid/Bayesian search or simulated annealing) that runs in the
   same pipeline so results can be compared honestly. Never hardcode a
   "quantum wins" result — log both and let the data decide.
4. All quantum code runs on SIMULATORS by default (Qiskit Aer or
   PennyLane's default.qubit). Keep real-hardware backends (e.g., IBM
   Quantum) behind a feature flag, off by default, clearly documented as
   optional/experimental.
5. Keep qubit counts small (≤ 15–20 logical qubits) and circuit depth
   shallow — this must run on a laptop without real quantum hardware.
6. Clearly separate, in code comments and docs, three tiers per feature:
   [IMPLEMENTED] fully working now, [EXPERIMENTAL] runs but unvalidated,
   [FUTURE WORK] designed but not built. Never mark something IMPLEMENTED
   unless it actually runs end-to-end.
7. Do not fabricate benchmark numbers, papers, or datasets in comments or
   docs. If example numbers are needed for a placeholder, label them
   clearly as "PLACEHOLDER — replace after running experiments."
8. Federated learning must keep raw data on the client at all times. Only
   model weights/gradients (optionally DP-noised) cross the network.
9. Prefer well-known, stable libraries over exotic ones: PyTorch,
   scikit-learn, XGBoost, Flower (flwr), Qiskit, PennyLane, FastAPI,
   SQLAlchemy, PostgreSQL, React, Tailwind, Chart.js.
10. Every module needs at least one unit test before being considered done.
11. ERROR-HANDLING DISCIPLINE: after writing each file or small group of
    files, run it (or its test) immediately. Do not write the next file on
    top of unverified code. If a command fails, fix it before moving on —
    do not silently continue and leave a broken import for a "later phase"
    to discover. If you are not sure a library API call is correct, say so
    explicitly rather than guessing silently.
12. Assume the environment in docs/architecture.md Section 3 (Prerequisites)
    is already installed and verified. If you need an additional package,
    name it explicitly and add it to requirements.txt / package.json rather
    than assuming it's present.

TECH STACK:
- Backend: Python 3.11, FastAPI, SQLAlchemy + PostgreSQL, Alembic
- ML: PyTorch (Transformer + baseline models), scikit-learn, XGBoost
- Federated learning: Flower (flwr)
- Quantum: Qiskit (Aer simulator) and/or PennyLane (default.qubit)
- Privacy: Opacus (DP-SGD) or a custom DP noise module
- Explainability: SHAP, LIME
- Frontend: React + TypeScript + Tailwind CSS + Chart.js
- Auth: JWT (python-jose or PyJWT)
- Containerization: Docker + docker-compose
- CI: GitHub Actions

FOLDER STRUCTURE: Use exactly this structure (create empty __init__.py /
placeholder files first, then fill in):

qfedautoml/
  backend/{api,federated,quantum,automl,models,privacy,security,
           explainability,evaluation,database,monitoring,tests}/
  clients_simulation/
  frontend/src/{api,pages,components,store}/
  notebooks/
  docs/research/
  infra/{docker,ci}/

(See the full annotated tree in docs/architecture.md that you will generate
in Phase 1 — reuse the structure exactly as given to you.)

BUILD IN THIS ORDER (do not skip ahead):

PHASE 1 — Scaffolding & Docs
- Initialize repo, requirements.txt/pyproject.toml, docker-compose.yml.
- Create the full folder tree above with placeholder files.
- Write docs/architecture.md (condense the architecture given to you).
- Set up FastAPI skeleton (backend/main.py) with a working /system/health
  endpoint. Verify it runs.

PHASE 2 — Database & Auth
- Design and implement the schema: users, clients, datasets,
  training_rounds, model_versions, experiments, quantum_jobs, metrics,
  predictions, security_events (define columns, types, PK/FK).
- Implement Alembic migrations.
- Implement JWT auth: POST /auth/register, POST /auth/login.
- Add auth dependency usable by all protected routes.

PHASE 3 — Classical ML Baseline (Baseline 1)
- Pick one starter dataset (tabular, e.g. a public healthcare or finance
  classification dataset) and implement a centralized classical model
  (XGBoost or sklearn) as the first baseline. No FL, no quantum yet.
- Log accuracy/F1/ROC-AUC to the database via evaluation/metrics.py.

PHASE 4 — Federated Learning Core (Baseline 2)
- Implement backend/federated/server.py and client.py using Flower.
- Implement clients_simulation/run_client.py to spin up N simulated
  clients with Non-IID data partitions (clients_simulation/
  data_partitioner.py).
- Implement FedAvg first; verify convergence over multiple rounds and log
  communication rounds + cost to federated_metrics.py.
- Add POST /training/start, GET /training/{id}, GET /training/{id}/metrics.

PHASE 5 — Transformer Client Model (Baseline 3, optional/data-dependent)
- Only implement if a sequential dataset is chosen (time-series/sensor
  data). Implement backend/models/transformer_model.py as an optional
  local client model, selectable per experiment.
- Wire it into the federated client as an alternative to the classical
  model.

PHASE 6 — Classical AutoML Engine
- Implement dataset_profiler.py, feature_selector.py (classical methods:
  mutual information, RFE), model_selector.py, hpo_classical.py (grid,
  random, Bayesian via optuna or scikit-optimize).
- Add a leaderboard.py that ranks candidate configs by validation metric.

PHASE 7 — Quantum Optimization Engine (mark EXPERIMENTAL until validated)
- Implement qubo_builder.py: generic QUBO formulation utility.
- Implement feature_selection_qubo.py: binary variables x_i ∈ {0,1} per
  feature, objective balancing relevance vs redundancy, penalty terms for
  constraints (e.g., max feature count).
- Implement client_selection_qubo.py similarly for selecting a client
  subset given data quality/latency/capacity/reliability scores.
- Implement qaoa_optimizer.py using Qiskit Aer (or PennyLane) — small
  qubit counts, shallow depth, documented circuit diagram in code comments.
- Implement classical_fallback.py (simulated annealing or exhaustive
  search for small N) — this must run alongside every quantum job.
- Implement job_manager.py + POST /quantum/optimize, GET /quantum/jobs/{id}.
- Implement quantum_metrics.py: log qubits used, circuit depth, iterations,
  objective value, execution time, and the classical baseline's same
  metrics side by side.
- DO NOT claim quantum superiority in code comments or docs — only report
  measured numbers.

PHASE 8 — Bridge AutoML ↔ Quantum
- Implement hpo_quantum_bridge.py: for a chosen hyperparameter search
  space, offer both the classical HPO path (Phase 6) and the quantum QUBO
  path (Phase 7), run both, store both results in `experiments`.

PHASE 9 — Privacy & Security
- Implement differential_privacy.py (DP-SGD via Opacus or manual gradient
  clipping + noise), privacy_budget_tracker.py (epsilon accounting).
- Implement secure_aggregation.py (masking-based secure sum, or clearly
  document as [FUTURE WORK] if full secure aggregation isn't implemented
  in this phase — do not claim it if it's just FedAvg with TLS).
- Implement security/auth.py (already partly done in Phase 2), tls_config.py,
  threat_detection.py (basic outlier/poisoning detection on client
  updates), audit_log.py.
- Document the threat model explicitly: which of {eavesdropping, gradient
  leakage, model poisoning, Byzantine clients, membership inference,
  Sybil attacks, malicious server, malicious clients} are mitigated by
  which component, and which remain open/future work.

PHASE 10 — Explainability
- Implement shap_explainer.py, lime_explainer.py for the classical/
  tabular model; attention_visualizer.py for the Transformer path if used.
- Add POST /explain endpoint returning feature importances / attention
  weights + a human-readable summary via report_generator.py.
- Add a code comment / doc note on the limitation of treating attention
  weights as causal explanations.

PHASE 11 — Backend API completion & Model Registry
- Complete remaining routes: /clients/*, /models/*, POST /predict.
- Implement models/model_registry.py: versioning, save/load, promote to
  production.

PHASE 12 — Frontend Dashboard
- Build React pages: Dashboard, Clients, TrainingRounds, QuantumJobs,
  ModelRegistry, Explainability, Login.
- Use Chart.js for training curves, quantum objective-value convergence,
  and classical-vs-quantum comparison charts.
- Wire to backend via a typed API client.

PHASE 13 — Evaluation & Benchmarking
- Implement benchmark_runner.py to run all 4 baselines from the project
  spec (centralized classical / federated classical / federated
  Transformer / federated Transformer + quantum-enhanced optimization)
  on the same dataset/splits and produce a comparison table.
- Store results in docs/research/experiment_results.md with real numbers
  from actual runs only.

PHASE 14 — Containerization & CI
- Write Dockerfiles for backend, frontend, client simulator.
- Write docker-compose.yml to bring up backend + Postgres + N simulated
  clients + frontend with one command.
- Add GitHub Actions workflow: lint, unit tests, build.

PHASE 15 — Documentation
- Finalize docs/architecture.md, docs/api_reference.md, docs/setup_guide.md.
- Write docs/research/literature_review.md citing only real, verifiable
  papers you can confirm exist — if uncertain about a citation, flag it
  for the user to verify rather than inventing details.

AT THE END OF EACH PHASE:
- Run and show the test suite for that phase.
- Summarize what is now [IMPLEMENTED] vs [EXPERIMENTAL] vs [FUTURE WORK].
- Ask before proceeding to the next phase if any design decision is
  ambiguous (e.g., which public dataset to use, which cloud provider for
  deployment).

Start with PHASE 1 now.
```

---

## 4. How to Use This

1. Save/paste the architecture and file tree into your repo's `docs/architecture.md` so the AI IDE has it as persistent context.
2. Paste the boxed master prompt as the first instruction to your AI coding agent.
3. Let it complete Phase 1, verify the health endpoint runs, then proceed phase by phase — this keeps the agent from generating a huge, untestable codebase in one shot, and keeps quantum claims honest throughout, matching your accuracy rules.
