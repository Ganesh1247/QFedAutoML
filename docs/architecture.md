# QFedAutoML System Architecture

## 1. System Overview & Core Principles

QFedAutoML is a **classical federated-learning + AutoML platform with an isolated quantum-combinatorial optimization engine**.

### 1.1 Non-Negotiable Accuracy Principles
1. **Classical Model Training**: Neural networks and tabular models (PyTorch, scikit-learn, XGBoost) always run on classical CPU/GPU hardware. Quantum computing NEVER trains the neural network weights.
2. **Quantum Combinatorial Scope**: Quantum computing is utilized solely for small combinatorial subproblems (feature selection, hyperparameter search, client selection) formulated as Quadratic Unconstrained Binary Optimization (QUBO) or Ising spin-glass problems.
3. **Paired Classical Baselines**: Every quantum optimization module is accompanied by an equivalent classical solver (simulated annealing, random/grid search, Bayesian optimization via Optuna) running in the identical pipeline. Results are logged side-by-side without hardcoded superiority claims.
4. **Simulator-First Execution**: All quantum algorithms execute on local simulators (Qiskit Aer, PennyLane `default.qubit`) by default. Real quantum hardware execution remains an optional, experimental flag.
5. **Shallow Circuits & Limited Qubits**: Logical qubit counts are capped at $\le 16-20$ qubits with shallow circuit depth ($p \le 3$ QAOA steps) to guarantee smooth local simulation on developer workstations.
6. **Data Locality & Privacy**: Raw training data strictly resides on the client devices. Only model parameters, gradients, or DP-noised updates cross the network boundary.
7. **Three-Tier Implementation Markers**:
   - `[IMPLEMENTED]`: Fully functional, tested end-to-end.
   - `[EXPERIMENTAL]`: Functional code running on simulators/mock environments, awaiting empirical benchmarking.
   - `[FUTURE WORK]`: Architectural design and interfaces outlined, implementation deferred.

---

## 2. Layered Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 6: PRESENTATION                                                      │
│   React 18 + Tailwind Dashboard  |  REST Consumers  |  CLI                 │
└───────────────────────────────────────────────────────────────────────────┘
                                   │  HTTPS / JWT
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 5: API GATEWAY                                                       │
│   FastAPI  →  Auth (JWT)  →  Rate Limiting  →  Request Routing             │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 4: ORCHESTRATION                                                     │
│   Federated Learning Controller  |  AutoML Controller  |  Experiment Mgr   │
└───────────────────────────────────────────────────────────────────────────┘
              │                    │                       │
              ▼                    ▼                       ▼
┌───────────────────┐  ┌────────────────────────┐  ┌───────────────────────┐
│ Layer 3a:          │  │ Layer 3b:               │  │ Layer 3c:              │
│ FEDERATED ENGINE   │  │ QUANTUM OPT. ENGINE     │  │ AUTOML ENGINE          │
│ (Flower-based)     │  │ (Qiskit/PennyLane, sim) │  │ (Optuna + Search)      │
│ - Client Registry  │  │ - QUBO Builder          │  │ - Dataset Profiler     │
│ - Client Selection │  │ - QAOA / VQE Optimizer  │  │ - Feature Selector     │
│ - FedAvg / FedProx │  │ - Classical Fallback    │  │ - Model Selector       │
│ - Secure Aggregation  │ - Job Manager          │  │ - Leaderboard          │
└───────────────────┘  └────────────────────────┘  └───────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 2: CLIENT / EDGE LAYER (Simulated processes or edge nodes)           │
│   ┌──────────────┬──────────────┬──────────────┐                          │
│   │  Client A     │  Client B     │  Client N    │                          │
│   │  Local Data   │  Local Data   │  Local Data  │  ← Raw data never leaves │
│   │  Preprocess   │  Preprocess   │  Preprocess  │                          │
│   │  PyTorch/ML   │  PyTorch/ML   │  PyTorch/ML  │                          │
│   │  DP Noise     │  DP Noise     │  DP Noise    │                          │
│   └──────────────┴──────────────┴──────────────┘                          │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
┌───────────────────────────────────────────────────────────────────────────┐
│ Layer 1: PHYSICAL / LOGICAL DATA SOURCES (Healthcare, Finance, IoT)        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema Overview

The relational persistence layer (PostgreSQL + SQLAlchemy) encompasses the following core tables:
- `users`: Authentication credentials and role-based access.
- `clients`: Registered edge clients, device capabilities, and availability stats.
- `datasets`: Metadata and feature specifications of local datasets.
- `training_rounds`: Federated training rounds, selected clients, aggregated losses, and convergence metrics.
- `model_versions`: Serialized global model artifacts, parameter checkpoints, and metadata.
- `experiments`: High-level benchmark/experiment orchestrations linking FL, AutoML, and Quantum runs.
- `quantum_jobs`: Quantum optimization job parameters (qubit count, circuit depth, QAOA energy, classical comparison).
- `metrics`: Fine-grained metric records across classical ML, federated rounds, and quantum executions.
- `security_events`: Audit logs for threat detection, authentication anomalies, and poisoning flags.
