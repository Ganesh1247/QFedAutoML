# QFedAutoML: Quantum-Enhanced Federated AutoML Platform

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Flower](https://img.shields.io/badge/Flower-FL-pink.svg)](https://flower.ai)
[![Qiskit](https://img.shields.io/badge/Qiskit-Aer%20QAOA-613399.svg)](https://qiskit.org)
[![Tests](https://img.shields.io/badge/Tests-56%2F56%20Passing-emerald.svg)](backend/tests)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff%20Clean-cyan.svg)](pyproject.toml)

**QFedAutoML** is a comprehensive, production-grade, privacy-preserving intelligent systems platform combining classical Federated Learning (Flower / PyTorch), automated machine learning (AutoML / Optuna), and quantum combinatorial optimization (QAOA / QUBO via Qiskit Aer simulators) for edge tabular and temporal sensor intelligence.

---

## 🏛️ Core Platform Architecture & Non-Negotiable Rules

1. **Strict Classical Neural & Model Training**: Neural networks (`TimeSeriesTransformerNN`, `TabularPyTorchNN`) and ML models (`XGBoost`, `RandomForest`) execute **strictly on classical CPU/GPU**. Quantum circuits NEVER train neural weights directly.
2. **Quantum Combinatorial Optimization (Simulators)**: Quantum computing is used strictly for small combinatorial optimization subproblems (feature selection, hyperparameter search, client selection formulated as QUBO/Ising problems, $\le 16\text{--}20$ logical qubits on Qiskit Aer).
3. **Honest Classical Counterparts**: Every quantum optimizer is paired with a classical solver (Simulated Annealing, Bayesian TPE) evaluated in the same pipeline without hardcoded advantage assertions.
4. **Data Sovereignty & Privacy**: Raw edge data never leaves local storage. Only gradient/weight updates cross the network under $(\epsilon, \delta)$ Differential Privacy (DP-SGD) and Byzantine threat filtering.

---

## 📊 Multi-Paradigm Comparative Benchmark Matrix

| Paradigm Baseline | Model Architecture | Privacy Guarantee | Accuracy | ROC-AUC | Communication Overhead | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1: Centralized ML** | Centralized XGBoost | None (Data centralized) | **0.9650** | **0.9900** | 0.00 MB | `[IMPLEMENTED]` |
| **Baseline 2: Classical FL** | Decentralized Tabular FedAvg | Data Sovereignty (No DP noise) | **0.9580** | **0.9650** | 11.25 MB | `[IMPLEMENTED]` |
| **Baseline 3: Federated Transformer** | TimeSeriesTransformerNN (4 Heads) | Temporal Data Sovereignty | **0.9600** | **0.9800** | 14.70 MB | `[IMPLEMENTED]` |
| **Baseline 4: Proposed QFedAutoML** | QAOA QUBO (6 Qubits) + Optuna HPO | $(\epsilon=1.42, \delta=10^{-5})$ + Byzantine Filter | **0.9780** | **0.9940** | **7.65 MB** (-32.0%) | `[IMPLEMENTED]` |

---

## 💻 Frontend Dashboard & Studios (React + TypeScript + Tailwind + Chart.js)

The platform includes a responsive dark-mode glassmorphic dashboard with 7 studios:
- **Executive Overview**: Real-time KPI scorecard, 4-baseline comparative matrix, edge nodes telemetry, and compliance checklist.
- **Federated Studio**: Live federated round runner with accuracy/loss convergence curves and network communication volume meters.
- **AutoML & Quantum Studio**: Classical vs QAOA optimizer selector, QUBO Hamiltonian mathematical formulations, and live leaderboard.
- **Model Registry & Staging**: Immutable model artifact tracker, validation scorecards, hyperparameter inspector, and production staging.
- **Explainability Center**: Global TreeSHAP bar charts, local LIME surrogate attribution, 4-head Transformer attention heatmaps, and downloadable HTML reports.
- **Privacy & Security Center**: Moments Accountant $(\epsilon, \delta)$ budget progress bars, DP noise calibrator, and Byzantine attack audit logs.
- **Inference Lab**: Low-latency ($< 15$ms) live diagnostic inference tester with confidence probability meters and database logging.

---

## 🚀 Quickstart & Local Execution

### 1. Bare Metal Setup

```bash
# Clone repository
git clone https://github.com/Ganesh1247/QFedAutoML.git
cd QFedAutoML

# Install backend dependencies
pip install -r requirements.txt

# Run all 56 unit and integration tests
pytest

# Launch FastAPI backend (port 8000)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```

- **Frontend Dashboard**: Open [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Docker Compose Full-Stack Deployment

```bash
docker-compose up --build -d
```

---

## 📚 Documentation
- [System Architecture](docs/architecture.md)
- [REST API Reference](docs/api_reference.md)
- [Developer & Setup Guide](docs/setup_guide.md)
- [Literature Review](docs/research/literature_review.md)
- [Capstone Experiment Results](docs/research/experiment_results.md)