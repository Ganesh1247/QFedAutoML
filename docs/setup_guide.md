# QFedAutoML Developer & Deployment Guide

## 1. Prerequisites
- **Python**: 3.11, 3.12, or 3.13
- **Node.js**: 18+ LTS or 20+
- **Docker & Docker Compose**: (Optional for containerized full-stack deployment)

---

## 2. Quickstart: Bare Metal Localhost Setup

### 2.1 Backend Setup
```bash
# Clone the repository
git clone https://github.com/Ganesh1247/QFedAutoML.git
cd QFedAutoML

# Install Python dependencies
pip install -r requirements.txt

# Run the complete test suite (56 tests)
pytest

# Launch FastAPI backend on port 8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2.2 Frontend Setup
```bash
# In a separate terminal:
cd frontend

# Install npm dependencies
npm install

# Start Vite live dev server on port 3000
npm run dev
```

- **Frontend Dashboard**: Open [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 3. Containerized Deployment with Docker Compose

To launch the full production environment (PostgreSQL 16, FastAPI backend, and Nginx frontend):

```bash
docker-compose up --build -d
```

- **Frontend App**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **PostgreSQL Database**: `localhost:5432` (`qfed_user` / `qfed_password` / `qfed_db`)

---

## 4. Running Benchmark Suite

To execute the 4-baseline comparative benchmark suite:
```bash
pytest backend/tests/unit/test_benchmarks.py -v
```
