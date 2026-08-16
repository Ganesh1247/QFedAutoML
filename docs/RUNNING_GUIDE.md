# QFedAutoML — How to Run Locally & Use Your Own Dataset

## Prerequisites

Install these once on your machine before anything else:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.11 or 3.13 | https://python.org |
| Node.js | 18+ | https://nodejs.org |
| Git | any | https://git-scm.com |

---

## Step 1 — Clone / Open the Project

```powershell
# If you haven't cloned yet:
git clone <your-repo-url>
cd QFedAutoML

# Or just open the existing folder:
cd "C:\Users\koila\OneDrive\Desktop\QFedAutoML\QFedAutoML"
```

---

## Step 2 — Install Python Dependencies

```powershell
# From the project root (QFedAutoML/)
pip install -r requirements.txt
```

> This installs FastAPI, PyTorch, Qiskit Aer, scikit-learn, XGBoost, Flower, SHAP, LIME, Optuna, etc.

---

## Step 3 — Configure Environment (Optional)

The app works out of the box with SQLite. No PostgreSQL needed locally.

```powershell
# Copy the example env file
copy .env.example .env
```

The default `.env` already sets:
```
DATABASE_URL=sqlite:///./qfedautoml.db   ← local file, no setup needed
QUANTUM_SIMULATOR_BACKEND=qiskit_aer     ← runs on your CPU
```

---

## Step 4 — Start the Backend (Terminal 1)

```powershell
cd "C:\Users\koila\OneDrive\Desktop\QFedAutoML\QFedAutoML"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

✅ You should see:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

## Step 5 — Start the Frontend (Terminal 2)

Open a **second** terminal window:

```powershell
cd "C:\Users\koila\OneDrive\Desktop\QFedAutoML\QFedAutoML\frontend"
npm install        # first time only
npm run dev
```

✅ You should see:
```
  VITE v5.x  ready in 600 ms
  ➜  Local:   http://localhost:3000/
```

---

## Step 6 — Open the Dashboard

Go to: **http://localhost:3000**

The Vite dev server automatically proxies all `/api/v1/...` calls to `http://localhost:8000`.

---

## Where to Put Your Own Dataset & Train Locally

### Current Behavior (Built-in Dataset)

By default the system uses the **Wisconsin Diagnostic Breast Cancer** dataset from scikit-learn (`sklearn.datasets.load_breast_cancer`). It loads automatically — no file needed.

**Code location:** [`backend/automl/preprocessing.py`](file:///c:/Users/koila/OneDrive/Desktop/QFedAutoML/QFedAutoML/backend/automl/preprocessing.py) → `load_starter_tabular_dataset()`

---

### How to Use Your Own CSV Dataset

#### 1. Drop your CSV file here:

```
QFedAutoML/
└── backend/
    └── data/           ← create this folder
        └── my_dataset.csv
```

#### 2. Edit `backend/automl/preprocessing.py`

Add your loader after line 63 (the `else: raise ValueError` block):

```python
elif dataset_name == "custom":
    import pandas as pd
    df = pd.read_csv("backend/data/my_dataset.csv")

    # Change "target" to your actual label column name
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    feature_names = feature_cols
    target_name = target_col
```

#### 3. Tell the system to use it

Change the default argument in the same function signature from:
```python
def load_starter_tabular_dataset(
    dataset_name: str = "breast_cancer",   # ← change to "custom"
```
to:
```python
def load_starter_tabular_dataset(
    dataset_name: str = "custom",
```

#### 4. Restart the backend

```powershell
# Stop uvicorn (Ctrl+C), then:
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Your dataset will now be used everywhere — AutoML, SHAP/LIME explainability, prediction, and federated training.

---

### CSV File Format Requirements

| Requirement | Detail |
|-------------|--------|
| Format | `.csv` with header row |
| Target column | Integer labels (0, 1, 2, ...) |
| Feature columns | Numeric values only (float/int) |
| Missing values | Remove or impute before loading |
| Min size | ≥ 100 rows recommended |

Example structure:
```csv
mean_radius,mean_texture,...,worst_fractal_dimension,target
17.99,10.38,...,0.1189,1
13.54,14.36,...,0.0726,0
```

---

## Quick Reference — What Each URL Does

| URL | What It Is |
|-----|-----------|
| http://localhost:3000 | Main Dashboard |
| http://localhost:3000 → Overview tab | System KPIs & client nodes |
| http://localhost:3000 → Federated tab | Run FL training rounds |
| http://localhost:3000 → AutoML tab | Run QAOA feature selection + HPO |
| http://localhost:3000 → Model Registry | Promote models to production |
| http://localhost:3000 → Predict Lab | Run live inference |
| http://localhost:3000 → Explainability | SHAP/LIME charts + Export HTML report |
| http://localhost:3000 → Privacy & Security | DP budget & Byzantine filter audit |
| http://127.0.0.1:8000/docs | Swagger API docs (all endpoints) |
| http://127.0.0.1:8000/redoc | ReDoc API reference |

---

## Run Tests (Optional)

```powershell
# From project root
python -m pytest backend/tests/ -v
# Expected: 56 passed
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 8000 already in use | `netstat -ano \| findstr :8000` → kill the PID |
| Port 3000 already in use | `npm run dev -- --port 3001` |
| ModuleNotFoundError | `pip install -r requirements.txt` again |
| Database errors | Delete `qfedautoml.db` and restart backend (it recreates itself) |
| Frontend shows blank | Hard-refresh `Ctrl+Shift+R` or check backend is running |
