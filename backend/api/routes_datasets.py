"""
[IMPLEMENTED] Dataset Management REST APIs.
Upload custom CSV datasets, inspect active dataset metadata,
and reset to the built-in starter dataset.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

router = APIRouter(prefix="/datasets", tags=["Dataset Management"])

# Paths
_DATA_DIR = Path("backend/data")
_UPLOADS_DIR = _DATA_DIR / "uploads"
_ACTIVE_REGISTRY = _DATA_DIR / "active_dataset.json"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_active(info: dict[str, Any]) -> None:
    _ACTIVE_REGISTRY.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _read_active() -> dict[str, Any] | None:
    if _ACTIVE_REGISTRY.exists():
        try:
            return json.loads(_ACTIVE_REGISTRY.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _clear_active() -> None:
    if _ACTIVE_REGISTRY.exists():
        _ACTIVE_REGISTRY.unlink()


def _validate_and_profile(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Validate CSV and return profile metadata."""
    if target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dataset must have at least 2 feature columns (excluding target)."
        )

    if len(df) < 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset too small: {len(df)} rows. Minimum 50 rows required."
        )

    # Auto-drop non-numeric feature columns
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        df = df.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    if len(feature_cols) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="After dropping non-numeric columns, fewer than 2 features remain."
        )

    # Validate target column — convert string labels to int if needed
    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        classes = sorted(y.unique())
        label_map = {lbl: idx for idx, lbl in enumerate(classes)}
        df[target_col] = y.map(label_map)

    unique_classes = sorted(df[target_col].dropna().unique().astype(int).tolist())
    num_classes = len(unique_classes)

    # Drop rows with NaN in target
    df = df.dropna(subset=[target_col])

    # Fill NaN features with column mean
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    class_counts = df[target_col].value_counts().sort_index().to_dict()

    return {
        "df": df,
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "num_samples": len(df),
        "num_classes": num_classes,
        "classes": unique_classes,
        "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
        "dropped_non_numeric": non_numeric,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file with header row"),
    target_column: str = Form(..., description="Column name to use as classification label"),
):
    """
    Upload a custom CSV dataset and activate it platform-wide.
    All subsequent AutoML, SHAP/LIME, Predict, and Federated training calls
    will use this dataset instead of the built-in starter dataset.
    """
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .csv files are supported."
        )

    # Size guard — read up to 50 MB
    MAX_BYTES = 50 * 1024 * 1024
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 50 MB limit."
        )

    # Parse CSV
    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV: {exc}"
        ) from exc

    # Validate & profile
    profile = _validate_and_profile(df, target_column.strip())
    cleaned_df = profile["df"]

    # Save cleaned CSV to uploads directory
    safe_name = Path(file.filename).stem.replace(" ", "_")[:60]
    save_path = _UPLOADS_DIR / f"{safe_name}.csv"
    cleaned_df.to_csv(save_path, index=False)

    # Register as active dataset
    registry = {
        "source": "user_upload",
        "filename": file.filename,
        "saved_path": str(save_path),
        "target_column": target_column.strip(),
        "feature_columns": profile["feature_cols"],
        "num_samples": profile["num_samples"],
        "num_features": profile["num_features"],
        "num_classes": profile["num_classes"],
        "classes": profile["classes"],
        "class_distribution": profile["class_distribution"],
        "dropped_non_numeric": profile["dropped_non_numeric"],
    }
    _write_active(registry)

    return {
        "status": "activated",
        "message": f"Dataset '{file.filename}' uploaded and activated successfully.",
        "dataset": {k: v for k, v in registry.items() if k != "df"},
    }


@router.get("/headers", status_code=status.HTTP_200_OK)
async def peek_csv_headers(
    file: UploadFile = File(..., description="CSV file to inspect column headers"),
):
    """
    Preview the column names of an uploaded CSV without activating it.
    Use this to populate the target column selector in the UI.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .csv files are supported."
        )
    content = await file.read(4096)  # read first 4 KB — enough for headers
    try:
        from io import BytesIO
        df_head = pd.read_csv(BytesIO(content), nrows=5)
        return {
            "columns": list(df_head.columns),
            "preview_rows": df_head.fillna("").to_dict(orient="records")
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV headers: {exc}"
        ) from exc


@router.get("/active", status_code=status.HTTP_200_OK)
def get_active_dataset():
    """
    Return metadata of the currently active dataset.
    Returns the built-in breast cancer profile if no user dataset is uploaded.
    """
    info = _read_active()
    if info:
        return info

    # Describe the built-in dataset
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer()
    return {
        "source": "builtin",
        "filename": "Wisconsin Diagnostic Breast Cancer (sklearn)",
        "target_column": "diagnosis_malignant",
        "feature_columns": [str(f) for f in bc.feature_names],
        "num_samples": bc.data.shape[0],
        "num_features": bc.data.shape[1],
        "num_classes": 2,
        "classes": [0, 1],
        "class_distribution": {
            int(k): int(v)
            for k, v in zip(*np.unique(bc.target, return_counts=True))
        },
        "dropped_non_numeric": [],
    }


@router.delete("/reset", status_code=status.HTTP_200_OK)
def reset_to_builtin():
    """
    Remove any uploaded dataset and revert the platform to the built-in
    Wisconsin Diagnostic Breast Cancer starter dataset.
    """
    _clear_active()
    # Clean uploads directory (keep .gitkeep)
    for f in _UPLOADS_DIR.iterdir():
        if f.name != ".gitkeep" and f.suffix == ".csv":
            f.unlink(missing_ok=True)
    return {
        "status": "reset",
        "message": "Reverted to built-in Wisconsin Diagnostic Breast Cancer dataset."
    }
