"""
SkPro Probabilistic ML API
FastAPI backend for probabilistic regression with uncertainty quantification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import time
import json

from models import get_available_models, fit_model, predict_with_uncertainty

app = FastAPI(
    title="SkPro Probabilistic ML API",
    description="Probabilistic supervised machine learning with uncertainty quantification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for uploaded datasets (simple approach for demo)
dataset_store: Dict[str, pd.DataFrame] = {}

# ─── Request/Response Models ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    model_type: str
    test_size: float = 0.2

class CompareRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    model_types: List[str]
    test_size: float = 0.2

class PredictionPoint(BaseModel):
    index: int
    x: float          # index in test set
    y_true: float
    y_pred: float
    y_lower: float
    y_upper: float

class ModelSummary(BaseModel):
    model_type: str
    r2_score: float
    mae: float
    rmse: float
    training_time_seconds: float
    n_train_samples: int
    n_test_samples: int
    n_features: int
    parameters: Dict[str, Any]

class PredictResponse(BaseModel):
    predictions: List[PredictionPoint]
    model_summary: ModelSummary

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SkPro Probabilistic ML API is running!", "docs": "/docs"}


@app.get("/models")
def get_models():
    """
    GET /models
    Returns list of available probabilistic model types with descriptions.
    """
    return {"models": get_available_models()}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    POST /upload
    Upload a CSV dataset. Returns dataset_id, columns, shape, and preview.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Basic validation
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Dataset too small. Need at least 10 rows.")

    # Store with a simple ID
    dataset_id = file.filename.replace(".csv", "").replace(" ", "_")
    dataset_store[dataset_id] = df

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Preview: first 5 rows as dict
    preview = df.head(5).fillna("NaN").to_dict(orient="records")

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "all_columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "preview": preview,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    POST /predict
    Fit a probabilistic model and return predictions with uncertainty intervals.
    """
    if req.dataset_id not in dataset_store:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found. Please upload first.")

    df = dataset_store[req.dataset_id].copy()

    # Validate columns
    all_cols = req.feature_columns + [req.target_column]
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")

    # Drop rows with missing values in selected columns
    df = df[all_cols].dropna()

    if len(df) < 10:
        raise HTTPException(status_code=400, detail="After removing missing values, dataset is too small (< 10 rows).")

    # Check numeric
    non_numeric = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"Non-numeric columns selected: {non_numeric}. Please select only numeric columns.")

    X = df[req.feature_columns].values
    y = df[req.target_column].values

    # Train/test split
    n = len(X)
    n_test = max(2, int(n * req.test_size))
    n_train = n - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Fit and predict
    start = time.time()
    result = fit_model(req.model_type, X_train, y_train, X_test, y_test)
    elapsed = time.time() - start

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Build prediction points
    predictions = []
    for i in range(len(y_test)):
        predictions.append(PredictionPoint(
            index=i,
            x=float(i),
            y_true=float(y_test[i]),
            y_pred=float(result["y_pred"][i]),
            y_lower=float(result["y_lower"][i]),
            y_upper=float(result["y_upper"][i]),
        ))

    summary = ModelSummary(
        model_type=req.model_type,
        r2_score=round(result["r2"], 4),
        mae=round(result["mae"], 4),
        rmse=round(result["rmse"], 4),
        training_time_seconds=round(elapsed, 3),
        n_train_samples=n_train,
        n_test_samples=n_test,
        n_features=len(req.feature_columns),
        parameters=result.get("parameters", {}),
    )

    return PredictResponse(predictions=predictions, model_summary=summary)


@app.post("/compare")
def compare_models(req: CompareRequest):
    """
    POST /compare
    Compare 2+ models side-by-side on the same dataset.
    """
    if req.dataset_id not in dataset_store:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")

    if len(req.model_types) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 model types for comparison.")

    df = dataset_store[req.dataset_id].copy()
    all_cols = req.feature_columns + [req.target_column]
    df = df[all_cols].dropna()

    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Dataset too small after removing missing values.")

    X = df[req.feature_columns].values
    y = df[req.target_column].values

    n = len(X)
    n_test = max(2, int(n * req.test_size))
    n_train = n - n_test
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    results = []
    for model_type in req.model_types:
        start = time.time()
        result = fit_model(model_type, X_train, y_train, X_test, y_test)
        elapsed = time.time() - start

        if "error" in result:
            results.append({"model_type": model_type, "error": result["error"]})
            continue

        predictions = [
            {
                "index": i,
                "x": float(i),
                "y_true": float(y_test[i]),
                "y_pred": float(result["y_pred"][i]),
                "y_lower": float(result["y_lower"][i]),
                "y_upper": float(result["y_upper"][i]),
            }
            for i in range(len(y_test))
        ]

        results.append({
            "model_type": model_type,
            "predictions": predictions,
            "summary": {
                "r2_score": round(result["r2"], 4),
                "mae": round(result["mae"], 4),
                "rmse": round(result["rmse"], 4),
                "training_time_seconds": round(elapsed, 3),
                "parameters": result.get("parameters", {}),
            }
        })

    return {"comparison": results, "n_test_samples": n_test}
