"""FastAPI REST API Service for Explainable Diabetes Risk Prediction Dashboard.

Exposes endpoints for:
- Single-patient prediction with SHAP contribution analysis
- Model performance metrics & ROC statistics
- What-If counterfactual scenario evaluation
- Batch screening with CSV upload
- Sample clinical archetypes & reference metadata
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.model import DiabetesRiskModel
from backend.preprocessor import (
    CLINICAL_METADATA,
    FEATURE_NAMES,
    load_raw_dataset,
)
from backend.xai import (
    SAMPLE_PATIENTS,
    ExplainableAIEngine,
)

# Global model and XAI engine instances
model_instance: Optional[DiabetesRiskModel] = None
xai_instance: Optional[ExplainableAIEngine] = None
ARTIFACT_DIR = "backend/artifacts"


def initialize_model_and_xai():
    """Load existing model artifacts or train a new one if not present."""
    global model_instance, xai_instance
    if os.path.exists(os.path.join(ARTIFACT_DIR, "xgb_model.joblib")):
        print("Loading trained model from artifacts...")
        model_instance = DiabetesRiskModel.load(ARTIFACT_DIR)
    else:
        print("Training model from diabetes.csv...")
        model_instance = DiabetesRiskModel()
        model_instance.train_and_evaluate("diabetes.csv")
        model_instance.save(ARTIFACT_DIR)
        print("Model trained and saved to artifacts.")

    xai_instance = ExplainableAIEngine(model_instance)
    print("Explainable AI engine initialized.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_model_and_xai()
    yield


app = FastAPI(
    title="Explainable Diabetes Risk Prediction Dashboard",
    description="Department of Artificial Intelligence and Data Science - Batch 10",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInputSchema(BaseModel):
    Glucose: float = Field(..., ge=0, le=500, description="Plasma glucose (mg/dL)")
    BloodPressure: float = Field(
        ..., ge=0, le=250, description="Diastolic blood pressure (mmHg)"
    )
    BMI: float = Field(..., ge=0, le=100, description="Body Mass Index (kg/m²)")
    Age: float = Field(..., ge=1, le=120, description="Age in years")
    Insulin: float = Field(
        ..., ge=0, le=1000, description="2-hour serum insulin (μU/mL)"
    )
    SkinThickness: float = Field(
        ..., ge=0, le=150, description="Triceps skinfold thickness (mm)"
    )
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0.0, le=5.0, description="Diabetes pedigree function score"
    )
    Pregnancies: float = Field(..., ge=0, le=30, description="Number of pregnancies")


class WhatIfRequestSchema(BaseModel):
    baseline: PatientInputSchema
    modified: PatientInputSchema


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None and model_instance.is_trained,
        "xai_ready": xai_instance is not None,
    }


@app.get("/api/metadata")
def get_metadata():
    """Return clinical reference parameters and feature definitions."""
    return {
        "features": FEATURE_NAMES,
        "metadata": CLINICAL_METADATA,
    }


@app.get("/api/sample-patients")
def get_sample_patients():
    """Return preset patient profiles for clinical demonstrations."""
    return SAMPLE_PATIENTS


@app.get("/api/model-stats")
def get_model_stats():
    """Return comprehensive model validation metrics, ROC curve, confusion matrix,

    and global feature importance rankings.
    """
    if not model_instance:
        raise HTTPException(status_code=500, detail="Model is not initialized.")
    return {
        "metrics": model_instance.metrics,
        "feature_importance": model_instance.feature_importance,
        "base_population_risk": round(xai_instance.base_probability * 100, 1),
    }


@app.post("/api/predict")
def predict_and_explain(patient: PatientInputSchema):
    """Predict diabetes risk and compute exact Shapley feature contribution analysis."""
    if not xai_instance:
        raise HTTPException(status_code=500, detail="XAI Engine is not initialized.")

    patient_dict = patient.model_dump()
    result = xai_instance.explain_patient(patient_dict)
    return result


@app.post("/api/what-if")
def simulate_counterfactual(payload: WhatIfRequestSchema):
    """Simulate targeted lifestyle/clinical interventions and observe real-time risk reduction."""
    if not xai_instance:
        raise HTTPException(status_code=500, detail="XAI Engine is not initialized.")

    baseline_dict = payload.baseline.model_dump()
    modified_dict = payload.modified.model_dump()
    result = xai_instance.simulate_what_if(baseline_dict, modified_dict)
    return result


@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Upload a CSV dataset of patients to receive cohort screening results

    and primary risk drivers for each patient.
    """
    if not xai_instance:
        raise HTTPException(status_code=500, detail="XAI Engine is not initialized.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read CSV file: {str(e)}"
        )

    # Check for required features
    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required feature columns: {missing_cols}",
        )

    results = []
    risk_counts = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}

    # Evaluate each patient row (cap at 500 rows for real-time responsiveness)
    max_rows = min(len(df), 500)
    for idx in range(max_rows):
        row_dict = df.iloc[idx][FEATURE_NAMES].to_dict()
        exp = xai_instance.explain_patient(row_dict)
        pred = exp["prediction"]
        top_driver = (
            exp["top_risk_factors"][0]["display_name"]
            if exp["top_risk_factors"]
            else "None"
        )

        risk_counts[pred["risk_tier"]] += 1
        results.append(
            {
                "patient_id": f"PT-{idx+1:04d}",
                "glucose": float(row_dict.get("Glucose", 0)),
                "bmi": float(row_dict.get("BMI", 0)),
                "age": int(row_dict.get("Age", 0)),
                "blood_pressure": float(row_dict.get("BloodPressure", 0)),
                "probability": pred["probability_percent"],
                "risk_tier": pred["risk_tier"],
                "risk_color": pred["risk_color"],
                "primary_driver": top_driver,
            }
        )

    return {
        "total_analyzed": len(results),
        "risk_breakdown": risk_counts,
        "patients": results,
    }


# Mount static frontend directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"message": "Static frontend not created yet. Please create static/index.html."}
    )
