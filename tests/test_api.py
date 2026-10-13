"""Automated API endpoint tests for FastAPI server."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.api import app, initialize_model_and_xai
from backend.xai import SAMPLE_PATIENTS

# Initialize model before tests
initialize_model_and_xai()
client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["xai_ready"] is True


def test_metadata_endpoint():
    res = client.get("/api/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "Glucose" in data["features"]
    assert "Glucose" in data["metadata"]


def test_sample_patients_endpoint():
    res = client.get("/api/sample-patients")
    assert res.status_code == 200
    patients = res.json()
    assert len(patients) >= 3


def test_model_stats_endpoint():
    res = client.get("/api/model-stats")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "feature_importance" in data
    assert "base_population_risk" in data


def test_predict_endpoint():
    patient_payload = SAMPLE_PATIENTS[0]["data"]
    res = client.post("/api/predict", json=patient_payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "feature_contributions" in data
    assert "probability" in data["prediction"]
    assert "risk_tier" in data["prediction"]
    assert len(data["feature_contributions"]) == 8


def test_what_if_endpoint():
    payload = {
        "baseline": SAMPLE_PATIENTS[2]["data"],
        "modified": {**SAMPLE_PATIENTS[2]["data"], "Glucose": 100.0, "BMI": 23.5}
    }
    res = client.post("/api/what-if", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "percentage_risk_reduction" in data
    assert data["is_improved"] is True


def test_batch_predict_endpoint():
    with open("sample_batch_patients.csv", "rb") as f:
        res = client.post(
            "/api/batch-predict",
            files={"file": ("sample_batch_patients.csv", f, "text/csv")}
        )
    assert res.status_code == 200
    data = res.json()
    assert data["total_analyzed"] == 25
    assert "risk_breakdown" in data
    assert len(data["patients"]) == 25


def test_static_index_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "Explainable Diabetes Risk Prediction Dashboard" in res.text


if __name__ == "__main__":
    print("Running API tests...")
    test_health_endpoint()
    print("✓ test_health_endpoint passed")
    test_metadata_endpoint()
    print("✓ test_metadata_endpoint passed")
    test_sample_patients_endpoint()
    print("✓ test_sample_patients_endpoint passed")
    test_model_stats_endpoint()
    print("✓ test_model_stats_endpoint passed")
    test_predict_endpoint()
    print("✓ test_predict_endpoint passed")
    test_what_if_endpoint()
    print("✓ test_what_if_endpoint passed")
    test_batch_predict_endpoint()
    print("✓ test_batch_predict_endpoint passed")
    test_static_index_endpoint()
    print("✓ test_static_index_endpoint passed")
    print("All API tests passed successfully!")
