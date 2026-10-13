"""Automated pipeline tests for diabetes preprocessing, model, and XAI engine."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from backend.preprocessor import ClinicalDataPreprocessor, load_raw_dataset, FEATURE_NAMES
from backend.model import DiabetesRiskModel
from backend.xai import ExplainableAIEngine, SAMPLE_PATIENTS


def test_biological_zero_handling():
    """Verify that zero values in Glucose, Blood Pressure, BMI, Insulin, etc. are imputed."""
    preprocessor = ClinicalDataPreprocessor()
    df_raw = pd.DataFrame([{
        "Pregnancies": 0,  # 0 is valid for pregnancies
        "Glucose": 0,      # 0 is biologically invalid
        "BloodPressure": 0,# 0 is biologically invalid
        "SkinThickness": 0,# 0 is biologically invalid
        "Insulin": 0,      # 0 is biologically invalid
        "BMI": 0.0,        # 0 is biologically invalid
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }])
    df_clean = preprocessor.clean_raw_dataframe(df_raw)
    assert np.isnan(df_clean.loc[0, "Glucose"]), "Glucose 0 must be converted to NaN"
    assert np.isnan(df_clean.loc[0, "BloodPressure"]), "Blood Pressure 0 must be converted to NaN"
    assert np.isnan(df_clean.loc[0, "BMI"]), "BMI 0 must be converted to NaN"
    assert df_clean.loc[0, "Pregnancies"] == 0, "Pregnancies 0 must be preserved"


def test_model_training_and_metrics():
    """Verify that model achieves target accuracy and ROC-AUC on benchmark dataset."""
    model = DiabetesRiskModel()
    metrics = model.train_and_evaluate("diabetes.csv")
    
    assert metrics["accuracy"] >= 0.70, f"Accuracy {metrics['accuracy']} should be >= 0.70"
    assert metrics["roc_auc"] >= 0.80, f"ROC-AUC {metrics['roc_auc']} should be >= 0.80"
    assert "confusion_matrix" in metrics
    assert len(model.feature_importance) == len(FEATURE_NAMES)


def test_shap_additivity():
    """Verify mathematical exactness of SHAP additivity theorem."""
    model = DiabetesRiskModel.load("backend/artifacts")
    xai = ExplainableAIEngine(model)

    test_patient = SAMPLE_PATIENTS[0]["data"]
    exp = xai.explain_patient(test_patient)

    # Base log-odds + sum of SHAP should equal model margin
    X_scaled, _ = model.preprocessor.transform_single(test_patient)
    model_prob = float(model.model.predict_proba(X_scaled)[0, 1])
    
    base_val = xai.expected_value
    shap_sum = sum(c["shap_value"] for c in exp["feature_contributions"])
    reconstructed_margin = base_val + shap_sum
    reconstructed_prob = 1.0 / (1.0 + np.exp(-reconstructed_margin))

    assert np.isclose(model_prob, reconstructed_prob, atol=1e-3), (
        f"Model prob {model_prob} and SHAP reconstructed prob {reconstructed_prob} must match"
    )


def test_what_if_simulation():
    """Verify counterfactual What-If simulation computes expected risk reduction."""
    model = DiabetesRiskModel.load("backend/artifacts")
    xai = ExplainableAIEngine(model)

    high_risk_data = SAMPLE_PATIENTS[2]["data"]
    modified_data = {**high_risk_data, "Glucose": 95.0, "BMI": 23.0}

    what_if = xai.simulate_what_if(high_risk_data, modified_data)
    assert what_if["is_improved"] is True
    assert what_if["percentage_risk_reduction"] > 40.0
    assert len(what_if["feature_shifts"]) > 0


if __name__ == "__main__":
    print("Running pipeline tests...")
    test_biological_zero_handling()
    print("✓ test_biological_zero_handling passed")
    test_model_training_and_metrics()
    print("✓ test_model_training_and_metrics passed")
    test_shap_additivity()
    print("✓ test_shap_additivity passed")
    test_what_if_simulation()
    print("✓ test_what_if_simulation passed")
    print("All pipeline tests passed successfully!")
