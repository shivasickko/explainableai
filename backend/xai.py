"""Explainable AI (XAI) Engine for Diabetes Risk Prediction.

Provides local and global feature contribution analysis using TreeSHAP,
translates Shapley values into clinical insights and risk attribution,
and simulates counterfactual "What-If" clinical scenarios.
"""

import numpy as np
import pandas as pd
import shap
from backend.preprocessor import CLINICAL_METADATA, FEATURE_NAMES


class ExplainableAIEngine:
    """Computes exact Shapley feature attributions and clinical explanations."""

    def __init__(self, risk_model):
        self.risk_model = risk_model
        self.explainer = shap.TreeExplainer(risk_model.model)
        self.expected_value = float(np.asarray(self.explainer.expected_value).ravel()[0])
        # Convert base log-odds to baseline population probability
        self.base_probability = float(1.0 / (1.0 + np.exp(-self.expected_value)))

    def explain_patient(self, patient_data: dict) -> dict:
        """Produce comprehensive prediction, SHAP contribution analysis,

        and clinical insights for a patient.
        """
        # Transform patient input
        X_scaled, sanitized_values = self.risk_model.preprocessor.transform_single(
            patient_data
        )

        # Predict probability
        prob = float(self.risk_model.model.predict_proba(X_scaled)[0, 1])

        # Compute SHAP values in margin (log-odds) space
        shap_values = self.explainer.shap_values(X_scaled)[0]

        # Determine risk tier
        if prob < 0.30:
            risk_tier = "Low Risk"
            risk_severity = "low"
            risk_color = "#10B981"  # Emerald green
            clinical_summary = (
                "Patient demonstrates low overall probability of diabetes based on clinical "
                "parameters. Key biomarkers fall within healthy reference windows."
            )
        elif prob < 0.65:
            risk_tier = "Moderate Risk"
            risk_severity = "moderate"
            risk_color = "#F59E0B"  # Amber / Warning
            clinical_summary = (
                "Patient displays borderline or elevated risk markers consistent with pre-diabetes. "
                "Targeted clinical and lifestyle interventions are recommended."
            )
        else:
            risk_tier = "High Risk"
            risk_severity = "high"
            risk_color = "#EF4444"  # Red / Alert
            clinical_summary = (
                "High clinical likelihood of diabetes identified. Multiple key biomarkers "
                "substantially elevate the predictive risk. Immediate clinical review advised."
            )

        # Process individual feature contributions
        contributions = []
        total_positive_shap = sum(v for v in shap_values if v > 0)
        total_negative_shap = sum(abs(v) for v in shap_values if v < 0)

        for i, feat_name in enumerate(FEATURE_NAMES):
            shap_val = float(shap_values[i])
            patient_val = float(sanitized_values[feat_name])
            meta = CLINICAL_METADATA.get(feat_name, {})
            normal_rng = meta.get("normal_range", [0, 0])

            # Determine direction and impact
            if shap_val > 0.02:
                direction = "Risk-Increasing"
                effect_type = "positive"
            elif shap_val < -0.02:
                direction = "Protective (Risk-Decreasing)"
                effect_type = "negative"
            else:
                direction = "Neutral / Minimal"
                effect_type = "neutral"

            # Clinical status evaluation
            status, interpretation, action_recommendation = (
                self._evaluate_clinical_status(feat_name, patient_val, shap_val)
            )

            contributions.append(
                {
                    "feature": feat_name,
                    "display_name": meta.get("display_name", feat_name),
                    "patient_value": round(patient_val, 2),
                    "unit": meta.get("unit", ""),
                    "normal_range": normal_rng,
                    "normal_range_str": f"{normal_rng[0]} - {normal_rng[1]} {meta.get('unit', '')}",
                    "category": meta.get("category", "General"),
                    "shap_value": round(shap_val, 4),
                    "direction": direction,
                    "effect_type": effect_type,
                    "magnitude": round(abs(shap_val), 4),
                    "clinical_status": status,
                    "interpretation": interpretation,
                    "action_recommendation": action_recommendation,
                }
            )

        # Sort contributions by magnitude (descending)
        sorted_contributions = sorted(
            contributions, key=lambda x: x["magnitude"], reverse=True
        )

        top_risk_factors = [
            c for c in sorted_contributions if c["effect_type"] == "positive"
        ][:3]
        top_protective_factors = [
            c for c in sorted_contributions if c["effect_type"] == "negative"
        ][:3]

        return {
            "prediction": {
                "probability": round(prob, 4),
                "probability_percent": round(prob * 100, 1),
                "risk_tier": risk_tier,
                "risk_severity": risk_severity,
                "risk_color": risk_color,
                "base_probability": round(self.base_probability, 4),
                "base_probability_percent": round(self.base_probability * 100, 1),
                "clinical_summary": clinical_summary,
            },
            "patient_values": sanitized_values,
            "feature_contributions": sorted_contributions,
            "top_risk_factors": top_risk_factors,
            "top_protective_factors": top_protective_factors,
        }

    def simulate_what_if(self, baseline_patient: dict, modified_patient: dict) -> dict:
        """Simulate how targeted clinical modifications impact predicted risk and SHAP scores."""
        baseline_res = self.explain_patient(baseline_patient)
        modified_res = self.explain_patient(modified_patient)

        base_p = baseline_res["prediction"]["probability"]
        mod_p = modified_res["prediction"]["probability"]
        diff_p = mod_p - base_p
        percent_reduction = (
            ((base_p - mod_p) / base_p * 100.0) if base_p > 0 else 0.0
        )

        # Calculate shift for each feature
        feature_shifts = []
        base_contrib_map = {
            c["feature"]: c for c in baseline_res["feature_contributions"]
        }
        mod_contrib_map = {
            c["feature"]: c for c in modified_res["feature_contributions"]
        }

        for feat in FEATURE_NAMES:
            b_c = base_contrib_map[feat]
            m_c = mod_contrib_map[feat]
            val_delta = m_c["patient_value"] - b_c["patient_value"]
            shap_delta = m_c["shap_value"] - b_c["shap_value"]

            if abs(val_delta) > 0.001 or abs(shap_delta) > 0.01:
                feature_shifts.append(
                    {
                        "feature": feat,
                        "display_name": b_c["display_name"],
                        "baseline_value": b_c["patient_value"],
                        "target_value": m_c["patient_value"],
                        "unit": b_c["unit"],
                        "value_change": round(val_delta, 2),
                        "baseline_shap": b_c["shap_value"],
                        "target_shap": m_c["shap_value"],
                        "shap_delta": round(shap_delta, 4),
                        "risk_impact": (
                            "Risk Reduced"
                            if shap_delta < 0
                            else ("Risk Increased" if shap_delta > 0 else "Unchanged")
                        ),
                    }
                )

        return {
            "baseline": baseline_res["prediction"],
            "target": modified_res["prediction"],
            "probability_difference": round(diff_p, 4),
            "percentage_risk_reduction": round(percent_reduction, 1),
            "is_improved": diff_p < 0,
            "feature_shifts": feature_shifts,
        }

    def _evaluate_clinical_status(
        self, feature: str, value: float, shap_val: float
    ) -> tuple[str, str, str]:
        """Generate clinical status, diagnostic interpretation, and actionable recommendation."""
        if feature == "Glucose":
            if value >= 200.0:
                return (
                    "High (Diabetic Range)",
                    f"Glucose level of {value:.1f} mg/dL exceeds the diagnostic threshold (≥200 mg/dL). Strongly drives model risk higher.",
                    "Urgent endocrinology consult recommended; confirm with HbA1c testing and initiate glycemic control protocol.",
                )
            elif value >= 140.0:
                return (
                    "Elevated (Pre-Diabetic)",
                    f"Glucose of {value:.1f} mg/dL is in impaired glucose tolerance range (140-199 mg/dL). Elevated risk contribution.",
                    "Adopt low glycemic-index nutritional regimen and structured moderate-intensity aerobic exercise.",
                )
            else:
                return (
                    "Normal Range",
                    f"Glucose of {value:.1f} mg/dL is within healthy physiological limits (<140 mg/dL). Serves as a protective factor.",
                    "Maintain current balanced dietary habits and regular physical activity.",
                )

        elif feature == "BMI":
            if value >= 30.0:
                return (
                    "Obese (Class I+)",
                    f"BMI of {value:.1f} kg/m² indicates obesity (≥30.0 kg/m²), promoting chronic insulin resistance and elevated risk.",
                    "Target 5-10% body weight reduction through structured caloric deficit and behavioral lifestyle support.",
                )
            elif value >= 25.0:
                return (
                    "Overweight",
                    f"BMI of {value:.1f} kg/m² is in the overweight tier (25.0-29.9 kg/m²), moderately elevating diabetes risk.",
                    "Engage in strength and cardio training 150 minutes weekly to improve metabolic clearance.",
                )
            elif value < 18.5:
                return (
                    "Underweight",
                    f"BMI of {value:.1f} kg/m² is below the standard physiological range (<18.5 kg/m²).",
                    "Nutritional assessment recommended to ensure adequate caloric and micronutrient intake.",
                )
            else:
                return (
                    "Optimal Weight",
                    f"BMI of {value:.1f} kg/m² is in healthy reference range (18.5-24.9 kg/m²), providing protection against metabolic dysfunction.",
                    "Continue maintaining healthy body composition.",
                )

        elif feature == "BloodPressure":
            if value >= 90.0:
                return (
                    "Stage 2 Hypertension",
                    f"Diastolic BP of {value:.1f} mmHg reflects elevated vascular strain, compounding metabolic complications.",
                    "Monitor blood pressure daily; consider DASH dietary pattern and sodium restriction.",
                )
            elif value >= 80.0:
                return (
                    "Elevated / Stage 1",
                    f"Diastolic BP of {value:.1f} mmHg is slightly elevated (80-89 mmHg).",
                    "Routine cardiovascular monitoring and stress-reduction techniques advised.",
                )
            else:
                return (
                    "Normal BP",
                    f"Diastolic BP of {value:.1f} mmHg is optimal (<80 mmHg).",
                    "Maintain heart-healthy lifestyle habits.",
                )

        elif feature == "Insulin":
            if value >= 166.0:
                return (
                    "Hyperinsulinemia",
                    f"Serum insulin of {value:.1f} μU/mL indicates hyperinsulinemic response compensatory for peripheral insulin resistance.",
                    "Focus on improving peripheral insulin sensitivity through resistance training and reducing refined carbohydrates.",
                )
            elif value < 16.0:
                return (
                    "Low Insulin",
                    f"Serum insulin of {value:.1f} μU/mL may suggest reduced beta-cell secretory capacity.",
                    "Correlate with fasting glucose and C-peptide levels.",
                )
            else:
                return (
                    "Normal Serum Insulin",
                    f"Serum insulin of {value:.1f} μU/mL is within typical reference limits (16-166 μU/mL).",
                    "Preserve pancreatic beta-cell health via low-inflammation nutrition.",
                )

        elif feature == "Age":
            if value >= 50.0:
                return (
                    "Mature Age Factor",
                    f"Age {int(value)} years represents an unmodifiable demographic risk factor associated with age-related beta-cell decline.",
                    "Annual metabolic screening and preventive health checkups recommended.",
                )
            elif value >= 35.0:
                return (
                    "Intermediate Age Tier",
                    f"Age {int(value)} years. Risk increases gradually past age 35.",
                    "Biennial routine metabolic screening.",
                )
            else:
                return (
                    "Younger Age Tier",
                    f"Age {int(value)} years is protective against age-driven metabolic degradation.",
                    "Establish long-term protective cardiovascular and dietary habits early.",
                )

        elif feature == "DiabetesPedigreeFunction":
            if value >= 0.80:
                return (
                    "Strong Genetic Risk",
                    f"Pedigree score of {value:.3f} indicates significant familial history of diabetes, conferring inherited susceptibility.",
                    "Heightened vigilance advised; genetic risk can be mitigated with strict adherence to healthy lifestyle.",
                )
            elif value >= 0.45:
                return (
                    "Moderate Genetic Predisposition",
                    f"Pedigree score of {value:.3f} reflects moderate genetic background.",
                    "Regular fasting glucose checks.",
                )
            else:
                return (
                    "Low Genetic Susceptibility",
                    f"Pedigree score of {value:.3f} indicates minimal known family history of diabetes.",
                    "Focus on environmental and lifestyle health factors.",
                )

        elif feature == "SkinThickness":
            if value >= 35.0:
                return (
                    "Elevated Subcutaneous Adiposity",
                    f"Skinfold thickness of {value:.1f} mm correlates with elevated peripheral adipose tissue.",
                    "Incorporate body recomposition exercises.",
                )
            else:
                return (
                    "Normal Subcutaneous Tissue",
                    f"Skinfold thickness of {value:.1f} mm is within standard ranges.",
                    "Maintain active lifestyle.",
                )

        elif feature == "Pregnancies":
            if value >= 5:
                return (
                    "Multiparity Factor",
                    f"History of {int(value)} pregnancies. Multiparity has clinical associations with gestational metabolic strain.",
                    "Screen for history of gestational diabetes mellitus (GDM).",
                )
            else:
                return (
                    "Standard Gestational History",
                    f"{int(value)} previous pregnancies.",
                    "Standard periodic wellness evaluation.",
                )

        return ("Evaluated", f"Value: {value:.2f}", "Standard clinical monitoring.")


SAMPLE_PATIENTS = [
    {
        "id": "healthy_patient",
        "title": "Healthy Adult (Low Risk)",
        "badge": "Low Risk (Healthy)",
        "badge_color": "emerald",
        "description": "Young adult with optimal fasting glucose, healthy BMI, and normal blood pressure.",
        "data": {
            "Pregnancies": 1,
            "Glucose": 88.0,
            "BloodPressure": 68.0,
            "SkinThickness": 20.0,
            "Insulin": 55.0,
            "BMI": 22.4,
            "DiabetesPedigreeFunction": 0.22,
            "Age": 26,
        },
    },
    {
        "id": "borderline_patient",
        "title": "Borderline Pre-Diabetic (Moderate Risk)",
        "badge": "Moderate Risk (Pre-Diabetic)",
        "badge_color": "amber",
        "description": "Middle-aged patient with impaired fasting glucose and overweight BMI.",
        "data": {
            "Pregnancies": 3,
            "Glucose": 142.0,
            "BloodPressure": 82.0,
            "SkinThickness": 28.0,
            "Insulin": 115.0,
            "BMI": 28.6,
            "DiabetesPedigreeFunction": 0.48,
            "Age": 44,
        },
    },
    {
        "id": "high_risk_patient",
        "title": "Metabolic Syndrome (High Risk)",
        "badge": "High Risk (Critical)",
        "badge_color": "rose",
        "description": "Senior patient with high glucose, obesity (BMI > 35), and strong family history.",
        "data": {
            "Pregnancies": 6,
            "Glucose": 178.0,
            "BloodPressure": 88.0,
            "SkinThickness": 38.0,
            "Insulin": 210.0,
            "BMI": 36.4,
            "DiabetesPedigreeFunction": 0.85,
            "Age": 54,
        },
    },
    {
        "id": "genetic_risk_patient",
        "title": "Young Adult with Strong Genetic Risk",
        "badge": "Elevated Genetic Risk",
        "badge_color": "purple",
        "description": "Young individual with prominent familial diabetes pedigree and mild glucose elevation.",
        "data": {
            "Pregnancies": 0,
            "Glucose": 128.0,
            "BloodPressure": 74.0,
            "SkinThickness": 24.0,
            "Insulin": 85.0,
            "BMI": 26.2,
            "DiabetesPedigreeFunction": 1.45,
            "Age": 23,
        },
    },
]
