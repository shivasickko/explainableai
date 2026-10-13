"""Diabetes Risk Classification Model Module.

Trains, evaluates, and serializes the predictive machine learning model.
Computes comprehensive validation metrics, ROC curves, confusion matrices,
and global feature importance.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from backend.preprocessor import (
    CLINICAL_METADATA,
    FEATURE_NAMES,
    ClinicalDataPreprocessor,
    load_raw_dataset,
)


class DiabetesRiskModel:
    """Wrapper around XGBoost classifier with clinical calibration and metrics tracking."""

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        self.preprocessor = ClinicalDataPreprocessor()
        self.metrics = {}
        self.feature_importance = {}
        self.is_trained = False

    def train_and_evaluate(
        self, csv_path: str = "diabetes.csv", test_size: float = 0.2
    ) -> dict:
        """Fit preprocessor and model, and compute validation metrics."""
        X_raw, y = load_raw_dataset(csv_path)

        # Stratified train/test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size, random_state=42, stratify=y
        )

        # Preprocess
        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        # 5-fold cross validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv, scoring="roc_auc"
        )
        cv_acc_scores = cross_val_score(
            self.model, X_train, y_train, cv=cv, scoring="accuracy"
        )

        # Fit model on training set
        self.model.fit(X_train, y_train)

        # Evaluate on held-out test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Compute ROC curve points for dashboard visualization
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        # Downsample ROC points to ~20 points for clean frontend rendering
        roc_points = [
            {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
            for f, t in zip(fpr, tpr)
        ]

        self.metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_roc_auc_mean": round(float(cv_auc_scores.mean()), 4),
            "cv_accuracy_mean": round(float(cv_acc_scores.mean()), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "total_test_samples": int(len(y_test)),
            },
            "roc_curve": roc_points,
        }

        # Calculate feature importance
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        self.feature_importance = [
            {
                "feature": FEATURE_NAMES[i],
                "display_name": CLINICAL_METADATA[FEATURE_NAMES[i]]["display_name"],
                "importance": round(float(importances[i]), 4),
                "percentage": round(float(importances[i] * 100), 2),
            }
            for i in sorted_indices
        ]

        self.is_trained = True
        return self.metrics

    def predict_single(self, patient_data: dict) -> tuple[float, int, dict]:
        """Predict probability and binary risk for a single patient."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before prediction.")
        X_scaled, sanitized_values = self.preprocessor.transform_single(patient_data)
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        pred = int(prob >= 0.5)
        return prob, pred, sanitized_values

    def predict_batch(self, df_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
        X_scaled = self.preprocessor.transform(df_raw)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    def save(self, artifact_dir: str = "backend/artifacts"):
        """Save model, preprocessor, and metrics."""
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(artifact_dir, "xgb_model.joblib"))
        self.preprocessor.save(os.path.join(artifact_dir, "preprocessor.joblib"))
        with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
            json.dump(
                {
                    "metrics": self.metrics,
                    "feature_importance": self.feature_importance,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, artifact_dir: str = "backend/artifacts") -> "DiabetesRiskModel":
        """Load trained model and preprocessor."""
        instance = cls()
        instance.model = joblib.load(os.path.join(artifact_dir, "xgb_model.joblib"))
        instance.preprocessor = ClinicalDataPreprocessor.load(
            os.path.join(artifact_dir, "preprocessor.joblib")
        )
        with open(os.path.join(artifact_dir, "metrics.json")) as f:
            data = json.load(f)
            instance.metrics = data["metrics"]
            instance.feature_importance = data["feature_importance"]
        instance.is_trained = True
        return instance
