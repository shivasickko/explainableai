"""Clinical Data Preprocessing Module for Diabetes Risk Prediction.

Handles biological zero-value anomaly cleaning, median-based clinical imputation,
feature scaling, and input validation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_NAME = "Outcome"

# Features where 0 is biologically invalid for a living human
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

CLINICAL_METADATA = {
    "Glucose": {
        "display_name": "Plasma Glucose Concentration",
        "unit": "mg/dL",
        "normal_range": [70.0, 139.0],
        "prediabetes_range": [140.0, 199.0],
        "diabetes_threshold": 200.0,
        "typical_min": 50.0,
        "typical_max": 250.0,
        "description": "2-hour oral glucose tolerance test level",
        "category": "Metabolic",
    },
    "BloodPressure": {
        "display_name": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "normal_range": [60.0, 80.0],
        "elevated_range": [80.0, 89.0],
        "hypertension_threshold": 90.0,
        "typical_min": 40.0,
        "typical_max": 130.0,
        "description": "Resting diastolic blood pressure",
        "category": "Cardiovascular",
    },
    "BMI": {
        "display_name": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "normal_range": [18.5, 24.9],
        "overweight_range": [25.0, 29.9],
        "obese_threshold": 30.0,
        "typical_min": 15.0,
        "typical_max": 60.0,
        "description": "Weight in kg divided by height in meters squared",
        "category": "Physical / Anthropometric",
    },
    "Insulin": {
        "display_name": "2-Hour Serum Insulin",
        "unit": "μU/mL",
        "normal_range": [16.0, 166.0],
        "typical_min": 5.0,
        "typical_max": 500.0,
        "description": "Post-challenge serum insulin concentration",
        "category": "Endocrine",
    },
    "Age": {
        "display_name": "Patient Age",
        "unit": "years",
        "normal_range": [18.0, 45.0],
        "typical_min": 18.0,
        "typical_max": 90.0,
        "description": "Chronological patient age in years",
        "category": "Demographic",
    },
    "DiabetesPedigreeFunction": {
        "display_name": "Diabetes Pedigree Score",
        "unit": "score",
        "normal_range": [0.08, 0.45],
        "elevated_range": [0.46, 0.85],
        "high_threshold": 0.85,
        "typical_min": 0.05,
        "typical_max": 2.5,
        "description": "Genetic score modeling family history of diabetes",
        "category": "Genetics / History",
    },
    "SkinThickness": {
        "display_name": "Triceps Skinfold Thickness",
        "unit": "mm",
        "normal_range": [10.0, 28.0],
        "typical_min": 5.0,
        "typical_max": 60.0,
        "description": "Clinical estimate of subcutaneous adipose tissue",
        "category": "Physical / Anthropometric",
    },
    "Pregnancies": {
        "display_name": "Number of Pregnancies",
        "unit": "count",
        "normal_range": [0, 4],
        "typical_min": 0,
        "typical_max": 17,
        "description": "Number of gestational occurrences",
        "category": "Obstetric / Reproductive",
    },
}


class ClinicalDataPreprocessor:
    """Preprocessor that handles missing value replacement and scaling."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES
        self.zero_as_missing = ZERO_AS_MISSING
        self.feature_medians_ = {}

    def clean_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace biologically impossible zeros with NaN."""
        df_clean = df.copy()
        for col in self.zero_as_missing:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)
        return df_clean

    def fit(self, X: pd.DataFrame) -> "ClinicalDataPreprocessor":
        """Fit imputer and scaler on training DataFrame."""
        X_clean = self.clean_raw_dataframe(X[self.feature_names])
        X_imputed = self.imputer.fit_transform(X_clean)
        self.scaler.fit(X_imputed)

        for i, col in enumerate(self.feature_names):
            self.feature_medians_[col] = float(self.imputer.statistics_[i])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Impute and scale input features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X_clean = self.clean_raw_dataframe(X[self.feature_names])
        X_imputed = self.imputer.transform(X_clean)
        return self.scaler.transform(X_imputed)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform_single(self, patient_dict: dict) -> tuple[np.ndarray, dict]:
        """Convert a single patient dictionary into model-ready numpy array

        and return imputed/sanitized values.
        """
        row = {col: patient_dict.get(col, np.nan) for col in self.feature_names}
        df_row = pd.DataFrame([row])
        X_clean = self.clean_raw_dataframe(df_row)
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed)

        sanitized_dict = {
            col: float(X_imputed[0, i])
            for i, col in enumerate(self.feature_names)
        }
        return X_scaled, sanitized_dict

    def save(self, filepath: str):
        """Save preprocessor state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClinicalDataPreprocessor":
        """Load preprocessor state from disk."""
        return joblib.load(filepath)


def load_raw_dataset(csv_path: str = "diabetes.csv") -> tuple[pd.DataFrame, pd.Series]:
    """Load the Pima Indians dataset with proper headers."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file {csv_path} not found.")

    first_line = pd.read_csv(csv_path, nrows=1, header=None)
    # Check if first line is numeric (no header) or text (has header)
    try:
        float(first_line.iloc[0, 0])
        # Has no header
        df = pd.read_csv(csv_path, header=None, names=FEATURE_NAMES + [TARGET_NAME])
    except ValueError:
        # Has header
        df = pd.read_csv(csv_path)

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return X, y
