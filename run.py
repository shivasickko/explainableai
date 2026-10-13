#!/usr/bin/env python3
"""Launcher for Explainable Diabetes Risk Prediction Dashboard.

Dhanalakshmi College of Engineering - Department of AI & Data Science
Batch 10
"""

import os
import sys

# Automatically use .venv python if present and not already running in it
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python"))
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)

import uvicorn
from backend.api import initialize_model_and_xai, app

if __name__ == "__main__":
    print("=" * 70)
    print(" Explainable Diabetes Risk Prediction Dashboard")
    print(" Clinical Feature Contribution Analysis via Explainable AI (SHAP)")
    print(" Dhanalakshmi College of Engineering — Dept of AI & Data Science")
    print(" Batch No: 10")
    print("=" * 70)

    # Initialize model and artifacts
    initialize_model_and_xai()

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n🚀 Dashboard server running at: http://localhost:{port}")
    print(f"📖 Interactive API Docs at:     http://localhost:{port}/docs")
    print("=" * 70)

    uvicorn.run("backend.api:app", host=host, port=port, reload=False)
