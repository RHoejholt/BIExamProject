# src/models/predict.py
import joblib
from pathlib import Path
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parents[0] / "artifacts"

def load_win_classifier():
    p = ARTIFACT_DIR / "win_classifier.joblib"
    if not p.exists():
        raise FileNotFoundError(p)
    return joblib.load(p)

def predict_win(df: pd.DataFrame):
    clf = load_win_classifier()
    return clf.predict(df), clf.predict_proba(df)[:, 1]
