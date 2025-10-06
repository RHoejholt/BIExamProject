from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import joblib

from src.config import Config


def _canon_weapon(w):
    # canonicalize weapon strings
    if pd.isna(w) or w is None:
        return "UNKNOWN"
    s = str(w).lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))
    s = s.replace("-", "").replace(" ", "_")
    return s or "UNKNOWN"


class Predictor:
    def __init__(self, cfg: Optional[Config] = None, artifact_path: Optional[Path] = None):
        # init: load artifact from disk
        self.cfg = cfg or Config()
        self.model_dir = Path(self.cfg.project_root) / "models"
        self.artifact_path = Path(artifact_path) if artifact_path else (self.model_dir / "mm_firstkill_binary.joblib")
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"artifact not found at {self.artifact_path}. train first.")
        self._load_artifact()

    def _load_artifact(self):
        # hent artifact dict med model og feature_columns
        data = joblib.load(str(self.artifact_path))
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError("unexpected artifact format")
        self.model = data["model"]
        self.feature_columns = data.get("feature_columns", [])
        self.label_map = data.get("label_map", {0: "first_kill_lost", 1: "first_kill_won"})

    def _prepare_from_opening_events(self, df_open: pd.DataFrame) -> pd.DataFrame:
        # lav one-row-per-round fra opening events og canonicalize weapon/team/winner hvis muligt
        df = df_open.copy()

        if "file" in df.columns and "round" in df.columns:
            df["_round_key"] = df["file"].astype(str) + "__r__" + df["round"].astype(str)
        elif "match_id" in df.columns and "round_no" in df.columns:
            df["_round_key"] = df["match_id"].astype(str) + "__r__" + df["round_no"].astype(str)
        else:
            df["_round_key"] = df.index.astype(str)

        sort_cols = [c for c in ("tick", "seconds") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
        else:
            df = df.groupby("_round_key", as_index=False).first()

        wp_col = next((c for c in ("opening_kill_weapon", "wp_canon", "wp", "weapon", "wp_type") if c in df.columns), None)
        if wp_col:
            df.loc[:, "first_kill_weapon"] = df[wp_col].apply(_canon_weapon).astype(str)
        else:
            df.loc[:, "first_kill_weapon"] = "UNKNOWN"

        if "opening_kill_team" in df.columns:
            df.loc[:, "first_kill_team"] = df["opening_kill_team"].astype(str)
        elif "att_team" in df.columns:
            df.loc[:, "first_kill_team"] = df["att_team"].astype(str)
        else:
            df.loc[:, "first_kill_team"] = "UNKNOWN"

        # winner_team if possible (simple mapping)
        if "winner_team" in df.columns:
            df.loc[:, "winner_team"] = df["winner_team"].astype(str)
        else:
            df.loc[:, "winner_team"] = df.get("res_match_winner", pd.NA).astype(str).fillna("UNKNOWN")

        return df.reset_index(drop=True)

    def _featurize(self, df_rounds: pd.DataFrame) -> pd.DataFrame:
        # featurize: one-hot weapon and align to training feature columns
        df = df_rounds.copy()
        df.loc[:, "first_kill_weapon"] = df.get("first_kill_weapon", "UNKNOWN").astype(str).apply(_canon_weapon)
        X = pd.get_dummies(df[["first_kill_weapon"]].astype(str), columns=["first_kill_weapon"], prefix=["wp"])
        # ensure all feature_columns exist
        for c in self.feature_columns:
            if c not in X.columns:
                X[c] = 0
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        return X.fillna(0.0)

    def predict_from_opening_events(self, df_openings: pd.DataFrame) -> pd.DataFrame:
        # predict fra opening-events
        df_rounds = self._prepare_from_opening_events(df_openings)
        X = self._featurize(df_rounds)
        probs = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        preds = self.model.predict(X)
        out = pd.DataFrame({
            "round_key": df_rounds["_round_key"].values,
            "pred_int": preds.astype(int),
            "pred_label": [self.label_map.get(intv, str(intv)) for intv in preds.astype(int)],
            "pred_proba": probs.tolist() if probs is not None else None,
        })
        return out

    def predict_from_rounds(self, df_rounds: pd.DataFrame) -> pd.DataFrame:
        # predict når du allerede har one-row-per-round DF
        X = self._featurize(df_rounds)
        probs = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        preds = self.model.predict(X)
        out = df_rounds.copy().reset_index(drop=True)
        out.loc[:, "pred_int"] = preds.astype(int)
        out.loc[:, "pred_label"] = out["pred_int"].map(self.label_map)
        out.loc[:, "pred_proba"] = probs.tolist() if probs is not None else None
        return out