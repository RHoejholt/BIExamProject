"""
Predictor for the binary first-kill model (first_kill_wins).

Reads saved artifact models/mm_firstkill_binary.joblib and predicts on provided
opening-kill events (or an already-aggregated per-round DataFrame).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import joblib

from ..config import Config


def _canon_weapon(w):
    if pd.isna(w) or w is None:
        return "UNKNOWN"
    s = str(w).lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))
    s = s.replace("-", "").replace(" ", "_")
    return s or "UNKNOWN"


class Predictor:
    def __init__(self, cfg: Optional[Config] = None, artifact_path: Optional[Path] = None):
        self.cfg = cfg or Config()
        self.model_dir = self.cfg.project_root / "models"
        self.artifact_path = Path(artifact_path) if artifact_path else (self.model_dir / "mm_firstkill_binary.joblib")
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {self.artifact_path}. Train first.")
        self._load_artifact()

    def _load_artifact(self):
        data = joblib.load(self.artifact_path)
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError("Unexpected artifact format.")
        self.model = data["model"]
        self.feature_columns = data.get("feature_columns", [])
        self.label_map = data.get("label_map", {0: "first_kill_lost", 1: "first_kill_won"})

    def _prepare_from_opening_events(self, df_open: pd.DataFrame) -> pd.DataFrame:
        """Reduce opening event rows to one row per round and canonicalize weapon."""
        df = df_open.copy()
        # round_key
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

        # weapon col candidates
        wp_col = next((c for c in ("opening_kill_weapon", "wp_canon", "wp", "weapon", "wp_type") if c in df.columns), None)
        if wp_col:
            df["first_kill_weapon"] = df[wp_col].apply(_canon_weapon).astype(str)
        else:
            df["first_kill_weapon"] = "UNKNOWN"

        # ensure first_kill_team and winner_team exist if later used for debug
        if "opening_kill_team" not in df.columns and "att_team" in df.columns:
            df["first_kill_team"] = df["att_team"]
        elif "opening_kill_team" in df.columns:
            df["first_kill_team"] = df["opening_kill_team"]

        if "winner_team" not in df.columns:
            # try common alternatives
            if "res_match_winner" in df.columns:
                df["winner_team"] = df["res_match_winner"]
            else:
                df["winner_team"] = pd.NA

        return df.reset_index(drop=True)

    def _featurize(self, df_rounds: pd.DataFrame) -> pd.DataFrame:
        df = df_rounds.copy()
        df["first_kill_weapon"] = df.get("first_kill_weapon", "UNKNOWN").astype(str).apply(_canon_weapon)
        X = pd.get_dummies(df[["first_kill_weapon"]].astype(str), columns=["first_kill_weapon"], prefix=["wp"])
        # ensure all training columns are present
        for c in self.feature_columns:
            if c not in X.columns:
                X[c] = 0
        X = X[self.feature_columns]
        return X.fillna(0.0)

    def predict_from_opening_events(self, df_openings: pd.DataFrame) -> pd.DataFrame:
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
        """Accept DataFrame already containing 'first_kill_weapon' (and round_key)."""
        X = self._featurize(df_rounds)
        probs = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        preds = self.model.predict(X)
        out = df_rounds.copy().reset_index(drop=True)
        out["pred_int"] = preds.astype(int)
        out["pred_label"] = out["pred_int"].map(self.label_map)
        out["pred_proba"] = probs.tolist() if probs is not None else None
        return out


if __name__ == "__main__":
    cfg = Config()
    processed = cfg.processed_dir / "mm_master_clean.parquet"
    if not processed.exists():
        raise SystemExit("Run ETL first")
    mm = pd.read_parquet(processed)
    # choose opening rows
    if "opening_kill" in mm.columns:
        openings = mm[mm["opening_kill"].astype(bool)]
    else:
        sort_cols = [c for c in ("tick", "seconds") if c in mm.columns]
        if sort_cols:
            openings = mm.sort_values(sort_cols).groupby(["file", "round"], as_index=False).first().reset_index()
        else:
            openings = mm.groupby(["file", "round"], as_index=False).first().reset_index()

    p = Predictor(cfg=cfg)
    preds = p.predict_from_opening_events(openings)
    print(preds.head(20))
