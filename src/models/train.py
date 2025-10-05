"""
Simple trainer that builds a binary classifier:
  target = first_kill_wins (1 if opening_kill_team == winner_team else 0)

Reads: processed/mm_master_clean.parquet
Writes: models/mm_firstkill_binary.joblib and models/mm_firstkill_binary_metrics.json

Dependencies: pandas, numpy, scikit-learn, joblib
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib

from ..config import Config


def _canon_weapon(w: Optional[str]) -> str:
    if pd.isna(w) or w is None:
        return "UNKNOWN"
    s = str(w).lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))
    s = s.replace("-", "").replace(" ", "_")
    return s or "UNKNOWN"


def _group_rare_categories(series: pd.Series, min_count: int = 40, other_label: str = "OTHER_WEAPON") -> pd.Series:
    counts = series.value_counts(dropna=True)
    keep = set(counts[counts >= min_count].index)
    return series.where(series.isin(keep), other_label)


class Trainer:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.processed_path = self.cfg.processed_dir / "mm_master_clean.parquet"
        self.model_dir = self.cfg.project_root / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.model_dir / "mm_firstkill_binary.joblib"
        self.metrics_path = self.model_dir / "mm_firstkill_binary_metrics.json"

    def load_processed(self) -> pd.DataFrame:
        if not self.processed_path.exists():
            raise FileNotFoundError(f"{self.processed_path} not found. Run ETL first.")
        return pd.read_parquet(self.processed_path)

    def build_round_level_dataset(self, mm: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame with columns:
         - round_key
         - first_kill_team
         - first_kill_weapon (canonicalized)
         - winner_team
        Prefer ETL's opening_kill_* fields; otherwise pick earliest event per round.
        """
        if mm is None or mm.empty:
            raise ValueError("Empty mm dataframe")

        df = mm.copy()

        # build round key
        if "file" in df.columns and "round" in df.columns:
            df["_round_key"] = df["file"].astype(str) + "__r__" + df["round"].astype(str)
        elif "match_id" in df.columns and "round_no" in df.columns:
            df["_round_key"] = df["match_id"].astype(str) + "__r__" + df["round_no"].astype(str)
        elif "_map" in df.columns and "round" in df.columns:
            df["_round_key"] = df["_map"].astype(str) + "__r__" + df["round"].astype(str)
        else:
            df["_round_key"] = df.index.astype(str)

        # If ETL produced explicit opening_kill columns, use them:
        if "opening_kill" in df.columns and "opening_kill_team" in df.columns:
            op = df[df["opening_kill"].astype(bool)].copy()
            sort_cols = [c for c in ("tick", "seconds") if c in op.columns]
            if sort_cols:
                op = op.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
            else:
                op = op.groupby("_round_key", as_index=False).first()

            winner_col = next((c for c in ("winner_team", "res_match_winner", "round_winner") if c in op.columns), None)
            winner_series = op[winner_col].astype(str) if winner_col and winner_col in op.columns else pd.Series([None] * len(op))
            weapon_col = "opening_kill_weapon" if "opening_kill_weapon" in op.columns else next((c for c in ("wp_canon", "wp", "weapon", "wp_type") if c in op.columns), None)

            out = pd.DataFrame({
                "round_key": op["_round_key"].astype(str),
                "first_kill_team": op["opening_kill_team"].astype(str).fillna("UNKNOWN"),
                "first_kill_weapon": op[weapon_col].apply(_canon_weapon).astype(str) if weapon_col else pd.Series(["UNKNOWN"] * len(op)),
                "winner_team": winner_series.fillna("UNKNOWN").astype(str),
            })
            return out.reset_index(drop=True)

        # fallback: earliest event per round
        sort_cols = [c for c in ("tick", "seconds") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
        else:
            df = df.groupby("_round_key", as_index=False).first()

        att_col = next((c for c in ("att_team", "attacker_team", "att_team_name") if c in df.columns), None)
        wp_col = next((c for c in ("wp_canon", "wp", "weapon", "wp_type") if c in df.columns), None)
        winner_col = next((c for c in ("winner_team", "res_match_winner", "round_winner") if c in df.columns), None)

        out = pd.DataFrame({
            "round_key": df["_round_key"].astype(str),
            "first_kill_team": df[att_col].astype(str).fillna("UNKNOWN") if att_col else pd.Series(["UNKNOWN"] * len(df)),
            "first_kill_weapon": df[wp_col].apply(_canon_weapon).astype(str).fillna("UNKNOWN") if wp_col else pd.Series(["UNKNOWN"] * len(df)),
            "winner_team": df[winner_col].astype(str).fillna("UNKNOWN") if winner_col else pd.Series(["UNKNOWN"] * len(df)),
        })
        return out.reset_index(drop=True)

    def featurize_binary(self, df_rounds: pd.DataFrame, min_weapon_count: int = 40) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Simple featurization:
         - canonicalize & group rare weapons
         - one-hot weapon categories -> features
         - target y: first_kill_wins (1 if first_kill_team == winner_team)
        """
        r = df_rounds.copy()
        if "first_kill_weapon" not in r.columns:
            r["first_kill_weapon"] = "UNKNOWN"
        r["first_kill_weapon"] = r["first_kill_weapon"].astype(str).fillna("UNKNOWN").apply(_canon_weapon)
        r["first_kill_weapon"] = _group_rare_categories(r["first_kill_weapon"], min_count=min_weapon_count)

        # target
        y = (r["first_kill_team"].astype(str).fillna("UNKNOWN") == r["winner_team"].astype(str).fillna("UNKNOWN")).astype(int)

        # one-hot weapon
        X = pd.get_dummies(r[["first_kill_weapon"]].astype(str), columns=["first_kill_weapon"], prefix=["wp"])
        feature_cols = X.columns.tolist()

        # inside Trainer.featurize_binary (or Trainer.featurize_binary equivalent)
        # after weapon one-hot (X already contains weapon ohe)
        # Add map one-hot (optional, set include_map True):
        if "_map" in r.columns:
            map_ohe = pd.get_dummies(r[["_map"]].astype(str), columns=["_map"], prefix=["map"])
            X = pd.concat([X, map_ohe], axis=1)
            feature_cols += map_ohe.columns.tolist()

        # Add attacker side as a feature if available
        if "first_kill_side" in r.columns:
            side_ohe = pd.get_dummies(r[["first_kill_side"]].astype(str), columns=["first_kill_side"], prefix=["side"])
            X = pd.concat([X, side_ohe], axis=1)
            feature_cols += side_ohe.columns.tolist()
        elif "att_side" in r.columns:
            side_ohe = pd.get_dummies(r[["att_side"]].astype(str), columns=["att_side"], prefix=["side"])
            X = pd.concat([X, side_ohe], axis=1)
            feature_cols += side_ohe.columns.tolist()

        # add optional numeric features if present (harmless if absent)
        for num in ("ct_eq_val", "t_eq_val", "avg_match_rank", "hp_dmg", "arm_dmg"):
            if num in r.columns:
                X[num] = pd.to_numeric(r[num], errors="coerce").fillna(0.0)
                feature_cols.append(num)

        return X.fillna(0.0), y, feature_cols

    def train_binary(self, test_size: float = 0.2, random_state: int = 42, min_weapon_count: int = 40) -> dict:
        mm = self.load_processed()
        rounds = self.build_round_level_dataset(mm)
        X, y, feature_cols = self.featurize_binary(rounds, min_weapon_count=min_weapon_count)

        if len(y.unique()) < 2:
            raise ValueError("Not enough class variety to train (need both 0 and 1).")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        roc_auc = float(roc_auc_score(y_test, y_proba)) if (y_proba is not None and len(set(y_test.tolist())) == 2) else None
        cm = confusion_matrix(y_test, y_pred).tolist()

        artifact = {
            "model": clf,
            "feature_columns": feature_cols,
            "label_map": {0: "first_kill_lost", 1: "first_kill_won"},
            "min_weapon_count": int(min_weapon_count),
        }
        joblib.dump(artifact, self.artifact_path)

        metrics = {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "classification_report": report,
            "confusion_matrix": cm,
            "n_features": len(feature_cols),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        with open(self.metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        return {"artifact_path": str(self.artifact_path), "metrics_path": str(self.metrics_path), "metrics": metrics}


if __name__ == "__main__":
    t = Trainer()
    info = t.train_binary()
    print("Training finished. Artifact saved to:", info["artifact_path"])
    print("Metrics saved to:", info["metrics_path"])
    print(json.dumps(info["metrics"], indent=2))
