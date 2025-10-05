from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.config import Config

try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

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
        self.model_dir = Path(self.cfg.project_root) / "models"
        self.artifact_path = Path(artifact_path) if artifact_path else (self.model_dir / "mm_firstkill_binary.joblib")
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {self.artifact_path}. Train first.")
        self._load_artifact()

    def _load_artifact(self):
        data = joblib.load(str(self.artifact_path))
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError("Unexpected artifact format.")
        self.model = data["model"]
        self.feature_columns = data.get("feature_columns", [])
        self.label_map = data.get("label_map", {0: "first_kill_lost", 1: "first_kill_won"})

    def _prepare_from_opening_events(self, df_open: pd.DataFrame) -> pd.DataFrame:
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

        wp_col = next((c for c in ("opening_kill_weapon","wp_canon","wp","weapon","wp_type") if c in df.columns), None)
        if wp_col:
            df["first_kill_weapon"] = df[wp_col].apply(_canon_weapon).astype(str)
        else:
            df["first_kill_weapon"] = "UNKNOWN"

        if "opening_kill_team" not in df.columns and "att_team" in df.columns:
            df["first_kill_team"] = df["att_team"]
        elif "opening_kill_team" in df.columns:
            df["first_kill_team"] = df["opening_kill_team"]

        if "winner_team" not in df.columns:
            if "res_match_winner" in df.columns:
                df["winner_team"] = df["res_match_winner"]
            else:
                df["winner_team"] = pd.NA

        return df.reset_index(drop=True)

    def _featurize(self, df_rounds: pd.DataFrame) -> pd.DataFrame:
        df = df_rounds.copy()
        df["first_kill_weapon"] = df.get("first_kill_weapon", "UNKNOWN").astype(str).apply(_canon_weapon)
        X = pd.get_dummies(df[["first_kill_weapon"]].astype(str), columns=["first_kill_weapon"], prefix=["wp"])
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
        X = self._featurize(df_rounds)
        probs = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        preds = self.model.predict(X)
        out = df_rounds.copy().reset_index(drop=True)
        out["pred_int"] = preds.astype(int)
        out["pred_label"] = out["pred_int"].map(self.label_map)
        out["pred_proba"] = probs.tolist() if probs is not None else None
        return out

    # ---------- plotting helpers ----------
    def _plot_map_side_heatmaps(self, rounds: pd.DataFrame, outdir: Path):
        if ("_map" not in rounds.columns) or ("first_kill_side" not in rounds.columns and "att_side" not in rounds.columns):
            print("Skipping map×side heatmaps: missing _map or side column")
            return None

        # ensure side column
        if "first_kill_side" not in rounds.columns and "att_side" in rounds.columns:
            rounds["first_kill_side"] = rounds["att_side"].astype(str)

        grp = rounds.groupby(["_map", "first_kill_side"]).agg(
            actual_winrate=("actual_win", "mean"),
            pred_proba_mean=("pred_proba", "mean"),
            n=("round_key", "count")
        ).reset_index()

        if grp.empty:
            print("Map×side grouping empty, skipping.")
            return None

        pivot_act = grp.pivot(index="_map", columns="first_kill_side", values="actual_winrate")
        pivot_pred = grp.pivot(index="_map", columns="first_kill_side", values="pred_proba_mean")

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.5 * len(pivot_act))))
        if HAVE_SNS:
            sns.heatmap(pivot_act, annot=True, fmt=".2f", ax=axes[0], cmap="vlag", cbar_kws={"label": "actual winrate"})
            sns.heatmap(pivot_pred, annot=True, fmt=".2f", ax=axes[1], cmap="vlag", cbar_kws={"label": "predicted prob"})
        else:
            axes[0].imshow(pivot_act.fillna(0.0).values, aspect="auto")
            axes[0].set_yticks(range(len(pivot_act.index))); axes[0].set_yticklabels(pivot_act.index)
            axes[0].set_xticks(range(len(pivot_act.columns))); axes[0].set_xticklabels(pivot_act.columns)
            axes[1].imshow(pivot_pred.fillna(0.0).values, aspect="auto")
            axes[1].set_yticks(range(len(pivot_pred.index))); axes[1].set_yticklabels(pivot_pred.index)
            axes[1].set_xticks(range(len(pivot_pred.columns))); axes[1].set_xticklabels(pivot_pred.columns)

        axes[0].set_title("Actual winrate of team with opening kill (map × side)")
        axes[1].set_title("Predicted mean probability (map × side)")
        plt.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / "firstkill_winrate_map_side_heatmaps.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("Saved:", out)
        return out

    def _plot_weapon_compare(self, rounds: pd.DataFrame, outdir: Path, top_n: int = 20):
        if "first_kill_weapon" not in rounds.columns:
            print("Skipping weapon plot: missing first_kill_weapon")
            return None

        rounds["_wp"] = rounds["first_kill_weapon"].astype(str)
        counts = rounds["_wp"].value_counts()
        top = counts.nlargest(top_n).index.tolist()
        sub = rounds[rounds["_wp"].isin(top)].copy()
        if sub.empty:
            print("No weapon data available for weapon plot; skipping.")
            return None

        grp = sub.groupby("_wp").agg(
            actual_win=("actual_win", "mean"),
            pred_proba=("pred_proba", "mean"),
            n=("round_key", "count")
        ).reset_index().sort_values("actual_win", ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(grp))))
        y = list(range(len(grp)))
        width = 0.35
        ax.barh([i - width/2 for i in y], grp["actual_win"], height=width, label="actual winrate")
        ax.barh([i + width/2 for i in y], grp["pred_proba"], height=width, label="predicted prob")
        ax.set_yticks(list(y))
        ax.set_yticklabels(grp["_wp"])
        ax.invert_yaxis()
        ax.set_xlabel("Winrate / predicted probability")
        ax.set_title(f"First-kill winrate by weapon (top {top_n}) — actual vs predicted")
        ax.legend()
        plt.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / "firstkill_winrate_by_weapon_topN.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("Saved:", out)
        return out

    def predict_and_plot(self, rounds: Optional[pd.DataFrame] = None, outdir: Optional[Path] = None, top_n: int = 20) -> pd.DataFrame:
        # Build rounds using Trainer helper if not supplied
        if rounds is None:
            from src.models.train import Trainer
            t = Trainer()
            mm = t.load_processed()
            rounds = t.build_round_level_dataset(mm)

        # ensure weapon col
        if "first_kill_weapon" not in rounds.columns:
            if "opening_kill_weapon" in rounds.columns:
                rounds["first_kill_weapon"] = rounds["opening_kill_weapon"].astype(str).apply(_canon_weapon)
            else:
                rounds["first_kill_weapon"] = "UNKNOWN"

        # create predictions
        preds = self.predict_from_rounds(rounds)
        if "round_key" in rounds.columns and "round_key" in preds.columns:
            merged = rounds.merge(preds[["round_key","pred_proba","pred_int"]], on="round_key", how="left")
        else:
            merged = rounds.copy().reset_index(drop=True)
            merged["pred_proba"] = preds.get("pred_proba")
            merged["pred_int"] = preds.get("pred_int")

        # actual target
        merged["actual_win"] = (merged["first_kill_team"].astype(str).fillna("") == merged["winner_team"].astype(str).fillna("")).astype(int)

        # side fallback
        if "first_kill_side" not in merged.columns and "att_side" in merged.columns:
            merged["first_kill_side"] = merged["att_side"].astype(str)

        # outdir for figures
        processed_dir = Path(self.cfg.processed_dir) if hasattr(self.cfg, "processed_dir") else Path("data/processed")
        fig_out = (Path(outdir) if outdir else Path(processed_dir)) / "figures" / "firstkill"
        fig_out.mkdir(parents=True, exist_ok=True)

        self._plot_map_side_heatmaps(merged, fig_out)
        self._plot_weapon_compare(merged, fig_out, top_n=top_n)

        return merged

if __name__ == "__main__":
    cfg = Config()
    processed = Path(cfg.processed_dir) / "mm_master_clean.parquet"
    if not processed.exists():
        raise SystemExit("Run ETL first")
    mm = pd.read_parquet(processed)

    # choose opening rows
    if "opening_kill" in mm.columns:
        openings = mm[mm["opening_kill"].astype(bool)]
    else:
        sort_cols = [c for c in ("tick","seconds") if c in mm.columns]
        if sort_cols:
            openings = mm.sort_values(sort_cols).groupby(["file","round"], as_index=False).first().reset_index()
        else:
            openings = mm.groupby(["file","round"], as_index=False).first().reset_index()

    p = Predictor(cfg=cfg)
    merged = p.predict_and_plot(openings)
    print(merged[["round_key","pred_proba","actual_win"]].head(10))
