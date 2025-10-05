# scripts/run_predict_firstkill.py
from pathlib import Path
import pandas as pd

from src.models.train import Trainer          # reuse Trainer methods to build rounds
from src.models.predict import Predictor      # reuse your Predictor class to load artifact and featurize/predict
from src.viz.firstkill_predict_viz import plot_map_side_heatmaps, plot_weapon_compare
from src.config import Config

def main():
    cfg = Config()
    processed_dir = cfg.processed_dir

    # load processed mm_master_clean.parquet using Trainer helper
    trainer = Trainer()
    mm = trainer.load_processed()
    rounds = trainer.build_round_level_dataset(mm)   # returns round-level rows with first_kill_weapon, winner_team, etc.

    # instantiate Predictor (it will load artifact and feature_columns saved by Trainer)
    predictor = Predictor(cfg=cfg)   # will raise if artifact not present
    preds = predictor.predict_from_rounds(rounds)   # returns rounds + pred_int/pred_label/pred_proba

    # Merge preds back to rounds (predictor returns DataFrame copy including pred_proba)
    round_preds = preds.copy().reset_index(drop=True)

    # ensure columns: first_kill_weapon, first_kill_side, _map are preserved in the rounds dataframe
    # Predictor.predict_from_rounds returns 'pred_proba', 'pred_int' etc; but may not include first_kill_side/_map
    merged = rounds.reset_index(drop=True).copy()
    # If preds contains prediction columns and round_key, align on round_key:
    if "round_key" in merged.columns and "round_key" in round_preds.columns:
        merged = merged.merge(round_preds[["round_key","pred_proba","pred_int","pred_label"]], on="round_key", how="left")
    else:
        # fallback: append predictions in order (only safe if ordering matches)
        merged["pred_proba"] = round_preds.get("pred_proba")
        merged["pred_int"] = round_preds.get("pred_int")
        merged["pred_label"] = round_preds.get("pred_label")

    # create actual target (did first_kill_team == winner_team?)
    merged["actual_win"] = (merged["first_kill_team"].astype(str).fillna("") == merged["winner_team"].astype(str).fillna("")).astype(int)

    # If first_kill_side is missing but att_side exists, copy/rename
    if "first_kill_side" not in merged.columns and "att_side" in mm.columns:
        # try to propagate attacker side per round: use earliest event per round
        merged["first_kill_side"] = merged.get("first_kill_side")  # no-op if present

    # call viz functions
    plot_map_side_heatmaps(merged, processed_dir)
    plot_weapon_compare(merged, processed_dir, top_n=20)
    print("Predict+plot finished. Look in:", processed_dir / "figures" / "firstkill")

if __name__ == "__main__":
    main()
