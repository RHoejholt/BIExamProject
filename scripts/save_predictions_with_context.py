import pandas as pd
from src.config import Config
from src.models.predict import Predictor

cfg = Config()
proc_path = cfg.processed_dir / "mm_master_clean.parquet"
mm = pd.read_parquet(proc_path)

# choose opening rows (ETL should have opening_kill)
if "opening_kill" in mm.columns:
    openings = mm[mm["opening_kill"].astype(bool)].copy()
else:
    # fallback earliest event per round
    sort_cols = [c for c in ("tick","seconds") if c in mm.columns]
    if sort_cols:
        openings = mm.sort_values(sort_cols).groupby(["file","round"], as_index=False).first().reset_index(drop=True)
    else:
        openings = mm.groupby(["file","round"], as_index=False).first().reset_index(drop=True)

p = Predictor(cfg=cfg)
preds = p.predict_from_opening_events(openings)

# merge predictions back to openings for full context
out = openings.merge(preds, left_on=(
    openings["file"].astype(str) + "__r__" + openings["round"].astype(str)
) if "file" in openings.columns and "round" in openings.columns else openings["_round_key"],
                     right_on="round_key",
                     how="right", suffixes=("", "_pred"))

# pick columns to save
save_cols = [
    "round_key", "file", "round", "_map", "tick", "att_team", "opening_kill_team",
    "opening_kill_weapon", "first_kill_weapon", "winner_team", "pred_int", "pred_label", "pred_proba"
]
# keep only existing columns
save_cols = [c for c in save_cols if c in out.columns]
out[save_cols].to_csv("../models/predictions_with_context.csv", index=False)
print("Saved ../models/predictions_with_context.csv")
print(out[save_cols].head(20))
