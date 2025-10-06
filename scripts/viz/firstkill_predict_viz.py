from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

from src.config import Config
from src.models.predict import Predictor

# Tuning
MIN_MAP_OPENS = 100     # only include maps with at least this many opening events
MIN_WEAPON_COUNT = 40   # min times a weapon must appear to be included
TOP_WEAPONS = 30        # max weapons to plot
FIG_DPI = 150

def _canon_weapon(w):
    if pd.isna(w) or w is None:
        return "UNKNOWN"
    s = str(w).lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", " "))
    s = s.replace("-", "").replace(" ", "_")
    return s or "UNKNOWN"

def find_map_col(df: pd.DataFrame) -> str | None:
    for cand in ["_map", "map", "map_name"]:
        if cand in df.columns:
            return cand
    return None

def build_opening_rounds(mm: pd.DataFrame) -> pd.DataFrame:
    """Return one-row-per-round with round_key, first_kill_team, first_kill_weapon, winner_team, _map (if present)."""
    df = mm.copy()
    # round key
    if "file" in df.columns and "round" in df.columns:
        df["_round_key"] = df["file"].astype(str) + "__r__" + df["round"].astype(str)
    elif "match_id" in df.columns and "round_no" in df.columns:
        df["_round_key"] = df["match_id"].astype(str) + "__r__" + df["round_no"].astype(str)
    else:
        df["_round_key"] = df.index.astype(str)

    # prefer explicit opening_kill rows if present
    if "opening_kill" in df.columns and "opening_kill_team" in df.columns:
        op = df[df["opening_kill"].astype(bool)].copy()
        sort_cols = [c for c in ("tick", "seconds") if c in op.columns]
        if sort_cols:
            op = op.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
        else:
            op = op.groupby("_round_key", as_index=False).first()

        weapon_col = "opening_kill_weapon" if "opening_kill_weapon" in op.columns else next((c for c in ("wp_canon","wp","weapon","wp_type") if c in op.columns), None)
        winner_col = next((c for c in ("winner_team","res_match_winner","round_winner","winner") if c in op.columns), None)

        out = pd.DataFrame({
            "round_key": op["_round_key"].astype(str),
            "first_kill_team": op["opening_kill_team"].astype(str).fillna("UNKNOWN"),
            "first_kill_weapon": op[weapon_col].apply(_canon_weapon).astype(str) if weapon_col else pd.Series(["UNKNOWN"] * len(op)),
            "winner_team": op[winner_col].astype(str).fillna("UNKNOWN") if winner_col else pd.Series(["UNKNOWN"] * len(op)),
        })
        if "att_side" in op.columns:
            out["first_kill_side"] = op["att_side"].astype(str)
        if "_map" in op.columns:
            out["_map"] = op["_map"].astype(str)
        elif "map" in op.columns:
            out["_map"] = op["map"].astype(str)
        return out.reset_index(drop=True)

    # fallback earliest event per round
    sort_cols = [c for c in ("tick", "seconds") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
    else:
        df = df.groupby("_round_key", as_index=False).first()

    att_col = next((c for c in ("att_team","attacker_team","att_team_name") if c in df.columns), None)
    wp_col = next((c for c in ("wp_canon","wp","weapon","wp_type") if c in df.columns), None)
    winner_col = next((c for c in ("winner_team","res_match_winner","round_winner","winner") if c in df.columns), None)

    out = pd.DataFrame({
        "round_key": df["_round_key"].astype(str),
        "first_kill_team": df[att_col].astype(str).fillna("UNKNOWN") if att_col else pd.Series(["UNKNOWN"] * len(df)),
        "first_kill_weapon": df[wp_col].apply(_canon_weapon).astype(str).fillna("UNKNOWN") if wp_col else pd.Series(["UNKNOWN"] * len(df)),
        "winner_team": df[winner_col].astype(str).fillna("UNKNOWN") if winner_col else pd.Series(["UNKNOWN"] * len(df)),
    })
    if "att_side" in df.columns:
        out["first_kill_side"] = df["att_side"].astype(str)
    if "_map" in df.columns:
        out["_map"] = df["_map"].astype(str)
    elif "map" in df.columns:
        out["_map"] = df["map"].astype(str)
    return out.reset_index(drop=True)

def plot_map_actual_vs_pred(grp_map: pd.DataFrame, map_col: str, out_path: Path):
    """Plot actual vs predicted by map (grp_map must contain map_col, actual_win_rate, predicted_proba_mean, n)."""
    grp_map = grp_map.sort_values("actual_win_rate", ascending=False).reset_index(drop=True)
    x = np.arange(len(grp_map))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(grp_map))))
    ax.bar(x - width/2, grp_map["actual_win_rate"], width, label="Actual win rate", alpha=0.95)
    ax.bar(x + width/2, grp_map["predicted_proba_mean"], width, label="Predicted mean prob", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in grp_map[map_col].tolist()], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate / Predicted probability")
    ax.set_title(f"First-kill team: Actual vs Predicted win rate by map (maps with ≥ {MIN_MAP_OPENS} opens)")
    ax.legend()

    for i, (_, row) in enumerate(grp_map.iterrows()):
        actual_pct = row["actual_win_rate"]
        pred_pct = row["predicted_proba_mean"] if not pd.isna(row["predicted_proba_mean"]) else 0.0
        n = int(row["n"])
        ax.text(i - width/2, actual_pct + 0.02, f"{actual_pct:.1%}\n({n})", ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, pred_pct + 0.02, f"{pred_pct:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print("Saved:", out_path)

def plot_weapon_compare(merged: pd.DataFrame, out_path: Path, min_count: int = MIN_WEAPON_COUNT, top_n: int = TOP_WEAPONS):
    """Plot actual win rate vs predicted probability for top weapons."""
    if "first_kill_weapon" not in merged.columns:
        print("Skipping weapon plot: missing first_kill_weapon column.")
        return

    merged["_wp"] = merged["first_kill_weapon"].astype(str)
    # filter unknown
    merged = merged[~merged["_wp"].str.lower().isin(["unknown", "", "none", "nan"])].copy()
    if merged.empty:
        print("No known weapons found; skipping weapon plot.")
        return

    counts = merged["_wp"].value_counts()
    keep = counts[counts >= min_count].nlargest(top_n).index.tolist()
    sub = merged[merged["_wp"].isin(keep)].copy()
    if sub.empty:
        print("No weapons meet min_count; skipping weapon plot.")
        return

    sub["fk_won"] = (sub["first_kill_team"].astype(str).fillna("") == sub["winner_team"].astype(str).fillna("")).astype(int)
    # ensure pred_proba numeric
    if "pred_proba" not in sub.columns:
        sub["pred_proba"] = np.nan
    else:
        sub["pred_proba"] = pd.to_numeric(sub["pred_proba"], errors="coerce")

    grp = sub.groupby("_wp").agg(actual_win=("fk_won", "mean"), pred_proba=("pred_proba", "mean"), n=("round_key", "count")).reset_index()
    grp = grp.sort_values("actual_win", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(grp))))
    y = np.arange(len(grp))
    width = 0.35
    ax.barh(y - width/2, grp["actual_win"], height=width, label="actual winrate")
    ax.barh(y + width/2, grp["pred_proba"], height=width, label="predicted prob")
    ax.set_yticks(y)
    ax.set_yticklabels(grp["_wp"])
    ax.invert_yaxis()
    ax.set_xlabel("Winrate / predicted probability")
    ax.set_xlim(0, 1)
    ax.set_title(f"First-kill winrate by weapon (top {len(grp)} weapons, min_count={min_count})")
    ax.legend()

    for i, (act, pr, n, w) in enumerate(zip(grp["actual_win"], grp["pred_proba"], grp["n"], grp["_wp"])):
        ax.text(max(act, pr) + 0.01, i, f"{act:.1%} ({n})", va="center", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print("Saved:", out_path)

def main():
    cfg = Config()
    processed_path = Path(cfg.processed_dir) / "mm_master_clean.parquet"
    if not processed_path.exists():
        raise SystemExit(f"Processed file not found: {processed_path}. Run ETL first.")

    mm = pd.read_parquet(processed_path)
    print("Loaded", processed_path, "shape:", mm.shape)

    rounds = build_opening_rounds(mm)
    print("Built round-level openings:", rounds.shape)

    map_col = find_map_col(rounds)
    if map_col is None:
        raise SystemExit("No map column ('_map' or 'map') present in round-level data. Aborting.")

    rounds[map_col] = rounds[map_col].astype(str).str.strip().str.lower()
    bad_maps = set(["", "none", "nan", "unknown", "null", "none - t", "none - ct"])
    rounds = rounds[~rounds[map_col].isin(bad_maps)].copy()

    if "winner_team" not in rounds.columns or "first_kill_team" not in rounds.columns:
        raise SystemExit("rounds missing winner_team or first_kill_team columns.")

    try:
        predictor = Predictor(cfg=cfg)
    except FileNotFoundError:
        raise SystemExit("Model artifact not found. Train model first with: python src/models/train.py")

    preds = predictor.predict_from_rounds(rounds)
    if "pred_proba" in preds.columns:
        preds["pred_proba"] = pd.to_numeric(preds["pred_proba"], errors="coerce")
    else:
        preds["pred_proba"] = np.nan

    merged = rounds.merge(preds[["round_key","pred_proba","pred_int"]], on="round_key", how="left")
    merged["actual_win"] = (merged["first_kill_team"].astype(str).fillna("") == merged["winner_team"].astype(str).fillna("")).astype(int)

    # --- MAP plot data ---
    grp_map = merged.groupby(map_col).agg(
        actual_win_rate = ("actual_win", "mean"),
        predicted_proba_mean = ("pred_proba", "mean"),
        n = ("round_key", "count")
    ).reset_index()

    grp_map = grp_map[grp_map["n"] >= MIN_MAP_OPENS].copy()
    if grp_map.empty:
        raise SystemExit(f"No maps with >= {MIN_MAP_OPENS} openings. Lower MIN_MAP_OPENS or check data.")

    map_out = Path(cfg.processed_dir) / "figures" / "firstkill" / "firstkill_map_actual_vs_pred.png"
    map_out.parent.mkdir(parents=True, exist_ok=True)
    plot_map_actual_vs_pred(grp_map, map_col, map_out)

    # --- WEAPON plot ---
    weapon_out = Path(cfg.processed_dir) / "figures" / "firstkill" / "firstkill_winrate_by_weapon.png"
    plot_weapon_compare(merged, weapon_out, min_count=MIN_WEAPON_COUNT, top_n=TOP_WEAPONS)

    print("Done - figures saved in:", map_out.parent)

if __name__ == "__main__":
    main()
