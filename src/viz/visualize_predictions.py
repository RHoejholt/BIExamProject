# scripts/visualize_opening_analysis.py
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional seaborn for prettier default visuals; fallback to matplotlib if unavailable
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

from src.config import Config

# ---------- Configurable parameters ----------
TICKRATE = 64           # ticks per second (change to 128 if your data uses 128hz)
TOP_WEAPONS = 20        # top N weapons to display
MIN_MAP_OPENS_RATIO = 0.01  # keep maps with at least this fraction of opens
MIN_MAP_OPENS_ABS = 100     # or at least this absolute number (use max of both)
ROUND_MAX = 30          # display rounds 1..ROUND_MAX on the x-axis
# ------------------------------------------------

def find_predictions_csv(cfg: Config) -> Path:
    cands = [
        cfg.project_root / "models" / "predictions_with_context.csv",
        cfg.processed_dir / "predictions_with_context.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    found = list(cfg.project_root.glob("**/predictions_with_context*.csv"))
    return found[0] if found else None

def detect_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def ensure_output_dir(cfg: Config):
    out = cfg.processed_dir / "figures" / "opening_analysis"
    out.mkdir(parents=True, exist_ok=True)
    return out

def safe_read_predictions():
    cfg = Config()
    p = find_predictions_csv(cfg)
    if p is None:
        print("Predictions CSV not found. Run the pipeline that creates predictions_with_context.csv first.")
        sys.exit(1)
    print("Loading predictions from:", p)
    return pd.read_csv(p), cfg

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved:", path)

def main():
    df, cfg = safe_read_predictions()
    outdir = ensure_output_dir(cfg)

    # detect useful columns with fallback names
    col_open_team = detect_col(df, ["opening_kill_team", "opening_team", "att_team", "attacker_team"])
    col_winner = detect_col(df, ["winner_team", "round_winner", "winning_team", "winner"])
    col_weapon = detect_col(df, ["first_kill_weapon", "opening_kill_weapon", "weapon"])
    col_tick = detect_col(df, ["tick", "time", "seconds"])
    col_round = detect_col(df, ["round", "round_no", "round_number", "round_idx"])
    col_map = detect_col(df, ["_map", "map", "map_name"])

    if col_open_team is None or col_winner is None:
        print("Required columns missing: need opening-team and winner-team columns.")
        print("Found columns:", df.columns.tolist())
        sys.exit(1)

    # create boolean indicator: did the opening-team go on to win the round?
    df["opening_won"] = (df[col_winner].astype(str) == df[col_open_team].astype(str)).astype(int)

    # ------------------ 1) Win rate by map ------------------
    if col_map is not None:
        grp_map = df.groupby(col_map)["opening_won"].agg(["mean", "count"]).rename(columns={"mean":"win_rate"})
        min_count = max(MIN_MAP_OPENS_ABS, int(len(df) * MIN_MAP_OPENS_RATIO))
        grp_map = grp_map[grp_map["count"] >= min_count].sort_values("win_rate")
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(grp_map))))
        df_map = grp_map.reset_index()
        ycol_map = df_map.columns[0]
        if HAVE_SNS:
            sns.barplot(x="win_rate", y=ycol_map, data=df_map, ax=ax, palette="viridis", dodge=False)
        else:
            ax.barh(df_map[ycol_map], df_map["win_rate"])
        ax.set_xlabel("Opening-kill win rate")
        ax.set_title(f"Opening-kill win rate by map (min opens={min_count})")
        out_path = outdir / "opening_winrate_by_map.png"
        save_fig(fig, out_path)
        print("Top maps by win rate (desc):")
        print(grp_map.sort_values("win_rate", ascending=False).head(10))
    else:
        print("Skipping map plot: no map column found.")

    # ------------------ 2) Win rate by weapon (filter UNKNOWN + sort desc) ------------------
    if col_weapon is not None:
        # normalize weapon strings and filter unknowns
        w_series = df[col_weapon].fillna("UNKNOWN").astype(str).str.strip().str.lower()
        df["_weapon_norm"] = w_series
        # treat various unknown markers as unknown
        unknown_mask = df["_weapon_norm"].isin(["", "unknown", "nan", "none", "na"])
        df_known = df[~unknown_mask].copy()
        if df_known.empty:
            print("No known weapons found (all UNKNOWN). Skipping weapon plot.")
        else:
            counts = df_known["_weapon_norm"].value_counts()
            top_weapons = counts.nlargest(TOP_WEAPONS).index.tolist()
            sub = df_known[df_known["_weapon_norm"].isin(top_weapons)].copy()
            grp_w = sub.groupby("_weapon_norm")["opening_won"].agg(["mean","count"]).rename(columns={"mean":"win_rate"})
            # sort by descending win rate (most effective first)
            grp_w = grp_w.sort_values("win_rate", ascending=False)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(grp_w))))
            df_w = grp_w.reset_index()
            wcol = df_w.columns[0]
            if HAVE_SNS:
                sns.barplot(x="win_rate", y=wcol, data=df_w, ax=ax, palette="rocket", dodge=False)
            else:
                ax.barh(df_w[wcol], df_w["win_rate"])
            ax.set_xlabel("Opening-kill win rate")
            ax.set_title(f"Opening-kill win rate by weapon (top {TOP_WEAPONS} by count, unknowns excluded)")
            out_path = outdir / "opening_winrate_by_weapon_topN.png"
            save_fig(fig, out_path)
            print("Top weapons by win rate (desc):")
            print(grp_w.head(20))
    else:
        print("Skipping weapon plot: no weapon column found.")

    # ------------------ 3) Win rate vs in-game time (tick -> seconds) ------------------
    if col_tick is not None:
        df["_tick_num"] = pd.to_numeric(df[col_tick], errors="coerce")
        tick_series = df["_tick_num"].dropna()
        if tick_series.empty:
            print("No numeric tick values; skipping tick plot.")
        else:
            # convert ticks -> seconds using tickrate (configurable)
            df["_sec"] = df["_tick_num"] / float(TICKRATE)
            # bin by seconds (you can tune bins or use quantiles)
            n_bins = 60
            max_sec = math.ceil(df["_sec"].max())
            bins = np.linspace(0, max_sec, n_bins + 1)
            df["_sec_bin"] = pd.cut(df["_sec"], bins=bins, include_lowest=True, labels=False)
            tick_grp = df.groupby("_sec_bin")["opening_won"].agg(["mean","count"]).reset_index().dropna()
            # midpoints for plotting
            tick_grp["sec_mid"] = tick_grp["_sec_bin"].apply(lambda i: (bins[int(i)] + bins[int(i)+1]) / 2 if not pd.isna(i) else np.nan)
            tick_grp = tick_grp.sort_values("sec_mid")
            tick_grp["win_rate_smooth"] = tick_grp["mean"].rolling(window=3, min_periods=1, center=True).mean()
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(tick_grp["sec_mid"], tick_grp["win_rate_smooth"], marker="o", linewidth=1)
            ax.set_xlabel("Seconds (tick -> seconds using tickrate={})".format(TICKRATE))
            ax.set_ylabel("Opening-kill win rate (smoothed)")
            ax.set_title("Opening-kill win rate vs in-game time (seconds)")
            out_path = outdir / "opening_winrate_vs_seconds.png"
            save_fig(fig, out_path)
            print("Saved:", out_path)
    else:
        print("Skipping tick plot: no tick/time column found.")

    # ------------------ 4) Correlation: opening-win rate vs round number (x axis 1..ROUND_MAX) --------------
    if col_round is not None:
        df["_round_num"] = pd.to_numeric(df[col_round], errors="coerce")
        rn = df[~df["_round_num"].isna()].copy()
        if rn.empty:
            print("No numeric round values; skipping round correlation plot.")
        else:
            # compute win rate per round number across all matches
            round_grp = rn.groupby("_round_num")["opening_won"].agg(["mean","count"]).reset_index().sort_values("_round_num")
            # restrict to rounds 1..ROUND_MAX inclusive
            full_index = pd.Series(range(1, ROUND_MAX + 1), name="_round_num")
            round_grp = round_grp.set_index("_round_num").reindex(full_index).reset_index()
            # round_grp now has columns: _round_num, mean (win_rate), count (may be NaN)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(round_grp["_round_num"], round_grp["mean"], marker="o", linewidth=1)
            ax.set_xlabel("Round number")
            ax.set_xlim(1, ROUND_MAX)
            ax.set_xticks(list(range(1, ROUND_MAX + 1)))
            ax.set_ylabel("Opening-kill win rate")
            ax.set_title("Opening-kill win rate vs round number (1..{})".format(ROUND_MAX))
            out_path = outdir / "opening_winrate_vs_round_number.png"
            save_fig(fig, out_path)
            print("Saved:", out_path)
    else:
        print("Skipping round-number correlation: no round column found.")

    print("Done. Figures saved to:", outdir)

if __name__ == "__main__":
    main()
