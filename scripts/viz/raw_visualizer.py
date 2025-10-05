from pathlib import Path
import math
import sys
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

# adjust import according to your project layout
from src.config import Config

# Configurable params
TICKRATE = 64          # ticks per second
TOP_WEAPONS = 20       # number of weapons to display
MIN_MAP_OPENS_RATIO = 0.01
MIN_MAP_OPENS_ABS = 100
ROUND_MAX = 30

# --------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------
def detect_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """Return first matching column name from names or None."""
    for n in names:
        if n in df.columns:
            return n
    return None

def ensure_outdir(processed_dir: Path) -> Path:
    out = processed_dir / "figures" / "raw_visuals"
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("Saved:", path)

# --------------------------------------------------------------
# Plotting functions (pure functions operating on DataFrame)
# --------------------------------------------------------------
def plot_winrate_by_map(df: pd.DataFrame, map_col: str, open_col: str, winner_col: str, outdir: Path):
    if map_col is None:
        print("Skipping map plot: no map column found.")
        return
    df2 = df.copy()
    if "opening_won" not in df2.columns:
        df2["opening_won"] = (df2[winner_col].astype(str) == df2[open_col].astype(str)).astype(int)
    grp = df2.groupby(map_col)["opening_won"].agg(["mean", "count"]).rename(columns={"mean": "win_rate"})
    min_count = max(MIN_MAP_OPENS_ABS, int(len(df2) * MIN_MAP_OPENS_RATIO))
    grp = grp[grp["count"] >= min_count].sort_values("win_rate")
    if grp.empty:
        print("No maps meet minimum opens threshold; skipping map plot.")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(grp))))
    df_map = grp.reset_index()
    ycol = df_map.columns[0]
    if HAVE_SNS:
        sns.barplot(x="win_rate", y=ycol, data=df_map, ax=ax, palette="viridis", dodge=False)
    else:
        ax.barh(df_map[ycol], df_map["win_rate"])
    ax.set_xlabel("Opening-kill win rate")
    ax.set_title(f"Opening-kill win rate by map (min opens={min_count})")
    save_fig(fig, outdir / "opening_winrate_by_map.png")
    print("Top maps by win rate (desc):")
    print(grp.sort_values("win_rate", ascending=False).head(10))


def plot_winrate_by_weapon(df: pd.DataFrame, weapon_col: str, open_col: str, winner_col: str, outdir: Path, top_n: int = TOP_WEAPONS):
    if weapon_col is None:
        print("Skipping weapon plot: no weapon column found.")
        return
    df2 = df.copy()
    df2["_weapon_norm"] = df2[weapon_col].fillna("UNKNOWN").astype(str).str.strip().str.lower()
    unknowns = df2["_weapon_norm"].isin(["", "unknown", "nan", "none", "na"])
    df_known = df2[~unknowns].copy()
    if df_known.empty:
        print("No known weapons found (all UNKNOWN). Skipping weapon plot.")
        return
    counts = df_known["_weapon_norm"].value_counts()
    top_weapons = counts.nlargest(top_n).index.tolist()
    sub = df_known[df_known["_weapon_norm"].isin(top_weapons)].copy()
    if "opening_won" not in sub.columns:
        sub["opening_won"] = (sub[winner_col].astype(str) == sub[open_col].astype(str)).astype(int)
    grp = sub.groupby("_weapon_norm")["opening_won"].agg(["mean", "count"]).rename(columns={"mean": "win_rate"})
    grp = grp.sort_values("win_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(grp))))
    df_w = grp.reset_index()
    wcol = df_w.columns[0]
    if HAVE_SNS:
        sns.barplot(x="win_rate", y=wcol, data=df_w, ax=ax, palette="rocket", dodge=False)
    else:
        ax.barh(df_w[wcol], df_w["win_rate"])
    ax.set_xlabel("Opening-kill win rate")
    ax.set_title(f"Opening-kill win rate by weapon (top {top_n} by count)")
    save_fig(fig, outdir / "opening_winrate_by_weapon_topN.png")
    print("Top weapons by win rate (desc):")
    print(grp.head(20))


def plot_winrate_vs_round_number(df: pd.DataFrame, round_col: str, open_col: str, winner_col: str, outdir: Path, round_max: int = ROUND_MAX):
    if round_col is None:
        print("Skipping round-number correlation: no round column found.")
        return
    df2 = df.copy()
    df2["_round_num"] = pd.to_numeric(df2[round_col], errors="coerce")
    rn = df2[~df2["_round_num"].isna()].copy()
    if rn.empty:
        print("No numeric round values; skipping round correlation plot.")
        return
    if "opening_won" not in rn.columns:
        rn["opening_won"] = (rn[winner_col].astype(str) == rn[open_col].astype(str)).astype(int)
    round_grp = rn.groupby("_round_num")["opening_won"].agg(["mean", "count"]).reset_index().sort_values("_round_num")
    full_index = pd.Series(range(1, round_max + 1), name="_round_num")
    round_grp = round_grp.set_index("_round_num").reindex(full_index).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(round_grp["_round_num"], round_grp["mean"], marker="o", linewidth=1)
    ax.set_xlabel("Round number")
    ax.set_xlim(1, round_max)
    ax.set_xticks(list(range(1, round_max + 1)))
    ax.set_ylabel("Opening-kill win rate")
    ax.set_title(f"Opening-kill win rate vs round number (1..{round_max})")
    save_fig(fig, outdir / "opening_winrate_vs_round_number.png")
    print("Saved round-number plot.")


# --------------------------------------------------------------
# Main wiring: load preferred parquet and call the 4 plotters
# --------------------------------------------------------------
def main():
    cfg = Config()
    processed = cfg.processed_dir
    # Prioritize event-level mm_master_clean.parquet (per-event duels)
    mm_path = processed / "mm_master_clean.parquet"
    merged_path = processed / "merged_professional.parquet"

    if mm_path.exists():
        print("Loading event-level data from", mm_path)
        df = pd.read_parquet(mm_path)
        data_source = "mm_master_clean"
    elif merged_path.exists():
        print("mm_master_clean.parquet not found. Falling back to merged_professional.parquet (may be aggregated).")
        df = pd.read_parquet(merged_path)
        data_source = "merged_professional"
    else:
        print("No processed parquet found. Please run the ETL to produce mm_master_clean.parquet or merged_professional.parquet under data/processed.")
        print("Checked:", mm_path, merged_path)
        sys.exit(1)

    print("Loaded shape:", df.shape)
    print("Columns available:", df.columns.tolist()[:80])

    # Detect columns (flexible names)
    col_open_team = detect_col(df, ["opening_kill_team", "opening_team", "att_team", "attacker_team", "att_team"])
    col_winner = detect_col(df, ["winner_team", "round_winner", "winning_team", "winner", "res_match_winner", "res_map_winner"])
    col_weapon = detect_col(df, ["first_kill_weapon", "opening_kill_weapon", "wp", "weapon"])
    col_tick = detect_col(df, ["tick", "time", "seconds"])
    col_round = detect_col(df, ["round", "round_no", "round_number", "round_idx"])
    col_map = detect_col(df, ["_map", "map", "map_name"])

    # If winner/opening not found, try heuristics for mm_master (opening_kill may be "opening_kill" boolean)
    if col_open_team is None and "opening_kill" in df.columns:
        # if we have 'att_team' and 'victim' and opening_kill boolean, derive opening_team by selecting att_team where opening_kill True
        # But for plotting we need per-row opening_team -- many datasets already have explicit opening columns, so keep simple:
        print("Warning: no explicit opening-team column found. Expect limited ability to compute opening-team-based stats.")
    if col_winner is None:
        # try winner-like columns prefixed with 'res_' in merged_professional merges
        cands = [c for c in df.columns if ("winner" in c.lower() or "result" in c.lower())]
        if cands:
            print("No direct winner column found, candidate winner-like columns:", cands[:6])
            col_winner = cands[0]
            print("Using", col_winner, "as winner column (best guess).")
        else:
            print("No winner-like column found; cannot compute opening win rates.")
            sys.exit(1)

    outdir = ensure_outdir(processed)

    # Create opening_won if possible: require both open_team and winner columns
    # If opening_team not found but dataset is mm_master_clean and has opening_kill boolean and attacker team, we can create opening_team
    if col_open_team is None:
        # Attempt to create opening_team on event-level df if 'opening_kill' boolean & 'att_team' available
        if data_source == "mm_master_clean" and "opening_kill" in df.columns and "att_team" in df.columns:
            print("Creating opening_team column from 'opening_kill' and 'att_team' (event-level heuristic).")
            df = df.copy()
            # for only rows with opening_kill==True, set opening_team = att_team, else NaN (these visuals will count only explicit opening events)
            df["opening_team_generated"] = df.apply(lambda r: r["att_team"] if (r.get("opening_kill") in (True, "True", 1, "1")) else None, axis=1)
            col_open_team = "opening_team_generated"
        else:
            print("No opening-team column and cannot derive it; map/weapon/time-round plots that require opening-team will be skipped.")
    else:
        # ensure opening_won column exists
        if "opening_won" not in df.columns:
            df = df.copy()
            df["opening_won"] = (df[col_winner].astype(str) == df[col_open_team].astype(str)).astype(int)

    # Now call the four plotters (some may be skipped depending on availability)
    plot_winrate_by_map(df, map_col=col_map, open_col=col_open_team, winner_col=col_winner, outdir=outdir)
    plot_winrate_by_weapon(df, weapon_col=col_weapon, open_col=col_open_team, winner_col=col_winner, outdir=outdir)
    plot_winrate_vs_round_number(df, round_col=col_round, open_col=col_open_team, winner_col=col_winner, outdir=outdir, round_max=ROUND_MAX)

    print("All done. Look in:", outdir)


if __name__ == "__main__":
    main()
