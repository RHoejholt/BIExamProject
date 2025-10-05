# src/visualization/firstkill_viz.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

def ensure_outdir(processed_dir: Path) -> Path:
    out = processed_dir / "figures" / "firstkill"
    out.mkdir(parents=True, exist_ok=True)
    return out

def plot_map_side_heatmaps(rounds: pd.DataFrame, processed_dir: Path):
    """
    rounds: DataFrame must contain columns:
      - _map, first_kill_side, actual_win (0/1), pred_proba (0..1)
    Produces two side-by-side heatmaps: actual winrate and mean predicted prob.
    """
    out = ensure_outdir(processed_dir)
    if "_map" not in rounds.columns or "first_kill_side" not in rounds.columns:
        print("plot_map_side_heatmaps: missing _map or first_kill_side; skipping")
        return None

    grp = rounds.groupby(["_map", "first_kill_side"]).agg(
        actual_winrate=("actual_win", "mean"),
        pred_proba_mean=("pred_proba", "mean"),
        n=("round_key", "count")
    ).reset_index()

    if grp.empty:
        print("plot_map_side_heatmaps: grouping resulted in empty DataFrame; skipping")
        return None

    pivot_act = grp.pivot(index="_map", columns="first_kill_side", values="actual_winrate")
    pivot_pred = grp.pivot(index="_map", columns="first_kill_side", values="pred_proba_mean")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.5 * len(pivot_act))))
    if HAVE_SNS:
        sns.heatmap(pivot_act, annot=True, fmt=".2f", ax=axes[0], cmap="vlag", cbar_kws={"label": "actual winrate"})
        sns.heatmap(pivot_pred, annot=True, fmt=".2f", ax=axes[1], cmap="vlag", cbar_kws={"label": "predicted prob"})
    else:
        im0 = axes[0].imshow(pivot_act.fillna(0.0).values, aspect="auto", cmap="vlag")
        axes[0].set_yticks(np.arange(len(pivot_act.index))); axes[0].set_yticklabels(pivot_act.index)
        axes[0].set_xticks(np.arange(len(pivot_act.columns))); axes[0].set_xticklabels(pivot_act.columns)
        axes[0].set_title("Actual winrate by map × side")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(pivot_pred.fillna(0.0).values, aspect="auto", cmap="vlag")
        axes[1].set_yticks(np.arange(len(pivot_pred.index))); axes[1].set_yticklabels(pivot_pred.index)
        axes[1].set_xticks(np.arange(len(pivot_pred.columns))); axes[1].set_xticklabels(pivot_pred.columns)
        axes[1].set_title("Predicted mean probability by map × side")
        fig.colorbar(im1, ax=axes[1])

    axes[0].set_title("Actual winrate of team with opening kill (by map × side)")
    axes[1].set_title("Predicted mean probability (by map × side)")
    plt.tight_layout()
    outpath = out / "firstkill_winrate_map_side_heatmaps.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print("Saved:", outpath)
    return outpath

def plot_weapon_compare(rounds: pd.DataFrame, processed_dir: Path, top_n: int = 20):
    """
    rounds: DataFrame must contain columns:
      - first_kill_weapon, actual_win (0/1), pred_proba (0..1)
    Produces a horizontal grouped bar chart for top_n weapons by count.
    """
    out = ensure_outdir(processed_dir)
    if "first_kill_weapon" not in rounds.columns:
        print("plot_weapon_compare: missing first_kill_weapon; skipping")
        return None

    rounds["_wp"] = rounds["first_kill_weapon"].astype(str)
    counts = rounds["_wp"].value_counts()
    top = counts.nlargest(top_n).index.tolist()
    sub = rounds[rounds["_wp"].isin(top)].copy()
    if sub.empty:
        print("plot_weapon_compare: no rows after selecting top weapons; skipping")
        return None

    grp = sub.groupby("_wp").agg(actual_win=("actual_win", "mean"), pred_proba=("pred_proba", "mean"), n=("round_key","count")).reset_index()
    grp = grp.sort_values("actual_win", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(grp))))
    y = np.arange(len(grp))
    width = 0.35
    ax.barh(y - width/2, grp["actual_win"], height=width, label="actual winrate")
    ax.barh(y + width/2, grp["pred_proba"], height=width, label="predicted prob")
    ax.set_yticks(y)
    ax.set_yticklabels(grp["_wp"])
    ax.invert_yaxis()
    ax.set_xlabel("Winrate / predicted probability")
    ax.set_title(f"First-kill winrate by weapon (top {top_n}) — actual vs predicted")
    ax.legend()
    plt.tight_layout()
    outpath = out / "firstkill_winrate_by_weapon_topN.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print("Saved:", outpath)
    return outpath
