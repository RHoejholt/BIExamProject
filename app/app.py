import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import re

st.set_page_config(page_title="First-kill Predictor (interactive)", layout="wide")

# ---------- Paths ----------
PROJECT_ROOT = Path.cwd()
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "mm_master_clean.parquet"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "mm_firstkill_binary.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "mm_firstkill_binary_metrics.json"

# ---------- Helpers ----------
def _canon_weapon(w):
    if pd.isna(w) or w is None:
        return "UNKNOWN"
    s = str(w).lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))
    s = s.replace("-", "").replace(" ", "_")
    return s or "UNKNOWN"

def _canon_map(m):
    if pd.isna(m) or m is None:
        return "unknown"
    s = str(m).lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = s.strip("_")
    return s or "unknown"

def _canon_side(s):
    if pd.isna(s) or s is None:
        return "unknown"
    x = str(s).lower().strip()
    if x in ("t", "terrorist", "terrorists"):
        return "terrorist"
    if x in ("ct", "counterterrorist", "counter_terrorist", "counterterrorists"):
        return "counterterrorist"
    return re.sub(r"[^a-z0-9_]+", "_", x)

def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run Trainer.train_binary() first.")
    data = joblib.load(str(path))
    if not isinstance(data, dict) or "model" not in data or "feature_columns" not in data:
        raise ValueError("Artifact format unexpected. Expected dict with keys 'model' and 'feature_columns'.")
    return data

def find_matching_column(prefix_candidates, value, artifact_cols):
    if value is None:
        return None
    cleaned = _canon_weapon(value) if prefix_candidates and prefix_candidates[0].startswith("wp") else (_canon_map(value) if prefix_candidates and prefix_candidates[0].startswith("map") else _canon_side(value))
    for p in prefix_candidates:
        cand = f"{p}{cleaned}"
        if cand in artifact_cols:
            return cand
    for c in artifact_cols:
        low = c.lower()
        if cleaned in low:
            for p in prefix_candidates:
                if p in low:
                    return c
    return None

def build_feature_row_from_inputs(artifact_cols, weapon, map_name=None, side=None, numeric_inputs: dict=None):
    numeric_inputs = numeric_inputs or {}
    row = {c: 0.0 for c in artifact_cols}

    # weapon
    wp_clean = _canon_weapon(weapon)
    wp_exact = f"wp_{wp_clean}"
    if wp_exact in row:
        row[wp_exact] = 1.0
    else:
        matched = find_matching_column(["wp_"], weapon, artifact_cols)
        if matched:
            row[matched] = 1.0

    # map
    if map_name:
        map_exact = f"map_{_canon_map(map_name)}"
        if map_exact in row:
            row[map_exact] = 1.0
        else:
            matched = find_matching_column(["map_"], map_name, artifact_cols)
            if matched:
                row[matched] = 1.0

    # side
    if side:
        side_clean = _canon_side(side)
        side_exact = f"side_{side_clean}"
        if side_exact in row:
            row[side_exact] = 1.0
        else:
            matched = find_matching_column(["side_"], side, artifact_cols)
            if matched:
                row[matched] = 1.0

    # numeric features
    for k, v in numeric_inputs.items():
        if k in row:
            try:
                row[k] = float(v)
            except Exception:
                row[k] = 0.0

    df = pd.DataFrame([row], columns=artifact_cols)
    return df.fillna(0.0)

def load_processed_rounds():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_PATH} missing. Run ETL first.")
    mm = pd.read_parquet(PROCESSED_PATH)
    df = mm.copy()
    if "file" in df.columns and "round" in df.columns:
        df["_round_key"] = df["file"].astype(str) + "__r__" + df["round"].astype(str)
    elif "match_id" in df.columns and "round" in df.columns:
        df["_round_key"] = df["match_id"].astype(str) + "__r__" + df["round"].astype(str)
    else:
        df["_round_key"] = df.index.astype(str)

    if "opening_kill" in df.columns and "opening_kill_team" in df.columns:
        op = df[df["opening_kill"].astype(bool)].copy()
        sort_cols = [c for c in ("tick", "seconds") if c in op.columns]
        if sort_cols:
            op = op.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
        else:
            op = op.groupby("_round_key", as_index=False).first()
        wp_col = "opening_kill_weapon" if "opening_kill_weapon" in op.columns else next((c for c in ("wp_canon","wp","weapon","wp_type") if c in op.columns), None)
        op["first_kill_weapon"] = op[wp_col].astype(str).fillna("UNKNOWN").apply(_canon_weapon) if wp_col else "UNKNOWN"
        op["first_kill_team"] = op["opening_kill_team"].astype(str).fillna("UNKNOWN") if "opening_kill_team" in op.columns else op.get("att_team", pd.Series(["UNKNOWN"]*len(op)))
        win_col = next((c for c in ("winner_team","res_match_winner","round_winner","res_map_winner") if c in op.columns), None)
        op["winner_team"] = op[win_col].astype(str).fillna("UNKNOWN") if win_col else "UNKNOWN"
        op["first_kill_side"] = op.get("att_side", pd.Series(["UNKNOWN"]*len(op))).astype(str) if "att_side" in op.columns else op.get("first_kill_side", pd.Series(["UNKNOWN"]*len(op)))
        if "_map" in op.columns:
            op["_map"] = op["_map"].astype(str)
        return op.reset_index(drop=True)

    sort_cols = [c for c in ("tick","seconds") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
    else:
        df = df.groupby("_round_key", as_index=False).first()

    wp_col = next((c for c in ("wp_canon","wp","weapon","wp_type") if c in df.columns), None)
    df["first_kill_weapon"] = df[wp_col].astype(str).fillna("UNKNOWN").apply(_canon_weapon) if wp_col else "UNKNOWN"
    att_col = next((c for c in ("att_team","attacker_team","att_team_name") if c in df.columns), None)
    df["first_kill_team"] = df[att_col].astype(str).fillna("UNKNOWN") if att_col else "UNKNOWN"
    win_col = next((c for c in ("winner_team","res_match_winner","round_winner","res_map_winner") if c in df.columns), None)
    df["winner_team"] = df[win_col].astype(str).fillna("UNKNOWN") if win_col else "UNKNOWN"
    if "_map" in df.columns:
        df["_map"] = df["_map"].astype(str)
    if "att_side" in df.columns:
        df["first_kill_side"] = df["att_side"].astype(str)
    else:
        df["first_kill_side"] = "UNKNOWN"
    return df.reset_index(drop=True)

# ---------- UI ----------
st.title("Interactive first-kill predictor")
st.markdown(
    "Predict whether the team with the opening kill wins the round. Choose map/side/weapon."
)

# load artifact
artifact = None
artifact_missing = False
try:
    artifact = load_artifact(ARTIFACT_PATH)
except Exception as e:
    artifact_missing = True
    st.warning(f"Artifact could not be loaded: {e}")

# load processed rounds
rounds = None
proc_missing = False
try:
    rounds = load_processed_rounds()
except Exception as e:
    proc_missing = True
    st.warning(f"Processed rounds could not be loaded: {e}")

# layout
left, middle, right = st.columns([3,1,3])

with left:
    st.header("Inputs (single prediction)")
    map_choices = sorted(rounds["_map"].fillna("UNKNOWN").unique().tolist()) if rounds is not None else ["unknown"]
    map_choice = st.selectbox("Map", options=map_choices)
    side_choices = sorted(rounds["first_kill_side"].fillna("UNKNOWN").unique().tolist()) if rounds is not None else ["terrorist","counterterrorist","UNKNOWN"]
    side_choice = st.selectbox("Side (attacker side)", options=side_choices)

    top_weapons = rounds["first_kill_weapon"].value_counts().nlargest(30).index.tolist() if rounds is not None else []
    weapon_choice = st.selectbox("Weapon (or type a new one)", options=["<use text input>"] + top_weapons)
    weapon_text = st.text_input("Weapon (text) — canonicalized", value=("" if weapon_choice == "<use text input>" else weapon_choice))
    if weapon_choice != "<use text input>":
        weapon_text = weapon_choice

    st.write("Optional numeric features (if present in artifact):")
    numeric_options = [c for c in (artifact.get("feature_columns", []) if artifact else []) if c in ("ct_eq_val","t_eq_val","avg_match_rank","hp_dmg","arm_dmg")]
    numeric_values = {}
    for n in numeric_options:
        numeric_values[n] = st.number_input(n, value=0.0, step=1.0)

    st.markdown("---")
    if not artifact_missing:
        predict_button = st.button("Predict probability (single)")
    else:
        st.button("Predict probability (single)", disabled=True)

with middle:
    # intentionally minimal: do not display long model dump or the "bulk" UI.
    # show a concise model-loaded indicator only
    st.header("Model status")
    if artifact_missing:
        st.info("Model artifact not found. Place model at: " + str(ARTIFACT_PATH))
    else:
        st.success("Model loaded.")

with right:
    st.header("Prediction & historical context")
    pred_placeholder = st.empty()
    hist_weapon_plot = st.empty()
    hist_map_side_plot = st.empty()

# ---------- Single prediction ----------
if (not artifact_missing) and ('predict_button' in locals() and predict_button):
    feat_cols = artifact["feature_columns"]
    weapon_canon = _canon_weapon(weapon_text)
    numeric_inputs = {k:v for k,v in numeric_values.items()}
    X_row = build_feature_row_from_inputs(feat_cols, weapon_canon, map_name=map_choice, side=side_choice, numeric_inputs=numeric_inputs)
    model = artifact["model"]
    try:
        proba = float(model.predict_proba(X_row)[0,1])
    except Exception:
        try:
            proba = float(model.predict(X_row)[0])
        except Exception:
            proba = None

    pred_placeholder.subheader("Model prediction")
    if proba is not None:
        pred_placeholder.metric(label="P(team with opening kill wins)", value=f"{proba*100:.1f}%")
    else:
        pred_placeholder.warning("Model could not produce a probability for this input.")

    st.write("Feature vector (non-zero entries):")
    nonzero = {k:v for k,v in X_row.iloc[0].to_dict().items() if float(v) != 0.0}
    st.json(nonzero)

    # historical context charts
    if rounds is not None:
        hist_weapon = rounds.copy()
        hist_weapon["actual_win"] = (hist_weapon["first_kill_team"].astype(str).fillna("") == hist_weapon["winner_team"].astype(str).fillna("")).astype(int)
        w = weapon_canon
        grp = hist_weapon.groupby("first_kill_weapon").agg(actual_winrate=("actual_win","mean"), count=("first_kill_weapon","count")).reset_index()
        grp = grp[grp["count"]>=20].sort_values("count", ascending=False).head(20)
        if w not in grp["first_kill_weapon"].values:
            extra = hist_weapon[hist_weapon["first_kill_weapon"]==w].groupby("first_kill_weapon").agg(actual_winrate=("actual_win","mean"), count=("first_kill_weapon","count")).reset_index()
            if not extra.empty:
                grp = pd.concat([extra, grp], ignore_index=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(grp))))
        ax.barh(grp["first_kill_weapon"], grp["actual_winrate"], color="#2b8cbe", alpha=0.8)
        if w in grp["first_kill_weapon"].values and proba is not None:
            idx = grp.index[grp["first_kill_weapon"]==w].tolist()
            if idx:
                y_pos = idx[0]
                ax.plot([proba], [y_pos], marker="D", color="orange", markersize=8, label="model prob")
                ax.legend()
        ax.set_xlabel("Actual winrate (first-kill team)")
        ax.set_xlim(0,1)
        ax.set_title("Top weapons — actual first-kill winrate (historical)")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        hist_weapon_plot.pyplot(fig)

        ms = hist_weapon.groupby(["_map","first_kill_side"]).agg(actual_winrate=("actual_win","mean"), n=("first_kill_side","count")).reset_index()
        ms_sub = ms[ms["_map"]==map_choice]
        if not ms_sub.empty:
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.bar(ms_sub["first_kill_side"], ms_sub["actual_winrate"], color=["#f03b20","#2ca25f"])
            ax2.set_ylim(0,1)
            ax2.set_ylabel("Actual winrate (first-kill team)")
            ax2.set_title(f"Actual winrate by side on map {map_choice}")
            hist_map_side_plot.pyplot(fig2)
        else:
            hist_map_side_plot.write("No map×side historical data for selected map.")

# No bulk-predict UI and no caption are included.

