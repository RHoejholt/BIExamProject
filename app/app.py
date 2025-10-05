# streamlit_predict_app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import math

st.set_page_config(page_title="First-kill Predictor (interactive)", layout="wide")

# ---------- Helpers ----------
PROJECT_ROOT = Path.cwd()
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "mm_master_clean.parquet"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "mm_firstkill_binary.joblib"

def _canon_weapon(w):
    if pd.isna(w) or w is None:
        return "UNKNOWN"
    s = str(w).lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))
    s = s.replace("-", "").replace(" ", "_")
    return s or "UNKNOWN"

def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run Trainer.train_binary() first.")
    data = joblib.load(str(path))
    if not isinstance(data, dict) or "model" not in data or "feature_columns" not in data:
        raise ValueError("Artifact format unexpected. Expected dict with keys 'model' and 'feature_columns'.")
    return data

def build_feature_row_from_inputs(artifact_cols, weapon, map_name=None, side=None, numeric_inputs: dict=None):
    """
    Create a single-row DataFrame with columns = artifact_cols filled with 0/NA,
    then set the columns corresponding to weapon/map/side to 1 if present.
    numeric_inputs is a dict mapping numeric column names to numeric values (ct_eq_val etc).
    """
    numeric_inputs = numeric_inputs or {}
    row = {c: 0.0 for c in artifact_cols}
    # weapon feature naming convention: wp_<weaponcanon> or prefix "wp_"
    wp_col = f"wp_{_canon_weapon(weapon)}"
    if wp_col in row:
        row[wp_col] = 1.0
    # try variants if training used other prefix forms (defensive)
    elif f"wp_{weapon}" in row:
        row[f"wp_{weapon}"] = 1.0
    # map
    if map_name:
        map_col = f"map_{str(map_name)}"
        if map_col in row:
            row[map_col] = 1.0
    # side
    if side:
        side_col = f"side_{str(side)}"
        if side_col in row:
            row[side_col] = 1.0
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
    # Use Trainer-like logic to build "round-level" dataset for historical stats.
    # We'll fallback to just using earliest event per round if explicit opening_kill rows exist or not.
    mm = pd.read_parquet(PROCESSED_PATH)
    df = mm.copy()
    # build a simple round key
    if "file" in df.columns and "round" in df.columns:
        df["_round_key"] = df["file"].astype(str) + "__r__" + df["round"].astype(str)
    elif "match_id" in df.columns and "round" in df.columns:
        df["_round_key"] = df["match_id"].astype(str) + "__r__" + df["round"].astype(str)
    else:
        df["_round_key"] = df.index.astype(str)
    # prefer explicit opening_kill rows if present
    if "opening_kill" in df.columns and "opening_kill" in df.columns:
        op = df[df["opening_kill"].astype(bool)].copy()
        sort_cols = [c for c in ("tick", "seconds") if c in op.columns]
        if sort_cols:
            op = op.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
        else:
            op = op.groupby("_round_key", as_index=False).first()
        # canonical weapon column
        wp_col = "opening_kill_weapon" if "opening_kill_weapon" in op.columns else next((c for c in ("wp_canon","wp","weapon","wp_type") if c in op.columns), None)
        op["first_kill_weapon"] = op[wp_col].astype(str).fillna("UNKNOWN").apply(_canon_weapon) if wp_col else "UNKNOWN"
        # first_kill_team
        if "opening_kill_team" in op.columns:
            op["first_kill_team"] = op["opening_kill_team"].astype(str).fillna("UNKNOWN")
        elif "att_team" in op.columns:
            op["first_kill_team"] = op["att_team"].astype(str).fillna("UNKNOWN")
        else:
            op["first_kill_team"] = "UNKNOWN"
        # winner team resolution
        win_col = next((c for c in ("winner_team","res_match_winner","round_winner","res_map_winner") if c in op.columns), None)
        if win_col:
            op["winner_team"] = op[win_col].astype(str).fillna("UNKNOWN")
        else:
            op["winner_team"] = "UNKNOWN"
        # side
        if "att_side" in op.columns:
            op["first_kill_side"] = op["att_side"].astype(str)
        else:
            op["first_kill_side"] = op.get("first_kill_side", pd.Series(["UNKNOWN"]*len(op)))
        # map
        if "_map" in op.columns:
            op["_map"] = op["_map"].astype(str)
        return op.reset_index(drop=True)
    # fallback: earliest event per round
    sort_cols = [c for c in ("tick","seconds") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).groupby("_round_key", as_index=False).first()
    else:
        df = df.groupby("_round_key", as_index=False).first()
    # weapon
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
st.markdown("Use a trained artifact to predict whether the team with the opening kill wins the round.  \
            Choose map/side/weapon or try bulk predict top weapons. Historical rates are shown for context.")

# load artifact
artifact_missing = False
artifact = None
try:
    artifact = load_artifact(ARTIFACT_PATH)
except Exception as e:
    artifact_missing = True
    st.warning(f"Artifact could not be loaded: {e}")

# load processed round dataset for historical stats
proc_missing = False
rounds = None
try:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_PATH} missing")
    rounds = load_processed_rounds()
except Exception as e:
    proc_missing = True
    st.warning(f"Processed data could not be loaded: {e}")

# UI columns
left, middle, right = st.columns([3,2,3])

with left:
    st.header("Inputs (single prediction)")
    map_choice = st.selectbox("Map", options=(sorted(rounds["_map"].unique().tolist()) if rounds is not None else ["UNKNOWN"]))
    side_choice = st.selectbox("Side (attacker side)", options=(["terrorist","counterterrorist","T","CT","UNKNOWN"] if rounds is None else sorted(rounds["first_kill_side"].fillna("UNKNOWN").unique().tolist())))
    # weapon: allow picking top weapons or free text
    top_weapons = rounds["first_kill_weapon"].value_counts().nlargest(30).index.tolist() if rounds is not None else []
    weapon_choice = st.selectbox("Weapon (or type a new one below)", options=["<use text input>"] + top_weapons)
    weapon_text = st.text_input("Weapon (text) — will be canonicalized", value=("" if weapon_choice != "<use text input>" else "usp"))
    if weapon_choice != "<use text input>":
        weapon_text = weapon_choice
    # numeric optional inputs
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
    st.header("Model & artifact")
    if artifact_missing:
        st.info("Model artifact not found. Train first or place model at:\n" + str(ARTIFACT_PATH))
    else:
        st.write("Model artifact loaded.")
        st.write("Model type:", type(artifact["model"]))
        st.write("Number of feature columns:", len(artifact["feature_columns"]))
        st.write("Examples of feature columns (first 40):")
        st.write(artifact["feature_columns"][:40])

        # let user inspect raw metrics file if present
        metrics_file = PROJECT_ROOT / "models" / "mm_firstkill_binary_metrics.json"
        if metrics_file.exists():
            st.write("Training metrics (from model run):")
            st.json(json.loads(metrics_file.read_text()))

with right:
    st.header("Prediction & historical context")
    pred_placeholder = st.empty()
    hist_weapon_plot = st.empty()
    hist_map_side_plot = st.empty()

# ---------- Single prediction action ----------
if (not artifact_missing) and ('predict_button' in locals() and predict_button):
    # create a one-row features DF according to artifact columns
    feat_cols = artifact["feature_columns"]
    weapon_canon = _canon_weapon(weapon_text)
    numeric_inputs = {k:v for k,v in numeric_values.items()}
    X_row = build_feature_row_from_inputs(feat_cols, weapon_canon, map_name=map_choice, side=side_choice, numeric_inputs=numeric_inputs)
    model = artifact["model"]
    try:
        proba = float(model.predict_proba(X_row.values)[:,1][0]) if hasattr(model, "predict_proba") else float(model.predict(X_row.values)[0])
    except Exception:
        # sklearn estimators expect DataFrames with columns sometimes. Try as DataFrame with column labels
        try:
            proba = float(model.predict_proba(X_row)[0][:,1])  # unlikely
        except Exception as e:
            # fallback to predict on numpy array and hope.
            preds = model.predict(X_row.values)
            proba = float(preds[0])
    # show nicely
    pred_placeholder.subheader("Model prediction")
    pred_placeholder.metric(label="P(team with opening kill wins)", value=f"{proba*100:.1f}%", delta=None)
    st.write("Feature vector (non-zero entries):")
    nonzero = {k:v for k,v in X_row.iloc[0].to_dict().items() if float(v) != 0.0}
    st.json(nonzero)

    # show historical weapon stats if available
    if rounds is not None:
        hist_weapon = rounds.copy()
        # compute actual win per round
        hist_weapon["actual_win"] = (hist_weapon["first_kill_team"].astype(str).fillna("") == hist_weapon["winner_team"].astype(str).fillna("")).astype(int)
        # weapon ranks
        w = weapon_canon
        # top weapons bar chart: actual winrates for top 20
        grp = hist_weapon.groupby("first_kill_weapon").agg(actual_winrate=("actual_win","mean"), count=("first_kill_weapon","count")).reset_index()
        grp = grp[grp["count"]>=20].sort_values("count", ascending=False).head(20)
        # bring selected weapon to top if not present
        if w not in grp["first_kill_weapon"].values:
            extra = hist_weapon[hist_weapon["first_kill_weapon"]==w].groupby("first_kill_weapon").agg(actual_winrate=("actual_win","mean"), count=("first_kill_weapon","count")).reset_index()
            if not extra.empty:
                grp = pd.concat([extra, grp], ignore_index=True)
        # plot with streamlit (matplotlib)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(grp))))
        ax.barh(grp["first_kill_weapon"], grp["actual_winrate"], color="#2b8cbe", alpha=0.8)
        # overlay predicted marker for chosen weapon
        if w:
            idx = grp.index[grp["first_kill_weapon"]==w].tolist()
            if idx:
                y_pos = idx[0]
                ax.plot([proba], [y_pos], marker="D", color="orange", markersize=8, label="model prob")
        ax.set_xlabel("Actual winrate (team with opening kill)")
        ax.set_title("Top weapons — actual first-kill winrate (historical)")
        ax.set_xlim(0,1)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        if w in grp["first_kill_weapon"].values:
            ax.legend()
        hist_weapon_plot.pyplot(fig)

        # Map x side aggregated table / bar plot
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

# ---------- Bulk predict top weapons for chosen map/side ----------
if (not artifact_missing) and (rounds is not None):
    st.markdown("---")
    st.header("Bulk: compare model prob across top weapons on chosen map & side")
    if st.button("Predict top 30 weapons for this map & side"):
        feat_cols = artifact["feature_columns"]
        top_weps = rounds["first_kill_weapon"].value_counts().head(50).index.tolist()
        rows = []
        for wp in top_weps:
            Xr = build_feature_row_from_inputs(feat_cols, wp, map_name=map_choice, side=side_choice)
            rows.append((wp, float(artifact["model"].predict_proba(Xr)[:,1][0]) if hasattr(artifact["model"], "predict_proba") else float(artifact["model"].predict(Xr)[0])))
        df_top = pd.DataFrame(rows, columns=["weapon","pred_prob"]).sort_values("pred_prob", ascending=False)
        st.write("Top weapons (by model predicted probability):")
        st.dataframe(df_top.head(30))
        # small bar chart
        import matplotlib.pyplot as plt
        fig3, ax3 = plt.subplots(figsize=(8, max(3, 0.25*len(df_top.head(20)))))
        ax3.barh(df_top["weapon"].head(20), df_top["pred_prob"].head(20))
        ax3.invert_yaxis()
        ax3.set_xlabel("Predicted probability (first-kill team wins)")
        st.pyplot(fig3)

st.markdown("---")
st.caption("This UI builds a feature row compatible with the artifact's feature columns. The naming conventions it expects are those produced by Trainer.featurize_binary (weapon columns prefixed with 'wp_', map with 'map_', side with 'side_').")
