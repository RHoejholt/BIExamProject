# src/models/train.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _safe_load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)

def build_features_for_win_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature dataframe to predict round winner from per-event/round data.
    Tries to compute:
      - first_frag (opening_kill or event_rank==1)
      - team_econ or eco_team_value (if present)
      - round_no, map, tick
      - attacker/victim stats aggregated per round if present
    Result is per-row; may need groupby to per-round depending on input.
    """
    df = df.copy()

    # prefer an explicit 'opening_kill' column
    if "opening_kill" not in df.columns and "event_rank" in df.columns:
        df["opening_kill"] = df["event_rank"] == 1

    # if we have per-round economy columns, try to infer current team's econ
    # This is heuristic: look for columns with 'econ' or 'team_money' names
    econ_cols = [c for c in df.columns if "econ" in c.lower() or "money" in c.lower() or "cash" in c.lower()]
    # We'll create a numeric 'team_econ' if any found: take first match per row
    if econ_cols:
        df["team_econ"] = pd.to_numeric(df[econ_cols[0]], errors="coerce")
    else:
        df["team_econ"] = np.nan

    # Decide target: if 'round_winner' or 'winner' present use it (boolean T/CT)
    target_col = None
    for cand in ("round_winner", "winner", "winning_team", "res_winner"):
        if cand in df.columns:
            target_col = cand
            break

    # If target_col is None, try to create a proxy: if attacker/victim flags + team side exist
    # For simplicity, this function returns rows where target_col exists; training script will enforce that.
    if target_col is None:
        # nothing to do here; return df and let caller decide
        return df

    # Build features DataFrame (per-row)
    feat_cols = []
    # basic numeric features
    for c in ("team_econ", "tick", "round_no"):
        if c in df.columns:
            feat_cols.append(c)
    # boolean opening kill
    if "opening_kill" in df.columns:
        df["opening_kill_flag"] = df["opening_kill"].astype(int)
        feat_cols.append("opening_kill_flag")

    # map as categorical (we'll one-hot encode later)
    if "_map" in df.columns:
        feat_cols.append("_map")

    # If attacker or victim weapons/ids exist we can include simple indicators
    if "weapon" in df.columns:
        feat_cols.append("weapon")

    # drop rows missing target
    df = df[df[target_col].notna()].copy()

    return df, feat_cols, target_col

def train_classification_model():
    """
    Load merged_professional or mm_master_clean and train a classifier to predict round winner.
    Saves model at src/models/artifacts/win_classifier.joblib
    """
    # try merged_professional first, then mm_master_clean
    try:
        df = _safe_load("merged_professional")
    except Exception:
        df = _safe_load("mm_master_clean")

    df = df.reset_index(drop=True)

    df, feat_cols, target_col = build_features_for_win_prediction(df)

    # simple fallback: if target holds strings for team names, convert to binary using a column 'team' if present
    # If target is 'T'/'CT' or 't'/'ct', map to 1/0
    y_raw = df[target_col]
    if y_raw.dtype == object:
        y = y_raw.apply(lambda v: 1 if str(v).lower().startswith("t") else 0)
    else:
        # numeric -> assume already 0/1
        y = pd.to_numeric(y_raw, errors="coerce").astype(int)

    # select candidate features, use only those available
    X = df.copy()
    # keep the features detected earlier
    # Expand categorical columns
    candidate_num = [c for c in ("team_econ", "tick", "round_no", "opening_kill_flag") if c in X.columns]
    candidate_cat = [c for c in ("_map", "weapon") if c in X.columns]

    # build preprocessing
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    num_transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_transform = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="NA")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_transform, candidate_num),
            ("cat", cat_transform, candidate_cat),
        ],
        remainder="drop"
    )

    X_use = X[candidate_num + candidate_cat].copy().reset_index(drop=True)
    # if nothing to train on, abort
    if X_use.shape[1] == 0:
        raise RuntimeError("No features available to train on.")

    X_train, X_test, y_train, y_test = train_test_split(X_use, y, test_size=0.2, random_state=42, stratify=y)

    # classifier
    clf = Pipeline([("pre", preproc), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))])

    # cross-val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print("CV ROC-AUC:", scores.mean(), "+/-", scores.std())

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    model_path = MODELS_DIR / "win_classifier.joblib"
    joblib.dump(clf, model_path)
    print("Saved classifier to", model_path)
    return clf

def train_regression_model(target_column: str = "team_econ"):
    """
    Train a regression model to predict a numeric target (team economy or similar).
    """
    try:
        df = _safe_load("merged_professional")
    except Exception:
        df = _safe_load("mm_master_clean")

    df = df.reset_index(drop=True)

    # simple features: opening_kill_flag and round_no, _map
    if "opening_kill" not in df.columns and "event_rank" in df.columns:
        df["opening_kill"] = df["event_rank"] == 1
    if "opening_kill" in df.columns:
        df["opening_kill_flag"] = df["opening_kill"].astype(int)

    if target_column not in df.columns:
        raise RuntimeError(f"Target {target_column} not found in data.")

    y = pd.to_numeric(df[target_column], errors="coerce")
    X = df[["opening_kill_flag"]].copy() if "opening_kill_flag" in df.columns else pd.DataFrame()
    if "_map" in df.columns:
        X["_map"] = df["_map"]

    # simple preprocessing (numeric + map OHE)
    num_cols = [c for c in X.columns if X[c].dtype != object and c != "_map"]
    cat_cols = [c for c in X.columns if c == "_map"]

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    num_transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_transform = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="NA")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preproc = ColumnTransformer(transformers=[("num", num_transform, num_cols), ("cat", cat_transform, cat_cols)], remainder="drop")
    X_use = X[num_cols + cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(X_use, y, test_size=0.2, random_state=42)

    model = Pipeline([("pre", preproc), ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)

    yp = model.predict(X_test)
    print("RMSE:", mean_squared_error(y_test, yp, squared=False))
    print("MAE:", mean_absolute_error(y_test, yp))
    print("R2:", r2_score(y_test, yp))

    model_path = MODELS_DIR / "econ_regressor.joblib"
    joblib.dump(model, model_path)
    print("Saved regressor to", model_path)
    return model
