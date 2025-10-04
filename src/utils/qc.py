
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    cols = []
    for c in df.columns:
        missing = int(df[c].isna().sum())
        pct = (missing / total * 100) if total > 0 else 0.0
        cols.append({"column": c, "missing_count": missing, "missing_pct": pct})
    return pd.DataFrame(cols).sort_values("missing_pct", ascending=False)


def assert_coords_in_bounds(
    df: pd.DataFrame,
    map_bounds: Dict[str, Dict[str, float]],
    map_col: str = "_map",
    x_col: str = "x",
    y_col: str = "y"
) -> pd.DataFrame:
    """
    Assert that coordinates fall within the configured map bounds.
    Returns a DataFrame of rows out of bounds (empty if all good).
    """
    if x_col not in df.columns or y_col not in df.columns or map_col not in df.columns:
        return pd.DataFrame()

    bad_rows = []
    for idx, row in df.iterrows():
        m = row.get(map_col)
        if m not in map_bounds:
            # skip maps we don't have bounds for
            continue
        b = map_bounds[m]
        try:
            x = float(row.get(x_col))
            y = float(row.get(y_col))
        except Exception:
            bad_rows.append((idx, "nan_or_non_numeric"))
            continue
        if np.isnan(x) or np.isnan(y):
            bad_rows.append((idx, "nan_coord"))
            continue
        if not (b["xmin"] <= x <= b["xmax"]) or not (b["ymin"] <= y <= b["ymax"]):
            bad_rows.append((idx, x, y, m))

    if not bad_rows:
        return pd.DataFrame()

    # Return a nice DataFrame depending on tuple shape
    first = bad_rows[0]
    if len(first) == 4:
        return pd.DataFrame(bad_rows, columns=["index", "x", "y", "map"])
    else:
        return pd.DataFrame(bad_rows, columns=["index", "reason"])


def detect_outliers_zscore(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Return rows flagged as outliers based on z-score across the provided numeric columns.
    Requires scipy.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # If there are no numeric columns, return empty
    if not numeric_cols:
        return pd.DataFrame()

    # subset and drop rows with NaNs in these columns (zscore needs complete rows)
    numeric = df[numeric_cols].dropna()
    if numeric.empty:
        return pd.DataFrame()

    # lazy import to avoid requiring scipy if user doesn't call this function
    try:
        from scipy import stats
    except Exception as e:
        raise ImportError("scipy is required for detect_outliers_zscore. Install with `pip install scipy`.") from e

    # compute z-scores; nan_policy='omit' handles columns with constant values
    z = np.abs(stats.zscore(numeric, nan_policy="omit"))
    if z.ndim == 1:
        # single column case: z is 1D
        mask = z > threshold
    else:
        mask = (z > threshold).any(axis=1)

    # return original rows corresponding to outlier indices
    outlier_idx = numeric.index[mask]
    return df.loc[outlier_idx]


def assert_non_negative(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """
    Returns list of column names that have any negative values (after coercion).
    """
    bad = []
    for c in cols:
        if c in df.columns:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if (coerced < 0).any(skipna=True):
                bad.append(c)
    return bad
