# src/etl/transform.py

import re
import pandas as pd
import numpy as np
from typing import Iterable, Optional
from ..config import Config

_BRACED_RE = re.compile(r'^\s*"?\{[\d,\s]+\}"?\s*$')

def decode_braced_ascii_value(s):
    if pd.isna(s):
        return s
    if isinstance(s, bytes):
        try:
            return s.decode("utf-8")
        except Exception:
            return s
    if not isinstance(s, str):
        return s
    if not _BRACED_RE.match(s):
        return s
    stripped = s.strip().strip('"').strip("'")
    inner = stripped[1:-1].strip()
    if not inner:
        return ""
    parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
    try:
        byte_vals = bytes([int(p) for p in parts])
        try:
            return byte_vals.decode("utf-8")
        except UnicodeDecodeError:
            return byte_vals.decode("latin-1", errors="ignore")
    except Exception:
        return s


class Transformer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def basic_clean(self, df: pd.DataFrame, decode_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        - strip column names
        - drop rows that are all-NaN
        - normalize common column names (map-> _map, attackers/victims coords)
        - decode braced-ascii values for map/team/player columns if present
        - produce canonical x,y columns (prefer attacker coords then victim then existing x/y)
        """
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(how="all")

        # rename map -> _map if necessary (helps if this is called on raw competitive data)
        if "_map" not in df.columns and "map" in df.columns:
            df = df.rename(columns={"map": "_map"})

        # rename common attacker/victim position columns
        rename_map = {}
        if "att_pos_x" in df.columns: rename_map["att_pos_x"] = "att_x"
        if "att_pos_y" in df.columns: rename_map["att_pos_y"] = "att_y"
        if "vic_pos_x" in df.columns: rename_map["vic_pos_x"] = "vic_x"
        if "vic_pos_y" in df.columns: rename_map["vic_pos_y"] = "vic_y"
        if "att_x" not in df.columns and "attacker_x" in df.columns:
            rename_map["attacker_x"] = "att_x"
        if "att_y" not in df.columns and "attacker_y" in df.columns:
            rename_map["attacker_y"] = "att_y"
        if "vic_x" not in df.columns and "victim_x" in df.columns:
            rename_map["victim_x"] = "vic_x"
        if "vic_y" not in df.columns and "victim_y" in df.columns:
            rename_map["victim_y"] = "vic_y"
        if rename_map:
            df = df.rename(columns=rename_map)

        # decode braced ascii in target columns if detected
        if decode_cols is None:
            decode_cols = ["_map", "team_1", "team_2", "att_team", "vic_team", "att_id", "vic_id"]
        for col in decode_cols:
            if col in df.columns:
                sample = df[col].dropna().astype(str).head(200).tolist()
                matches = sum(1 for v in sample if _BRACED_RE.match(v))
                if matches >= 1:
                    df[col] = df[col].apply(decode_braced_ascii_value)

        # normalize map string (basic normalization)
        if "_map" in df.columns:
            df["_map"] = df["_map"].astype(str).str.lower().str.replace(" ", "_")

        # build canonical x,y: prefer attacker coords -> victim coords -> x/y if present
        def coalesce_numeric(cols):
            for c in cols:
                if c in df.columns:
                    vals = pd.to_numeric(df[c], errors="coerce")
                    yield vals
            # if none present, yield empty series
            yield pd.Series([np.nan]*len(df))

        x_series = None
        for s in coalesce_numeric(["att_x", "vic_x", "x"]):
            if x_series is None:
                x_series = s
            else:
                x_series = x_series.fillna(s)
        y_series = None
        for s in coalesce_numeric(["att_y", "vic_y", "y"]):
            if y_series is None:
                y_series = s
            else:
                y_series = y_series.fillna(s)

        if x_series is not None and y_series is not None:
            df["x"] = x_series
            df["y"] = y_series

        return df

    def normalize_maps(self, df: pd.DataFrame, map_col: str = "_map") -> pd.DataFrame:
        """
        Enforce a single canonical map column: '_map'.
        - If input has 'map' but not '_map', rename 'map' -> '_map'
        - If both present, drop the original 'map' (keep '_map')
        - Normalize casing/spacing (lowercase, underscores)
        - Decode braced ascii if it looks like braced ascii
        This is intentionally hard-coded for this project (only target data).
        """
        if df is None:
            return df
        df = df.copy()

        # rename if needed
        if map_col not in df.columns and "map" in df.columns:
            df = df.rename(columns={"map": map_col})

        # If both exist prefer map_col and drop legacy 'map'
        if map_col in df.columns and "map" in df.columns:
            # keep map_col, drop legacy
            if map_col != "map":
                df = df.drop(columns=["map"])

        # Normalize final column contents
        if map_col in df.columns:
            # apply basic normalization -> lowercase, underscores, strip
            df[map_col] = df[map_col].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

            # If values appear like braced ascii, decode them (safe to call)
            sample = df[map_col].dropna().astype(str).head(200).tolist()
            matches = sum(1 for v in sample if _BRACED_RE.match(v))
            if matches >= 1:
                df[map_col] = df[map_col].apply(decode_braced_ascii_value)
                # run the normalization again after decode
                df[map_col] = df[map_col].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

        return df

    def scale_coordinates(self, df: pd.DataFrame, map_name_col: str = "_map", x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
        """
        Vectorized scaling to pixel coordinates (x_map, y_map) using cfg.map_bounds.
        Assumes cfg.map_bounds keys correspond to the map strings in df[map_name_col].
        Flips Y to image coordinates (0 top).
        """
        df = df.copy()
        if map_name_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
            # nothing to do
            return df

        # ensure numeric
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        # initialize empty columns
        df["x_map"] = np.nan
        df["y_map"] = np.nan

        for map_name, bounds in self.cfg.map_bounds.items():
            mask = df[map_name_col] == map_name
            if not mask.any():
                continue
            b = bounds
            denom_x = (b["xmax"] - b["xmin"]) or 1.0
            denom_y = (b["ymax"] - b["ymin"]) or 1.0
            xs = df.loc[mask, x_col]
            ys = df.loc[mask, y_col]
            x_map = (xs - b["xmin"]) / denom_x * b["width"]
            y_map = (ys - b["ymin"]) / denom_y * b["height"]
            # flip y to image coordinates
            y_map = b["height"] - y_map
            df.loc[mask, "x_map"] = x_map
            df.loc[mask, "y_map"] = y_map

        return df



def filter_and_scale_maps(self, df: pd.DataFrame, map_name_col: str = "_map", x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """
    Keep only rows whose map is present in self.cfg.map_bounds and compute x_map/y_map
    scaled to the configured width/height for each map.

    Returns a copy of the DataFrame with x_map and y_map columns (0..width, 0..height).
    Rows for maps not in cfg.map_bounds or with non-numeric coords are dropped.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # canonicalize map column name (if the DF has "map" instead of "_map")
    if map_name_col not in df.columns and "map" in df.columns:
        df[map_name_col] = df["map"].astype(str).str.strip().str.lower()
    else:
        df[map_name_col] = df[map_name_col].astype(str).str.strip().str.lower()

    # allowed maps from config
    allowed_maps = set(self.cfg.map_bounds.keys())

    # filter to allowed maps
    df = df[df[map_name_col].isin(allowed_maps)].copy()

    # coerce numeric coords
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # compute scaled x_map / y_map using existing scale_coordinates method
    # (scale_coordinates iterates cfg.map_bounds, so it's compatible)
    df = self.scale_coordinates(df, map_name_col=map_name_col, x_col=x_col, y_col=y_col)

    # drop rows where scaling failed (x_map/y_map NaN)
    if "x_map" in df.columns and "y_map" in df.columns:
        df = df.dropna(subset=["x_map", "y_map"], how="any").reset_index(drop=True)

    return df
