import pandas as pd
from typing import Dict
from ..config import Config
import numpy as np

class Transformer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: strip column names, drop obvious fully-empty rows."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        # drop rows that are all NaN
        df = df.dropna(how="all")
        return df

    def normalize_maps(self, df: pd.DataFrame, map_col: str = "_map") -> pd.DataFrame:
        """Normalize map names into a consistent format."""
        df = df.copy()
        if map_col in df.columns:
            df[map_col] = df[map_col].astype(str).str.lower().str.replace(" ", "_")
        return df

    def scale_coordinates(self, df: pd.DataFrame, map_name_col: str = "_map", x_col: str = "x", y_col: str = "y"):
        """
        Deterministic linear scaling of in-game coords to pixel coords.
        Requires cfg.map_bounds to include the map.
        Returns df with x_map, y_map columns when possible.
        """
        df = df.copy()
        def _scale_row(row):
            m = row.get(map_name_col)
            if m is None or m not in self.cfg.map_bounds:
                return np.nan, np.nan
            b = self.cfg.map_bounds[m]
            # linear map from [xmin,xmax] to [0,width]
            try:
                x_map = (row[x_col] - b['xmin']) / (b['xmax'] - b['xmin']) * b['width']
                y_map = (row[y_col] - b['ymin']) / (b['ymax'] - b['ymin']) * b['height']
                return x_map, y_map
            except Exception:
                return np.nan, np.nan

        if x_col in df.columns and y_col in df.columns and map_name_col in df.columns:
            scaled = df.apply(lambda r: _scale_row(r), axis=1, result_type="expand")
            df["x_map"] = scaled[0]
            df["y_map"] = scaled[1]
        return df

    def aggregate_rounds(self, duels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Very small example: create a per-round aggregation table with counts of duels and sum damage.
        Expects duels_df to have columns: match_id, round, damage, attacker, victim
        """
        if duels_df is None or duels_df.empty:
            return pd.DataFrame()
        dd = duels_df.copy()
        # try to find round column alternatives
        round_cols = [c for c in dd.columns if c.lower().startswith("round")]
        round_col = round_cols[0] if round_cols else "round"
        dd[round_col] = dd[round_col].fillna(-1).astype(int)
        agg = dd.groupby(["match_id", round_col]).agg(
            duels_count=pd.NamedAgg(column="damage", aggfunc="count"),
            damage_sum=pd.NamedAgg(column="damage", aggfunc="sum"),
            unique_attackers=pd.NamedAgg(column="attacker", aggfunc=lambda s: s.nunique())
        ).reset_index()
        agg = agg.rename(columns={round_col: "round_no"})
        return agg
