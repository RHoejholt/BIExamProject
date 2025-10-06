
import pandas as pd
import numpy as np
from typing import Iterable, Optional
from ..config import Config

# simple transformer med kun de nødvendige funktioner

class Transformer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # do etl stuff: strip cols og drop fuldt tomme rækker
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(how="all")
        # rename map -> _map ved behov
        if "_map" not in df.columns and "map" in df.columns:
            df = df.rename(columns={"map": "_map"})
        # standardiser pos-kolonner kort
        rename_map = {}
        if "att_pos_x" in df.columns: rename_map["att_pos_x"] = "att_x"
        if "att_pos_y" in df.columns: rename_map["att_pos_y"] = "att_y"
        if "vic_pos_x" in df.columns: rename_map["vic_pos_x"] = "vic_x"
        if "vic_pos_y" in df.columns: rename_map["vic_pos_y"] = "vic_y"
        if rename_map:
            df = df.rename(columns=rename_map)
        # bygg canonical x,y hvis muligt
        for col in ("x","att_x","vic_x"):
            if col in df.columns:
                df["x"] = pd.to_numeric(df["x"] if "x" in df.columns else df[col], errors="coerce")
                break
        for col in ("y","att_y","vic_y"):
            if col in df.columns:
                df["y"] = pd.to_numeric(df["y"] if "y" in df.columns else df[col], errors="coerce")
                break
        # normaliser map tekst hvis tilstede
        if "_map" in df.columns:
            df["_map"] = df["_map"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        return df

    def scale_coordinates(self, df: pd.DataFrame, map_name_col: str = "_map", x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
        # do etl stuff: skaler game coords til pixel coords using cfg.map_bounds
        if df is None or df.empty:
            return df
        df = df.copy()
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df["x_map"] = pd.NA
        df["y_map"] = pd.NA
        for map_name, b in self.cfg.map_bounds.items():
            mask = df.get(map_name_col) == map_name
            if not mask.any():
                continue
            denom_x = (b["xmax"] - b["xmin"]) or 1.0
            denom_y = (b["ymax"] - b["ymin"]) or 1.0
            xs = df.loc[mask, x_col].astype(float)
            ys = df.loc[mask, y_col].astype(float)
            x_map = (xs - b["xmin"]) / denom_x * b["width"]
            y_map = (ys - b["ymin"]) / denom_y * b["height"]
            y_map = b["height"] - y_map
            df.loc[mask, "x_map"] = x_map
            df.loc[mask, "y_map"] = y_map
        return df
