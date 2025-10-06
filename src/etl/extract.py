from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from ..config import Config

class Extractor:
    """
    enkel extractor der håndterer både competitive og professional csv'er.
    - primary extract class: læs hele competitive mm_master eller chunks
    - normalize_competitive_columns sikrer _map, x/y canonical cols
    """

    def __init__(self, cfg: Config, chunksize: int = 200_000):
        self.cfg = cfg
        self.chunksize = int(chunksize)

    def normalize_competitive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # normaliser kolonnenavne og dann canonical x,y og _map
        df = df.copy()
        if "_map" not in df.columns and "map" in df.columns:
            df = df.rename(columns={"map": "_map"})
        # position kolonne-forksellige navne -> standard
        if "att_pos_x" in df.columns:
            df = df.rename(columns={"att_pos_x": "att_x"})
        if "att_pos_y" in df.columns:
            df = df.rename(columns={"att_pos_y": "att_y"})
        if "vic_pos_x" in df.columns:
            df = df.rename(columns={"vic_pos_x": "vic_x"})
        if "vic_pos_y" in df.columns:
            df = df.rename(columns={"vic_pos_y": "vic_y"})
        # andre varianter
        if "attacker_x" in df.columns and "att_x" not in df.columns:
            df = df.rename(columns={"attacker_x": "att_x"})
        if "attacker_y" in df.columns and "att_y" not in df.columns:
            df = df.rename(columns={"attacker_y": "att_y"})
        if "victim_x" in df.columns and "vic_x" not in df.columns:
            df = df.rename(columns={"victim_x": "vic_x"})
        if "victim_y" in df.columns and "vic_y" not in df.columns:
            df = df.rename(columns={"victim_y": "vic_y"})

        # coalesce attacker/victim/x into canonical x,y
        x_candidates = [c for c in ("att_x", "vic_x", "x") if c in df.columns]
        y_candidates = [c for c in ("att_y", "vic_y", "y") if c in df.columns]

        if x_candidates:
            df["x"] = pd.to_numeric(df[x_candidates[0]], errors="coerce")
            for c in x_candidates[1:]:
                df["x"] = df["x"].fillna(pd.to_numeric(df[c], errors="coerce"))
        if y_candidates:
            df["y"] = pd.to_numeric(df[y_candidates[0]], errors="coerce")
            for c in y_candidates[1:]:
                df["y"] = df["y"].fillna(pd.to_numeric(df[c], errors="coerce"))

        if "_map" in df.columns:
            df["_map"] = df["_map"].astype(str).str.strip().str.lower().str.replace(" ", "_")

        return df

    def load_competitive(self) -> Dict[str, pd.DataFrame]:
        """
        læs competitive files fra data/raw/competitive
        returnerer dict med keys: mm_master (DataFrame), mm_grenades (DataFrame) hvis findes
        """
        base = self.cfg.raw_dir / "competitive"
        out = {}
        mm_path = base / "mm_master_demos.csv"
        if mm_path.exists():
            df = pd.read_csv(mm_path, low_memory=False)
            df = self.normalize_competitive_columns(df)
            out["mm_master"] = df
        gren_path = base / "mm_grenades_demos.csv"
        if gren_path.exists():
            mg = pd.read_csv(gren_path, low_memory=False)
            mg = self.normalize_competitive_columns(mg)
            out["mm_grenades"] = mg
        map_data_path = base / "map_data.csv"
        if map_data_path.exists():
            out["map_data"] = pd.read_csv(map_data_path)
        return out

    def load_professional(self) -> Dict[str, pd.DataFrame]:
        """
        læs professional csvs fra data/raw/professional
        returnerer dict med keys som filnavne uden .csv
        """
        base = self.cfg.raw_dir / "professional"
        out = {}
        for fname in ("results.csv", "economy.csv", "players.csv", "picks.csv"):
            p = base / fname
            if p.exists():
                parse_dates = ["date"] if fname == "economy.csv" else None
                out[fname.replace(".csv", "")] = pd.read_csv(p, low_memory=False, parse_dates=parse_dates)
        return out

    def read_competitive_chunked(self, filename: str = "mm_master_demos.csv"):
        """
        generator der yield'er normaliserede chunks (i tilfælde af memory constraints).
        hver yield er (i, chunk_df).
        """
        base = self.cfg.raw_dir / "competitive"
        file_path = base / filename
        if not file_path.exists():
            return
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=self.chunksize, low_memory=False)):
            yield i, self.normalize_competitive_columns(chunk)
