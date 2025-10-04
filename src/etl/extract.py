from pathlib import Path
import pandas as pd
from ..config import Config
from ..utils.io import read_csv

class Extractor:

    def normalize_competitive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names and create canonical columns:
          - map or _map  -> _map
          - attacker/victim position pairs:
              att_pos_x, att_pos_y -> att_x, att_y
              vic_pos_x, vic_pos_y -> vic_x, vic_y
          - canonical x,y: pick attacker coords if present else victim coords
        Returns the modified DataFrame (copy).
        """
        df = df.copy()

        # standardize common name variants -> single names
        rename_map = {}
        if "map" in df.columns and "_map" not in df.columns:
            rename_map["map"] = "_map"

        # attacker coords
        if "att_pos_x" in df.columns:
            rename_map["att_pos_x"] = "att_x"
        if "att_pos_y" in df.columns:
            rename_map["att_pos_y"] = "att_y"

        # victim coords
        if "vic_pos_x" in df.columns:
            rename_map["vic_pos_x"] = "vic_x"
        if "vic_pos_y" in df.columns:
            rename_map["vic_pos_y"] = "vic_y"

        # other common variants
        if "attacker_x" in df.columns and "att_x" not in df.columns:
            rename_map["attacker_x"] = "att_x"
        if "attacker_y" in df.columns and "att_y" not in df.columns:
            rename_map["attacker_y"] = "att_y"

        if "victim_x" in df.columns and "vic_x" not in df.columns:
            rename_map["victim_x"] = "vic_x"
        if "victim_y" in df.columns and "vic_y" not in df.columns:
            rename_map["victim_y"] = "vic_y"

        df = df.rename(columns=rename_map)

        # create canonical x,y: prefer attacker coords, fallback to victim coords
        # (use pd.to_numeric to coerce strings -> numbers)
        x_candidates = []
        y_candidates = []
        if "att_x" in df.columns:
            x_candidates.append("att_x")
        if "vic_x" in df.columns:
            x_candidates.append("vic_x")
        if "x" in df.columns:
            x_candidates.append("x")
        if "att_y" in df.columns:
            y_candidates.append("att_y")
        if "vic_y" in df.columns:
            y_candidates.append("vic_y")
        if "y" in df.columns:
            y_candidates.append("y")

        # build canonical x
        if x_candidates:
            # create a series by coalescing candidates
            df["x"] = pd.to_numeric(df[x_candidates[0]], errors="coerce")
            for c in x_candidates[1:]:
                df["x"] = df["x"].fillna(pd.to_numeric(df[c], errors="coerce"))
        # build canonical y
        if y_candidates:
            df["y"] = pd.to_numeric(df[y_candidates[0]], errors="coerce")
            for c in y_candidates[1:]:
                df["y"] = df["y"].fillna(pd.to_numeric(df[c], errors="coerce"))

        # optionally ensure _map is string and decode braced ascii if needed (reuse earlier decode helper)
        # If you have decode_braced_ascii_value in transformer, you can import and call it here.
        # For safety, do a quick strip/normalize:
        if "_map" in df.columns:
            df["_map"] = df["_map"].astype(str).str.strip()

        return df

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_competitive(self):
        base = self.cfg.raw_dir / "competitive"
        outputs = {}
        mm_path = base / "mm_master_demos.csv"
        if mm_path.exists():
            df = read_csv(mm_path, low_memory=False)
            df = self.normalize_competitive_columns(df)
            outputs["mm_master"] = df
        # other files...
        if (base / "mm_grenades_demos.csv").exists():
            mg = read_csv(base / "mm_grenades_demos.csv", low_memory=False)
            mg = self.normalize_competitive_columns(mg)  # safe no-op if cols absent
            outputs["mm_grenades"] = mg
        if (base / "map_data.csv").exists():
            outputs["map_data"] = read_csv(base / "map_data.csv")
        return outputs

    def load_professional(self):
        base = self.cfg.raw_dir / "professional"
        outputs = {}
        for fname in ["results.csv", "picks.csv", "economy.csv", "players.csv"]:
            p = base / fname
            if p.exists():
                outputs[fname.replace(".csv", "")] = read_csv(p, low_memory=False, parse_dates=["date"] if fname=="economy.csv" else None)
        return outputs
