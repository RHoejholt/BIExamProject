from pathlib import Path
import pandas as pd
from ..config import Config
from ..utils.io import read_csv

class Extractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_competitive(self):
        """Load key competitive files if present. Returns dict of DataFrames."""
        base = self.cfg.raw_dir / "competitive"
        outputs = {}
        if (base / "mm_master_demos.csv").exists():
            outputs["mm_master"] = read_csv(base / "mm_master_demos.csv", low_memory=False)
        if (base / "mm_grenades_demos.csv").exists():
            outputs["mm_grenades"] = read_csv(base / "mm_grenades_demos.csv", low_memory=False)
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
