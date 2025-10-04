from pathlib import Path
from ..config import Config
from ..utils.io import write_parquet
import pandas as pd

class Loader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # ensure dirs exist
        self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.interim_dir.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, name: str):
        """Save a DataFrame to processed/<name>.parquet"""
        target = self.cfg.processed_dir / f"{name}.parquet"
        write_parquet(df, target)
        return target
