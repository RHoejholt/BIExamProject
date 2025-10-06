from pathlib import Path
import pandas as pd
from ..utils.io import write_parquet
from ..config import Config

class Loader:
    # do etl stuff: write parquet til data/processed
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.interim_dir.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        target = self.cfg.processed_dir / f"{name}.parquet"
        write_parquet(df, target)
        return target
