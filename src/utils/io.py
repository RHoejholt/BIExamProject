import pandas as pd
from pathlib import Path
from typing import Iterator, Dict

def read_csv_chunked(path: Path, dtype: Dict = None, parse_dates: list = None, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks for large CSVs. Consumer can concat or process streaming."""
    for chunk in pd.read_csv(path, dtype=dtype, parse_dates=parse_dates, chunksize=chunksize):
        yield chunk

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def write_parquet(df, path: Path, partition_cols: list = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        df.to_parquet(path, index=False, partition_cols=partition_cols)
    else:
        df.to_parquet(path, index=False)
