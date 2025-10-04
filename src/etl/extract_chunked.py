# src/etl/extract_chunked.py
import re
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import pandas as pd
import numpy as np
from ..config import Config
from ..utils.io import read_csv  # optional helper; we'll use pandas directly here if needed

_BRACED_RE = re.compile(r'^\s*"?\{[\d,\s]+\}"?\s*$')

def decode_braced_ascii_value(s):
    """Decode '{100,101,...}' -> 'de_dust2'. Return original if not matching."""
    if pd.isna(s):
        return s
    if isinstance(s, bytes):
        try:
            return s.decode("utf-8")
        except Exception:
            try:
                return s.decode("latin-1", errors="ignore")
            except Exception:
                return s
    if not isinstance(s, str):
        return s
    if not _BRACED_RE.match(s):
        return s
    stripped = s.strip().strip('"').strip("'")
    inner = stripped[1:-1].strip()  # remove { and }
    if inner == "":
        return ""
    parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
    try:
        byte_vals = bytes([int(p) for p in parts])
        try:
            return byte_vals.decode("utf-8")
        except Exception:
            return byte_vals.decode("latin-1", errors="ignore")
    except Exception:
        return s

class ChunkedExtractor:
    """
    Chunked extractor for large competitive CSVs.
    - Reads CSV in chunks
    - Normalizes column names
    - Decodes braced ASCII strings in map columns (and optionally other columns)
    - Coerces x/y numeric columns
    - Writes processed chunks to data/interim/
    """

    def __init__(self, cfg: Config, chunk_size: int = 200_000, interim_dir: Optional[Path] = None):
        self.cfg = cfg
        self.chunk_size = int(chunk_size)
        self.interim_dir = Path(interim_dir) if interim_dir else (self.cfg.interim_dir)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def normalize_competitive_columns(self, df: pd.DataFrame, decode_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Normalize names and decode braced ascii for certain columns.
        This modifies and returns a copy.
        """
        df = df.copy()
        # rename map -> _map if needed
        if "_map" not in df.columns and "map" in df.columns:
            df = df.rename(columns={"map": "_map"})

        # rename common position variants
        rename_map = {}
        if "att_pos_x" in df.columns: rename_map["att_pos_x"] = "att_x"
        if "att_pos_y" in df.columns: rename_map["att_pos_y"] = "att_y"
        if "vic_pos_x" in df.columns: rename_map["vic_pos_x"] = "vic_x"
        if "vic_pos_y" in df.columns: rename_map["vic_pos_y"] = "vic_y"
        if "attacker_x" in df.columns and "att_x" not in df.columns: rename_map["attacker_x"] = "att_x"
        if "attacker_y" in df.columns and "att_y" not in df.columns: rename_map["attacker_y"] = "att_y"
        if "victim_x" in df.columns and "vic_x" not in df.columns: rename_map["victim_x"] = "vic_x"
        if "victim_y" in df.columns and "vic_y" not in df.columns: rename_map["victim_y"] = "vic_y"
        if rename_map:
            df = df.rename(columns=rename_map)

        # default decode columns
        if decode_cols is None:
            decode_cols = ["_map", "team_1", "team_2", "att_team", "vic_team"]

        # decode braced ascii
        for col in decode_cols:
            if col in df.columns:
                sample = df[col].dropna().astype(str).head(200).tolist()
                if any(_BRACED_RE.match(v) for v in sample):
                    df[col] = df[col].apply(decode_braced_ascii_value)

        # normalize _map strings to lowercase no-spaces
        if "_map" in df.columns:
            df["_map"] = df["_map"].astype(str).str.lower().str.strip().str.replace(" ", "_")

        # build canonical x/y (prefer attacker -> victim -> existing)
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

        return df

    def load_competitive_chunked(self, filename: str = "mm_master_demos.csv", output_prefix: str = "mm_master_chunk"):
        """
        Read the competitive mm_master_demos.csv in chunks and write normalized
        parquet chunks to the interim directory.
        Returns a list of saved chunk file paths.
        """
        base = self.cfg.raw_dir / "competitive"
        file_path = base / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        saved_paths = []
        reader = pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False, iterator=True)
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i} (rows: {len(chunk)})")
            chunk = self.normalize_competitive_columns(chunk)
            # ensure numeric coercion for x,y (safety)
            if "x" in chunk.columns:
                chunk["x"] = pd.to_numeric(chunk["x"], errors="coerce")
            if "y" in chunk.columns:
                chunk["y"] = pd.to_numeric(chunk["y"], errors="coerce")
            out_path = self.interim_dir / f"{output_prefix}_{i:04d}.parquet"
            chunk.to_parquet(out_path, index=False)
            saved_paths.append(out_path)
            print(f"Saved chunk to {out_path}")

        return saved_paths

    def load_competitive_yield(self, filename: str = "mm_master_demos.csv"):
        """
        Generator-style: yields normalized DataFrame chunks to the caller (no disk write).
        Caller must use/consume the chunks. Useful if you want to pipe chunk -> transform -> model
        without intermediate files.
        """
        base = self.cfg.raw_dir / "competitive"
        file_path = base / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        reader = pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False, iterator=True)
        for i, chunk in enumerate(reader):
            chunk = self.normalize_competitive_columns(chunk)
            # numeric coercion
            if "x" in chunk.columns:
                chunk["x"] = pd.to_numeric(chunk["x"], errors="coerce")
            if "y" in chunk.columns:
                chunk["y"] = pd.to_numeric(chunk["y"], errors="coerce")
            yield i, chunk
