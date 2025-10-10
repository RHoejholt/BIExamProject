
#TO RUN:
#1 Open anaconda prompt
#2 Navigate to project root
#3 venv\Scripts\activate
#4 python -m src.etl.etl_pipeline

import logging
import pandas as pd
from ..config import Config
from .extract import Extractor
from .transform import Transformer
from .load import Loader
from .map_utils import filter_and_scale_mm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_ids(df: pd.DataFrame, id_cols):
    # do etl stuff: simple id normalization
    df = df.copy()
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].where(pd.notna(df[c]), None)
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None})
    return df

def parse_rank_field(df: pd.DataFrame, col: str = "rank"):
    # do etl stuff: map rank strings to ordinals (very small helper)
    RANK_MAP = {"silver_1":1,"global_elite":18}
    if col not in df.columns:
        return df
    df = df.copy()
    df[col + "_ordinal"] = df[col].astype(str).str.lower().str.replace(" ", "_").map(RANK_MAP)
    return df

def run_all():
    cfg = Config()
    ext = Extractor(cfg)
    tr = Transformer(cfg)
    loader = Loader(cfg)

    prof = ext.load_professional()
    if prof:
        logger.info("Processing professional dataset")
        if "players" in prof:
            players = tr.basic_clean(prof["players"])
            players = normalize_ids(players, ["player_id","steam_id","player_steam_id"])
            players = parse_rank_field(players, col="rank")
            loader.save_dataframe(players, "players_clean")
        if "economy" in prof:
            econ = tr.basic_clean(prof["economy"])
            loader.save_dataframe(econ, "economy_clean")
        if "results" in prof:
            results = tr.basic_clean(prof["results"])
            loader.save_dataframe(results, "results_clean")
    else:
        logger.info("no professional data")

    comp = ext.load_competitive()
    if comp and "mm_master" in comp:
        logger.info("Processing competitive dataset")
        mm = tr.basic_clean(comp["mm_master"])
        mm = normalize_ids(mm, ["attackerSteamId","victimSteamId","attacker","victim"])
        # coerce numeric cols
        for c in ("x","y","damage","tick"):
            if c in mm.columns:
                mm[c] = pd.to_numeric(mm[c], errors="coerce")
        # filter maps and scale to pixel coords
        mm = filter_and_scale_mm(mm, cfg.map_bounds)
        loader.save_dataframe(mm, "mm_master_clean")
        logger.info("Saved mm_master_clean")
    else:
        logger.info("no competitive data")

    logger.info("ETL finished")

if __name__ == "__main__":
    run_all()
