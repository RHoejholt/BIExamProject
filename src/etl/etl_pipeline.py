
#TO RUN:
#1 Open anaconda prompt
#2 Navigate to project root
#3 venv\Scripts\activate
#4 -m src.etl.etl_pipeline

import logging
import pandas as pd
from ..config import Config
from .extract import Extractor
from .map_utils import filter_and_scale_mm
from .transform import Transformer
from .load import Loader
from ..features.build_features import add_opening_kill_flag
from ..etl.cleaning_and_merge import (
    harmonize_weapon_names, normalize_ids, parse_rank_field,
    expand_economy_wide_to_long, merge_players_results_economy,
)
from ..utils import qc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clamp_coords(df: pd.DataFrame, map_bounds: dict, map_col: str = "_map", x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """
    Clamp x/y coordinates to the provided map_bounds for each map.
    Keeps rows but snaps values into [xmin, xmax] and [ymin, ymax].
    """
    df = df.copy()
    if df is None or df.empty:
        return df
    for map_name, b in map_bounds.items():
        mask = df[map_col] == map_name
        if not mask.any():
            continue
        if x_col in df.columns:
            df.loc[mask, x_col] = pd.to_numeric(df.loc[mask, x_col], errors="coerce").clip(lower=b["xmin"], upper=b["xmax"])
        if y_col in df.columns:
            df.loc[mask, y_col] = pd.to_numeric(df.loc[mask, y_col], errors="coerce").clip(lower=b["ymin"], upper=b["ymax"])
    return df


def run_all():
    cfg = Config()
    extractor = Extractor(cfg)
    transformer = Transformer(cfg)
    loader = Loader(cfg)

    # PROFESSIONAL
    prof = extractor.load_professional()
    if prof:
        logger.info("Processing professional dataset")

        # players
        if "players" in prof:
            players = transformer.basic_clean(prof["players"])
            players = normalize_ids(players, ["player_id", "steam_id", "player_steam_id"])
            players = parse_rank_field(players, col="rank")
            players = harmonize_weapon_names(players, col="fav_weapon")
            loader.save_dataframe(players, "players_clean")
            logger.info("Saved players_clean")

        # economy
        if "economy" in prof:
            econ = transformer.basic_clean(prof["economy"])
            econ = transformer.normalize_maps(econ, map_col="_map")
            econ_long = expand_economy_wide_to_long(econ)
            loader.save_dataframe(econ, "economy_clean")
            loader.save_dataframe(econ_long, "economy_long")
            logger.info("Saved economy_clean and economy_long")

        # results
        if "results" in prof:
            results = transformer.basic_clean(prof["results"])
            results = transformer.normalize_maps(results, map_col="_map")
            loader.save_dataframe(results, "results_clean")
            logger.info("Saved results_clean")

        # merged dataset
        try:
            if all(k in prof for k in ("players", "results", "economy")):
                merged = merge_players_results_economy(
                    prof["players"], prof["results"], prof["economy"]
                )
                loader.save_dataframe(merged, "merged_professional")
                logger.info("Saved merged_professional")
        except Exception as e:
            logger.exception("Merging professional tables failed: %s", e)

    else:
        logger.info("No professional dataset found")

    # COMPETITIVE
    comp = extractor.load_competitive()
    if comp:
        logger.info("Processing competitive dataset")

        # src/etl/etl_pipeline.py (inside run_all(), in the COMPETITIVE mm_master section)
        if "mm_master" in comp:
            mm = transformer.basic_clean(comp["mm_master"])
            # ensure canonical _map present
            mm = transformer.normalize_maps(mm, map_col="_map")

            mm = normalize_ids(mm, ["attackerSteamId", "victimSteamId", "attacker", "victim"])

            for col in ["x", "y", "damage", "tick"]:
                if col in mm.columns:
                    mm[col] = pd.to_numeric(mm[col], errors="coerce")

            # filter to allowed maps AND compute x_map/y_map using exact csv bounds from Config
            mm = filter_and_scale_mm(mm, cfg.map_bounds, map_name_col="_map", x_col="x", y_col="y")

            if "tick" in mm.columns or "time" in mm.columns or "seconds" in mm.columns:
                mm = add_opening_kill_flag(mm)

            loader.save_dataframe(mm, "mm_master_clean")
            logger.info("Saved mm_master_clean")

        if "mm_grenades" in comp:
            mg = transformer.basic_clean(comp["mm_grenades"])
            mg = transformer.normalize_maps(mg, map_col="_map")
            mg = mg.rename(columns={c: c.strip() for c in mg.columns})
            loader.save_dataframe(mg, "mm_grenades_clean")
            logger.info("Saved mm_grenades_clean")

    else:
        logger.info("No competitive dataset found")

    logger.info("ETL finished")


if __name__ == "__main__":
    run_all()
