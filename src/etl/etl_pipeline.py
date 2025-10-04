from ..config import Config
from .extract import Extractor
from .transform import Transformer
from .load import Loader
from ..features.build_features import add_opening_kill_flag
import pandas as pd
import sys


def run_all(test_mode: bool = False):
    cfg = Config()
    extractor = Extractor(cfg)
    transformer = Transformer(cfg)
    loader = Loader(cfg)

    # --- Professional dataset ---
    prof = extractor.load_professional()
    if prof:
        print("Loaded professional tables:", list(prof.keys()))
        if "economy" in prof:
            econ = transformer.basic_clean(prof["economy"])
            econ = transformer.normalize_maps(econ, map_col="_map")
            out = loader.save_dataframe(econ, "economy_clean")
            print("Saved processed professional economy to", out)

        if "players" in prof:
            players = transformer.basic_clean(prof["players"])
            out = loader.save_dataframe(players, "players_clean")
            print("Saved processed professional players to", out)

    else:
        print("⚠️ No professional data found in data/raw/professional")

    # --- Competitive dataset ---
    comp = extractor.load_competitive()
    if comp:
        print("Loaded competitive tables:", list(comp.keys()))
        if "mm_master" in comp:
            mm = transformer.basic_clean(comp["mm_master"])
            for col in ["x", "y", "damage"]:
                if col in mm.columns:
                    mm[col] = pd.to_numeric(mm[col], errors="coerce")
            mm = transformer.scale_coordinates(mm, map_name_col="_map", x_col="x", y_col="y")
            if "tick" in mm.columns or "time" in mm.columns:
                mm = add_opening_kill_flag(mm)
            out = loader.save_dataframe(mm, "mm_master_clean")
            print("Saved processed competitive mm_master to", out)

    else:
        print("⚠️ No competitive data found in data/raw/competitive")

    print("✅ ETL process completed!")


if __name__ == "__main__":
    test = ("--test" in sys.argv)
    run_all(test_mode=test)
