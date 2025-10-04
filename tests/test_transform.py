# tests/test_transform.py
from src.config import Config
from src.etl.transform import Transformer
import pandas as pd

def test_basic_clean_and_normalize_maps(tmp_path):
    cfg = Config()
    t = Transformer(cfg)
    df = pd.DataFrame({
        "_map": ["De_Dust2", None, "  Mirage  "],
        "x": [100, None, 200],
        "y": [50, None, 400],
        "damage": [20, 0, 10]
    })
    out = t.basic_clean(df)
    assert not out.empty
    out2 = t.normalize_maps(out, map_col="_map")
    assert out2["_map"].iloc[0] == "de_dust2"
    assert out2["_map"].iloc[2] == "mirage"
