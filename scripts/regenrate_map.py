# scripts/regenerate_map.py
from src.config import Config
from src.etl.map_utils import filter_and_scale_mm, make_heatmap_overlay
from src.viz.heatmaps import composite_heatmap_on_map
import pandas as pd
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/regenerate_map.py <map_name> [alpha_scale] [vmax_percentile] [use_log]")
    sys.exit(1)

map_name = sys.argv[1].lower()
alpha_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
vmax_percentile = float(sys.argv[3]) if len(sys.argv) > 3 else 98.5
use_log = (sys.argv[4].lower() == "true") if len(sys.argv) > 4 else True

cfg = Config()
proc = cfg.processed_dir
mm_path = next(iter(list(proc.glob("mm_master_clean.parquet")) + list(proc.glob("mm_master*.parquet"))), None)
if mm_path is None:
    print("mm_master parquet not found; run ETL first")
    sys.exit(1)

mm = pd.read_parquet(mm_path)
mm = filter_and_scale_mm(mm, cfg.map_bounds)
if map_name not in set(mm["_map"].unique()):
    print("Map not found in mm:", map_name)
    sys.exit(1)

map_png = cfg.maps_dir / f"{map_name}.png"
overlay = proc / "overlays" / f"{map_name}_heat_custom.png"
composite = proc / "overlays" / f"{map_name}_composite_custom.png"
overlay.parent.mkdir(parents=True, exist_ok=True)

make_heatmap_overlay(mm, map_name, map_png if map_png.exists() else overlay, overlay, cfg.map_bounds,
                     bins=cfg.map_bounds[map_name]["width"],
                     use_log=use_log, vmax_percentile=vmax_percentile, alpha_scale=alpha_scale, blur=6, gamma=0.85)
if map_png.exists():
    composite_heatmap_on_map(map_png, overlay, composite, alpha=0.95)
    print("Saved composite:", composite)
else:
    print("Saved overlay:", overlay)
