# scripts/generate_all_heatmaps_strong.py
from src.config import Config
from src.etl.map_utils import amplify_overlay_alpha
from src.etl.map_utils import filter_and_scale_mm, make_heatmap_overlay
from src.viz.heatmaps import composite_heatmap_on_map
import pandas as pd
from pathlib import Path
import sys

cfg = Config()
proc = cfg.processed_dir
mm_path = next(iter(list(proc.glob("mm_master_clean.parquet")) + list(proc.glob("mm_master*.parquet"))), None)
if mm_path is None:
    print("mm_master parquet not found; run ETL first")
    sys.exit(1)

print("Loading mm from:", mm_path)
mm = pd.read_parquet(mm_path)
mm = filter_and_scale_mm(mm, cfg.map_bounds)

maps_in_mm = set(mm["_map"].unique())
maps_to_process = sorted(list(maps_in_mm & set(cfg.map_bounds.keys())))

print("Maps to process:", maps_to_process)

# stronger defaults for visibility
DEFAULTS = {
    "use_log": True,           # use log(1+heat) to boost low-density areas
    "vmax_percentile": 98.0,   # percentile for vmax
    "alpha_scale": 3.0,        # increase per-pixel alpha multiplier
    "blur": 4,
    "gamma": 0.85,             # slightly brighten low intensities
    "composite_alpha": 0.9     # global overlay opacity when compositing
}

for m in maps_to_process:
    print("Processing", m)
    map_png = cfg.maps_dir / f"{m}.png"
    has_png = map_png.exists()
    overlay = proc / "overlays" / f"{m}_heat.png"
    composite_path = proc / "overlays" / f"{m}_composite.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    try:
        make_heatmap_overlay(
            mm, m, map_png if has_png else proc / "overlays" / f"{m}_placeholder.png",
            overlay, cfg.map_bounds,
            bins=cfg.map_bounds[m].get("width", 1024),
            use_log=DEFAULTS["use_log"],
            vmax_percentile=DEFAULTS["vmax_percentile"],
            alpha_scale=DEFAULTS["alpha_scale"],
            blur=DEFAULTS["blur"],
            gamma=DEFAULTS["gamma"]
        )

        # NEW: permanently boost overlay transparency
        amplify_overlay_alpha(overlay, factor=3.0)  # <-- this line

    except Exception as e:
        print("  overlay FAILED for", m, ":", e)
        continue

    if has_png:
        composite_heatmap_on_map(map_png, overlay, composite_path, alpha=DEFAULTS["composite_alpha"])
        print("  saved composite ->", composite_path)
    else:
        print("  overlay saved ->", overlay, "(no base PNG found for composite)")
print("All done.")
