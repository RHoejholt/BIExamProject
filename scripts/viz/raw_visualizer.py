from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import sys
from src.config import Config
from src.etl.map_utils import amplify_overlay_alpha
from src.etl.map_utils import filter_and_scale_mm, make_heatmap_overlay



def composite_heatmap_on_map(map_png: Path, overlay_png: Path, out_path: Path, alpha: float = 0.75):
    """
    Composite overlay PNG (RGBA) over the base map PNG with specified global alpha in [0,1].
    Alpha multiplies overlay's per-pixel alpha: final_alpha = overlay_alpha * alpha.
    """
    base = Image.open(map_png).convert("RGBA")
    overlay = Image.open(overlay_png).convert("RGBA")

    # Resize overlay to base if mismatch
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, resample=Image.BILINEAR)

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

    # Multiply overlay alpha channel by alpha (preserving per-pixel variation)
    overlay_arr = np.array(overlay)
    # overlay_arr shape H,W,4
    alpha_channel = overlay_arr[:, :, 3].astype(np.float32)  # 0..255
    alpha_channel = np.clip(alpha_channel * float(alpha), 0, 255).astype(np.uint8)
    overlay_arr[:, :, 3] = alpha_channel
    overlay = Image.fromarray(overlay_arr, mode="RGBA")

    composite = Image.alpha_composite(base, overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)
    return out_path


# Main script

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
    "use_log": True,  # use log(1+heat) to boost low-density areas
    "vmax_percentile": 98.0,  # percentile for vmax
    "alpha_scale": 3.0 * 1.5,  # Amplified alpha scale (150% increase)
    "blur": 4,
    "gamma": 0.85,  # slightly brighten low intensities
    "composite_alpha": 0.9  # global overlay opacity when compositing
}

for m in maps_to_process:
    print("Processing", m)
    map_png = cfg.maps_dir / f"{m}.png"
    has_png = map_png.exists()
    overlay = proc / "overlays" / f"{m}_heat.png"
    composite_path = proc / "overlays" / f"{m}_composite.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Generate the heatmap overlay with the modified alpha scale
        make_heatmap_overlay(
            mm, m, map_png if has_png else proc / "overlays" / f"{m}_placeholder.png",
            overlay, cfg.map_bounds,
            bins=cfg.map_bounds[m].get("width", 1024),
            use_log=DEFAULTS["use_log"],
            vmax_percentile=DEFAULTS["vmax_percentile"],
            alpha_scale=DEFAULTS["alpha_scale"],  # Amplified scale here
            blur=DEFAULTS["blur"],
            gamma=DEFAULTS["gamma"]
        )

        # Amplify overlay transparency (as needed)
        amplify_overlay_alpha(overlay, factor=3.0)  # You can adjust this factor too if needed

    except Exception as e:
        print("  overlay FAILED for", m, ":", e)
        continue

    if has_png:
        composite_heatmap_on_map(map_png, overlay, composite_path, alpha=DEFAULTS["composite_alpha"])
        print("  saved composite ->", composite_path)
    else:
        print("  overlay saved ->", overlay, "(no base PNG found for composite)")

print("All done.")
