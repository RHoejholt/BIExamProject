from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import Config
from src.etl.map_utils import filter_and_scale_mm

# -------------------- Tunable parameters --------------------
USE_LOG = True                # apply log(1 + counts)
VMAX_PERCENTILE = 98.0        # percentile for clamping heat
ALPHA_MULTIPLIER = 0.72       # 0.6 * 1.2 → amplified by 120%
ALPHA_POWER = 1.0             # shaping of alpha curve
GAMMA = 0.9                   # gamma correction
# -------------------------------------------------------------

# Custom color map: blue → yellow → orange → red → white
CUSTOM_COLORS = ["#0000FF", "#FFFF00", "#FF8000", "#FF0000", "#FFFFFF"]
COLMAP = LinearSegmentedColormap.from_list("blue_yellow_orange_red_white", CUSTOM_COLORS, N=256)


def make_overlay(xs, ys, width=1024, height=1024):
    """Build RGBA numpy array overlay (heatmap)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]

    if len(xs) == 0:
        return np.zeros((height, width, 4), dtype=np.uint8)

    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    heat, _, _ = np.histogram2d(ys, xs, bins=[height, width], range=[[0, height], [0, width]])
    if USE_LOG:
        heat = np.log1p(heat)

    vmax = np.percentile(heat, VMAX_PERCENTILE) if np.any(heat > 0) else 1.0
    if vmax <= 0:
        vmax = heat.max() if heat.max() > 0 else 1.0

    norm = np.clip(heat / float(vmax), 0.0, 1.0)
    norm_rgb = np.power(norm, GAMMA)

    rgba_f = COLMAP(norm_rgb)
    alpha = np.clip(np.power(norm, ALPHA_POWER) * ALPHA_MULTIPLIER, 0.0, 1.0)

    rgba = np.empty((heat.shape[0], heat.shape[1], 4), dtype=np.float32)
    rgba[:, :, :3] = rgba_f[:, :, :3]
    rgba[:, :, 3] = alpha

    return (rgba * 255).astype(np.uint8)


def composite_on_map(map_png: Path, overlay_arr: np.ndarray, out_path: Path):
    """Composite the heatmap overlay onto a base map image."""
    base = Image.open(map_png).convert("RGBA")
    base_w, base_h = base.size

    overlay_img = Image.fromarray(overlay_arr.astype(np.uint8)).convert("RGBA")
    if (overlay_img.width, overlay_img.height) != (base_w, base_h):
        overlay_img = overlay_img.resize((base_w, base_h), resample=Image.BILINEAR)

    arr = np.array(overlay_img).astype(np.int32)
    arr[:, :, 3] = np.clip(arr[:, :, 3], 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")

    composite = Image.alpha_composite(base, overlay_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)
    return out_path


def main():
    cfg = Config()
    proc = Path(cfg.processed_dir)
    mm_path = next(iter(list(proc.glob("mm_master_clean.parquet")) + list(proc.glob("mm_master*.parquet"))), None)
    if mm_path is None:
        print("mm_master parquet not found; run ETL first")
        sys.exit(1)

    print("Loading:", mm_path)
    mm = pd.read_parquet(mm_path)

    try:
        mm = filter_and_scale_mm(mm, cfg.map_bounds)
    except Exception:
        pass

    maps = sorted(mm.get("_map", mm.get("map", pd.Series(dtype=str))).dropna().unique().tolist())
    maps_to_process = [m for m in maps if m in cfg.map_bounds]
    if not maps_to_process:
        print("No maps to process. Found:", maps)
        sys.exit(0)

    overlays_dir = proc / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    for m in maps_to_process:
        print("Processing map:", m)
        mp = mm[mm.get("_map", mm.get("map")) == m]
        if mp.empty:
            print("  no rows for map:", m)
            continue

        width = int(cfg.map_bounds.get(m, {}).get("width", 1024))
        height = int(cfg.map_bounds.get(m, {}).get("height", 1024))

        xcol = "x_map" if "x_map" in mp.columns else "x"
        ycol = "y_map" if "y_map" in mp.columns else "y"
        if xcol not in mp.columns or ycol not in mp.columns:
            print("  missing x_map/y_map columns for", m)
            continue

        xs, ys = mp[xcol].astype(float).to_numpy(), mp[ycol].astype(float).to_numpy()

        overlay_arr = make_overlay(xs, ys, width=width, height=height)
        map_png = cfg.maps_dir / f"{m}.png"
        composite_path = overlays_dir / f"{m}_composite.png"

        if map_png.exists():
            composite_on_map(map_png, overlay_arr, composite_path)
            print(f"  composite saved: {composite_path}")
        else:
            print(f"  base PNG missing for {m}")

    print("Done.")


if __name__ == "__main__":
    main()
