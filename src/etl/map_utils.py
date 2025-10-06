import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from PIL import Image, ImageFilter

def filter_and_scale_mm(mm_df: pd.DataFrame, map_bounds: Dict[str, Dict], map_name_col: str = "_map", x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    if mm_df is None or mm_df.empty:
        return mm_df
    mm = mm_df.copy()
    if map_name_col not in mm.columns and "map" in mm.columns:
        mm[map_name_col] = mm["map"].astype(str).str.strip().str.lower()
    else:
        mm[map_name_col] = mm[map_name_col].astype(str).str.strip().str.lower()
    allowed = set(map_bounds.keys())
    mm = mm[mm[map_name_col].isin(allowed)].copy()
    mm[x_col] = pd.to_numeric(mm[x_col], errors="coerce")
    mm[y_col] = pd.to_numeric(mm[y_col], errors="coerce")
    mm["x_map"] = pd.NA
    mm["y_map"] = pd.NA
    for map_name, b in map_bounds.items():
        mask = mm[map_name_col] == map_name
        if not mask.any(): continue
        denom_x = (b["xmax"] - b["xmin"]) or 1.0
        denom_y = (b["ymax"] - b["ymin"]) or 1.0
        xs = mm.loc[mask, x_col].astype(float)
        ys = mm.loc[mask, y_col].astype(float)
        x_map = (xs - b["xmin"]) / denom_x * b["width"]
        y_map = (ys - b["ymin"]) / denom_y * b["height"]
        y_map = b["height"] - y_map
        mm.loc[mask, "x_map"] = x_map
        mm.loc[mask, "y_map"] = y_map
    mm = mm.dropna(subset=["x_map", "y_map"], how="any").reset_index(drop=True)
    return mm

def amplify_overlay_alpha(overlay_path: str | Path, factor: float = 3.0, out_path: str | Path | None = None):
    """
    Multiply alpha channel by `factor` to make overlay less transparent.
    If out_path is None, overwrite the input file.
    """
    p = Path(overlay_path)
    img = Image.open(p).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3].astype(np.float32) * factor
    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    out_p = Path(out_path) if out_path else p
    Image.fromarray(arr, "RGBA").save(out_p)
    return out_p

def _simple_colormap(norm_arr: np.ndarray) -> np.ndarray:
    n = np.clip(norm_arr, 0.0, 1.0)
    r = (np.clip((n - 0.4) * 1.6, 0, 1) * 255).astype(np.uint8)
    g = (np.clip((n - 0.1) * 1.2, 0, 1) * 255).astype(np.uint8)
    b = (np.clip((1.0 - n) * 1.0, 0, 1) * 200).astype(np.uint8)
    return np.dstack([r, g, b])


def make_heatmap_overlay(
    mm_df: pd.DataFrame,
    map_name: str,
    map_png_path: Path,
    out_path: Path,
    map_bounds: Dict[str, Dict],
    bins: Optional[int] = None,
    use_log: bool = False,
    vmax_percentile: float = 99.0,
    alpha_scale: float = 2.0,
    blur: Optional[int] = 6,
    gamma: float = 1.0
):
    """
    Create an RGBA heatmap overlay with adjustable opacity/contrast.
    """
    if mm_df is None or mm_df.empty:
        raise ValueError("mm_df empty")
    mm = mm_df[mm_df["_map"] == map_name]
    if mm.empty:
        raise ValueError(f"No rows for map {map_name}")
    b = map_bounds[map_name]
    width = int(b.get("width", 1024))
    height = int(b.get("height", 1024))
    bins_x = width if bins is None else int(bins)
    bins_y = height if bins is None else int(bins)

    x = mm["x_map"].astype(float).clip(0, width - 1).to_numpy()
    y = mm["y_map"].astype(float).clip(0, height - 1).to_numpy()

    heat, xedges, yedges = np.histogram2d(y, x, bins=[height, width], range=[[0, height], [0, width]])
    heat = heat[::-1, :]

    if use_log:
        heat = np.log1p(heat)

    vmax = np.percentile(heat, vmax_percentile) if heat.size else 1.0
    if vmax <= 0:
        vmax = heat.max() if heat.max() > 0 else 1.0
    norm = heat / float(vmax)
    norm = np.clip(norm, 0.0, 1.0)

    if gamma != 1.0:
        norm = np.power(norm, gamma)

    rgb = _simple_colormap(norm)  # uint8

    alpha = np.clip(np.sqrt(norm) * alpha_scale, 0.0, 1.0)
    alpha_arr = (alpha * 255).astype(np.uint8)

    rgba = np.dstack([rgb, alpha_arr])
    img = Image.fromarray(rgba, mode="RGBA")

    if blur and blur > 0:
        try:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        except Exception:
            pass

    try:
        base = Image.open(map_png_path)
        if base.size != img.size:
            img = img.resize(base.size, resample=Image.BILINEAR)
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path

