# src/viz/heatmaps.py
from pathlib import Path
from PIL import Image
import numpy as np

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
