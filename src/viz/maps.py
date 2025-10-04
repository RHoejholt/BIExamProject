from PIL import Image, ImageDraw
import numpy as np

def draw_points_on_radar(radar_image_path: str, xs: list, ys: list, out_path: str):
    """
    Simple raster overlay: draw small circles on radar image.
    xs,ys are pixel coordinates.
    """
    img = Image.open(radar_image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    for x, y in zip(xs, ys):
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255,0,0,150))
    img.save(out_path)
