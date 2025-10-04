from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    raw_dir: Path = project_root / "data" / "raw"
    interim_dir: Path = project_root / "data" / "interim"
    processed_dir: Path = project_root / "data" / "processed"
    maps_dir: Path = project_root / "maps"
    map_bounds: Dict[str, Dict] = None

    def __post_init__(self):
        # expanded default bbox (conservative)
        default_bbox = {"xmin": -3000.0, "xmax": 3000.0,
                        "ymin": -3000.0, "ymax": 3000.0}
        width = 1024
        height = 1024

        # hardcoded exact map bounds (from your CSV)
        csv_bounds = {
            "de_cache": {
                "xmin": -2031.0, "xmax": 3752.0,
                "ymin": -2240.0, "ymax": 3187.0,
                "width": width, "height": height,
            },
            "de_cbble": {
                "xmin": -3819.0, "xmax": 2282.0,
                "ymin": -3073.0, "ymax": 3032.0,
                "width": width, "height": height,
            },
            "de_dust2": {
                "xmin": -2486.0, "xmax": 2127.0,
                "ymin": -1150.0, "ymax": 3455.0,
                "width": width, "height": height,
            },
            "de_inferno": {
                "xmin": -1960.0, "xmax": 2797.0,
                "ymin": -1062.0, "ymax": 3800.0,
                "width": width, "height": height,
            },
            "de_mirage": {
                "xmin": -3217.0, "xmax": 1912.0,
                "ymin": -3401.0, "ymax": 1682.0,
                "width": width, "height": height,
            },
            "de_overpass": {
                "xmin": -4820.0, "xmax": 503.0,
                "ymin": -3591.0, "ymax": 1740.0,
                "width": width, "height": height,
            },
            "de_train": {
                "xmin": -2436.0, "xmax": 2262.0,
                "ymin": -2469.0, "ymax": 2447.0,
                "width": width, "height": height,
            },
        }

        bounds: Dict[str, Dict] = {}
        if self.maps_dir.exists() and self.maps_dir.is_dir():
            for p in sorted(self.maps_dir.glob("*.png")):
                name = p.stem.lower()
                if name in csv_bounds:
                    bounds[name] = csv_bounds[name]
                else:
                    bounds[name] = {
                        "xmin": default_bbox["xmin"],
                        "xmax": default_bbox["xmax"],
                        "ymin": default_bbox["ymin"],
                        "ymax": default_bbox["ymax"],
                        "width": width, "height": height,
                    }

        # fallback if no PNGs found
        if not bounds:
            bounds = csv_bounds or {
                "de_dust2": {**default_bbox, "width": width, "height": height},
                "de_mirage": {**default_bbox, "width": width, "height": height},
            }

        object.__setattr__(self, "map_bounds", bounds)
