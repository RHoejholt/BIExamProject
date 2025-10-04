from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    raw_dir: Path = project_root / "data" / "raw"
    interim_dir: Path = project_root / "data" / "interim"
    processed_dir: Path = project_root / "data" / "processed"

    # map scaling parameters (example values; put maps params here later)
    map_bounds: dict = None

    def __post_init__(self):
        # Default map bounds dictionary for coordinate scaling.
        object.__setattr__(self, "map_bounds", {
            # example: 'de_dust2': {'xmin': -2500, 'xmax': 2500, 'ymin': -2500, 'ymax': 2500, 'width': 1024, 'height': 1024}
        })
