# MP2

# Quick runbook: ETL → Train model → Generate visuals

Prerequirements 
- Python 3.10+
- Install project requirements:
- 
    python -m venv .venv       # or use conda create -n bi venv python=3.13
- 
    .venv\Scripts\activate     # Windows
- 
    pip install -r requirements.txt

Run ETL (load & clean raw -> processed)
---------------------------------------
1. Clone this project

2. Download the two chosen datasets, 
    
    https://www.kaggle.com/datasets/skihikingkevin/csgo-matchmaking-damage/data
    
    https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches
    
    and drag the contained .csv files into their respective folders inside data/raw.


3.  Open anaconda prompt

4.  Navigate to project root

5.  Windows:
      venv\Scripts\activate 
    mac/linux
      source venv/bin/activate

6.  python -m src.etl.etl_pipeline

This writes cleaned parquet files to `data/processed/` such as:
    data/processed/mm_master_clean.parquet
    data/processed/players_clean.parquet
    data/processed/economy_clean.parquet
    data/processed/merged_professional.parquet  (if present)

Generate heatmap overlays (optional)
-----------------------------------

Example script to generate overlays (existing):
    python scripts/genrate_all_heatmaps.py