MP3 Machine learning


# Quick runbook: ETL → Train model → Generate visuals

Prerequirements 
- Python 3.10+
- Install project requirements:
    python -m venv .venv       # or use conda create -n bi venv python=3.13
    .venv\Scripts\activate     # Windows
    pip install -r requirements.txt

Run ETL (load & clean raw -> processed)
---------------------------------------
1. Clone this project

2. Make surey ou have the processed data files from mp2
    

3.  Open anaconda prompt

4.  Navigate to project root

5.  Windows:
      venv\Scripts\activate 
    mac/linux
      source venv/bin/activate

6.  python -m src.etl.etl_pipeline

Train the ML model
------------------
Train the "first-kill wins" binary model and save artifact:
    python src/models/train.py

This writes artifact(s) to `models/mm_firstkill_binary.joblib` (and metrics JSON).

Predict
---------------------------------------------
The model can now be used to make predictions on winning team.

python src/models/predict.py


Visualize predictions / generate figures
----------------------------------------
Several scripts are provided for visuals. 

First elimination prediction vs. actual visualisation and winrate by weapon:
python sripts/viz/firstkill_predict_viz.py



Outputs (images and parquets) go to: data/processed/