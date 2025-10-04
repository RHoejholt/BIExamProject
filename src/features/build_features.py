import pandas as pd

def add_opening_kill_flag(duels_df: pd.DataFrame, opening_threshold_ticks: int = 64) -> pd.DataFrame:
    """
    A tiny example: mark events that are 'opening' by tick/time threshold.
    Many datasets have a 'tick' or 'time' column — adapt as needed.
    """
    if duels_df is None or duels_df.empty:
        return duels_df
    df = duels_df.copy()
    if "tick" in df.columns:
        df["opening_kill"] = df["tick"] <= opening_threshold_ticks
    elif "time" in df.columns:
        df["opening_kill"] = df["time"] <= 5.0  # 5 seconds example
    else:
        # fallback: first two events in a round -> opening
        df["event_rank"] = df.groupby(["match_id", "round_no"]).cumcount() + 1
        df["opening_kill"] = df["event_rank"] <= 2
        df = df.drop(columns=["event_rank"])
    return df
