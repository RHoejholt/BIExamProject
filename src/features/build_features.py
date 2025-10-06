import pandas as pd
from typing import Optional, Iterable

def add_opening_kill_flag(duels_df: pd.DataFrame, opening_threshold_ticks: int = 64, weapon_col_candidates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    # do feature stuff: mark første event i hver round som opening_kill
    if duels_df is None or duels_df.empty:
        return duels_df
    df = duels_df.copy()
    if {"file","round"}.issubset(df.columns):
        df["_round_key"] = df["file"].astype(str) + "__r__" + df["round"].astype(str)
    elif {"match_id","round_no"}.issubset(df.columns):
        df["_round_key"] = df["match_id"].astype(str) + "__r__" + df["round_no"].astype(str)
    else:
        df["_round_key"] = df.index.astype(str)
    df["_event_rank"] = df.groupby("_round_key").cumcount() + 1
    df["opening_kill"] = df["_event_rank"] == 1

    attacker_team_col = next((c for c in ["att_team","attacker_team","team"] if c in df.columns), None)
    attacker_id_col = next((c for c in ["att_id","attacker","attackerSteamId"] if c in df.columns), None)
    df["opening_kill_team"] = df[attacker_team_col].where(df["opening_kill"], other=pd.NA) if attacker_team_col else pd.NA
    df["opening_kill_attacker_id"] = df[attacker_id_col].where(df["opening_kill"], other=pd.NA) if attacker_id_col else pd.NA

    if weapon_col_candidates is None:
        weapon_col_candidates = ["opening_kill_weapon", "wp_canon", "wp", "weapon"]
    wp_col = next((c for c in weapon_col_candidates if c in df.columns), None)
    if wp_col:
        df["opening_kill_weapon"] = df[wp_col].where(df["opening_kill"], other=pd.NA)
    else:
        df["opening_kill_weapon"] = pd.NA

    df = df.drop(columns=[c for c in ("_event_rank",) if c in df.columns])
    return df
