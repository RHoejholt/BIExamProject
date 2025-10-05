
import warnings
from typing import Optional, Iterable
import numpy as np
import pandas as pd

try:
    from ..etl.cleaning_and_merge import harmonize_weapon_names
except Exception:
    harmonize_weapon_names = None


def _choose_round_key_cols(df: pd.DataFrame):
    if {"file", "round"}.issubset(df.columns):
        return df["file"].astype(str) + "__r__" + df["round"].astype(str)
    if {"match_id", "round_no"}.issubset(df.columns):
        return df["match_id"].astype(str) + "__r__" + df["round_no"].astype(str)
    if {"_map", "round"}.issubset(df.columns):
        return df["_map"].astype(str) + "__r__" + df["round"].astype(str)
    return df.index.astype(str)


def add_opening_kill_flag(
    duels_df: pd.DataFrame,
    opening_threshold_ticks: int = 64,
    weapon_col_candidates: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Mark opening kills and attach who/what produced the opening kill.
    Primary opening_kill is the earliest event in the round (event_rank == 1).
    Adds:
      - opening_kill (bool)
      - opening_kill_team (str)
      - opening_kill_attacker_id (str)
      - opening_kill_weapon (str)
    Note: this version intentionally does NOT create an opening_kill_is_early diagnostic column.
    """
    if duels_df is None or duels_df.empty:
        return duels_df

    df = duels_df.copy()
    df["_round_key"] = _choose_round_key_cols(df)
    df["_event_rank"] = df.groupby("_round_key").cumcount() + 1
    df["opening_kill"] = df["_event_rank"] == 1

    # attacker/team and id
    attacker_team_col = next((c for c in ["att_team", "attacker_team", "att_team_name", "team"] if c in df.columns), None)
    attacker_id_col = next((c for c in ["att_id", "attacker_id", "attacker", "attackerSteamId", "attacker_steam_id"] if c in df.columns), None)

    df["opening_kill_team"] = df[attacker_team_col].where(df["opening_kill"], other=pd.NA) if attacker_team_col else pd.NA
    df["opening_kill_attacker_id"] = df[attacker_id_col].where(df["opening_kill"], other=pd.NA) if attacker_id_col else pd.NA

    # weapon canonicalization
    if weapon_col_candidates is None:
        weapon_col_candidates = ["wp_canon", "wp", "weapon", "wp_type", "weapon_name"]
    found_wp = next((c for c in weapon_col_candidates if c in df.columns), None)
    if found_wp:
        if harmonize_weapon_names is not None:
            try:
                df = harmonize_weapon_names(df, col=found_wp)
                wp_canon_col = f"{found_wp}_canon" if f"{found_wp}_canon" in df.columns else found_wp
            except Exception:
                wp_canon_col = found_wp
        else:
            wp_canon_col = found_wp
        df["opening_kill_weapon"] = df[wp_canon_col].where(df["opening_kill"], other=pd.NA)
    else:
        df["opening_kill_weapon"] = pd.NA

    # tidy: drop internal helper column
    df = df.drop(columns=[c for c in ["_event_rank"] if c in df.columns])

    return df
