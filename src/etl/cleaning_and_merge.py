import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# A simple canonical weapon map; extend as you find more aliases in your data
WEAPON_CANONICAL = {
    # pistols
    "glock": "glock",
    "us-p": "usps",
    "usp_s": "usps",
    "p250": "p250",
    "desert_eagle": "deagle",
    "deagle": "deagle",
    # rifles
    "ak47": "ak47",
    "ak-47": "ak47",
    "m4a1": "m4a1",
    "m4-a1": "m4a1",
    "m4a4": "m4a4",
    "m4-a4": "m4a4",
    # SMGs
    "mp9": "mp9",
    "p90": "p90",
    # snipers
    "awp": "awp",
}


def harmonize_weapon_names(df: pd.DataFrame, col: str = "weapon") -> pd.DataFrame:
    """
    Map weapon name variants to canonical names. Non-matches are lowercased.
    """
    if col not in df.columns:
        return df
    df = df.copy()
    def _canon(w):
        if pd.isna(w):
            return w
        s = str(w).lower()
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return WEAPON_CANONICAL.get(s, s)
    df[col + "_canon"] = df[col].apply(_canon)
    return df


def normalize_ids(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """
    Ensure IDs (steam IDs or numeric ids) are strings and trimmed. Also replace
    missing or 0 values with None.
    """
    df = df.copy()
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].where(pd.notna(df[c]), None)
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "None": None})
    return df


RANK_MAP = {
  "silver_1": 1, "silver_2": 2, "silver_3": 3, "silver_4": 4,
  "silver_elite": 5, "silver_elite_master": 6,
  "gold_nova_1": 7, "gold_nova_2": 8, "gold_nova_3": 9, "gold_nova_master": 10,
  "master_guardian_1": 11, "master_guardian_2": 12, "master_guardian_elite": 13,
  "distinguished_master_guardian": 14,
  "legendary_eagle": 15, "legendary_eagle_master": 16,
  "supreme_master_first_class": 17, "global_elite": 18,
}



def parse_rank_field(df: pd.DataFrame, col: str = "rank") -> pd.DataFrame:
    """
    If your players table contains a rank string, try to map to an integer
    ordinal. If the column is missing leave untouched.
    """
    if col not in df.columns:
        return df
    df = df.copy()
    def _map_rank(x):
        if pd.isna(x):
            return None
        s = str(x).lower().replace(" ", "_")
        return RANK_MAP.get(s, None)
    df[col + "_ordinal"] = df[col].apply(_map_rank)
    return df


def expand_economy_wide_to_long(econ_df: pd.DataFrame, round_prefixes: Optional[List[str]] = None) -> pd.DataFrame:
    if econ_df is None or econ_df.empty:
        return econ_df
    df = econ_df.copy()

    # find candidate round columns: numeric or like '1_t1' '1_t2'
    candidate_cols = [c for c in df.columns if re.match(r"^\d+(_t[12])?$", str(c))]
    if not candidate_cols:
        # fallback: look for columns starting with digit
        candidate_cols = [c for c in df.columns if str(c)[0:1].isdigit()]
    if not candidate_cols:
        # nothing to expand
        return df

    # we'll build a long-format DataFrame
    rows = []
    for _, row in df.iterrows():
        base = row.to_dict()
        match_id = base.get("match_id")
        map_name = base.get("_map")
        for col in candidate_cols:
            # Determine round number and team if encoded
            m = re.match(r"^(\d+)(_t([12]))?$", str(col))
            if not m:
                continue
            round_no = int(m.group(1))
            team_suffix = m.group(3)
            team = f"t{team_suffix}" if team_suffix else None
            val = row[col]
            rows.append({"match_id": match_id, "map": map_name, "round_no": round_no, "team_slot": team, "value": val})

    long_df = pd.DataFrame(rows)
    return long_df


def merge_players_results_economy(players: pd.DataFrame, results: pd.DataFrame, economy: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a unified table linking player-level information with match/map/round
    economy and match results. This is a pragmatic merge: merge players -> results
    on match_id and team, then optionally left-join economy on match_id/_map.

    The exact joins depend heavily on column names in your files. This function
    attempts reasonable defaults and keeps fields prefixed to avoid collisions.
    """
    # defensive copies
    p = players.copy()
    r = results.copy()
    e = economy.copy() if economy is not None else None

    # ensure common join keys exist
    for df, name in [(p, "players"), (r, "results")]:
        if "match_id" not in df.columns:
            raise ValueError(f"{name} missing match_id column")

    # If players contain team name column, try to match with results team columns
    # Heuristic: match on match_id + team name if possible
    team_cols = [c for c in r.columns if c.lower().startswith("team")]
    # choose common columns
    common = ["match_id"]

    # merge players with results (left join by match_id; more complex matching can be added)
    merged = p.merge(r.add_prefix("res_"), left_on="match_id", right_on="res_match_id", how="left")

    # attach economy aggregated per match/map if available
    if e is not None and ("match_id" in e.columns or "res_match_id" in e.columns):
        # if economy has many per-round columns, keep as-is or convert to long
        merged = merged.merge(e.add_prefix("eco_"), left_on="match_id", right_on="eco_match_id", how="left")

    return merged

