"""Construye data/processed/maps_dataset.csv: una fila por mapa jugado (VCT 2024-2026).

Une maps_scores (resultado), ids (Match ID / Game ID para ordenar en el tiempo),
overview (stats promedio de los 5 jugadores por equipo), eco_stats (economia),
draft_phase (quien eligio el mapa) y win_loss_methods_count (como se ganaron las rondas).

Variable objetivo: team_a_wins (1 si Team A gana el mapa, 0 si gana Team B).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "maps_dataset.csv"
YEARS = (2024, 2025, 2026)
KEY = ["Tournament", "Stage", "Match Type", "Match Name", "Map"]

# Stats de overview.csv que se promedian por equipo (Side == "both")
OVERVIEW_STATS = {
    "Rating": "rating",
    "Average Combat Score": "acs",
    "Kills": "kills",
    "Deaths": "deaths",
    "Assists": "assists",
    "Kill, Assist, Trade, Survive %": "kast",
    "Average Damage Per Round": "adr",
    "Headshot %": "hs_pct",
    "First Kills": "first_kills",
    "First Deaths": "first_deaths",
}
ECO_TYPES = {
    "Pistol Won": "pistol_won",
    "Eco (won)": "eco_won",
    "$ (won)": "semi_eco_won",
    "$$ (won)": "semi_buy_won",
    "$$$ (won)": "full_buy_won",
}
WIN_METHODS = {
    "Elimination": "win_elim",
    "Detonated": "win_detonated",
    "Defused": "win_defused",
    "Time Expiry (No Plant)": "win_time",
}


def read_years(rel_path: str) -> pd.DataFrame:
    """Concatena el mismo CSV de todos los años, añadiendo la columna year."""
    frames = []
    for y in YEARS:
        # keep_default_na=False evita que el jugador llamado "nan" se lea como nulo
        df = pd.read_csv(RAW / f"vct_{y}" / rel_path, keep_default_na=False, na_values=[""])
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pct_to_float(s: pd.Series) -> pd.Series:
    """'91%' -> 91.0"""
    return pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")


def duration_to_seconds(s: pd.Series) -> pd.Series:
    """'48:56' -> 2936 ; '1:02:10' -> 3730"""
    def conv(v):
        if pd.isna(v) or not isinstance(v, str) or ":" not in v:
            return np.nan
        parts = [int(p) for p in v.split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return s.map(conv)


def region_from_tournament(name: str) -> str:
    for r in ("Americas", "EMEA", "Pacific", "China"):
        if r in name:
            return r
    return "International"  # Masters / Champions


def stage_group(name: str) -> str:
    """Agrupa los stages en tres fases: liga/grupos, bracket eliminatorio y otros."""
    n = name.lower()
    if "playoff" in n or "main event" in n:
        return "Playoffs/Bracket"
    if "group" in n or "swiss" in n or "regular season" in n:
        return "Liga/Grupos"
    return "Otro"


def build() -> pd.DataFrame:
    # ---- 1. Resultado por mapa (base) --------------------------------------
    ms = read_years("matches/maps_scores.csv")
    ms = ms.rename(columns={
        "Team A": "team_a", "Team B": "team_b",
        "Team A Score": "score_a", "Team B Score": "score_b",
        "Team A Attacker Score": "atk_score_a", "Team A Defender Score": "def_score_a",
        "Team B Attacker Score": "atk_score_b", "Team B Defender Score": "def_score_b",
        "Team A Overtime Score": "ot_score_a", "Team B Overtime Score": "ot_score_b",
    })
    ms["overtime"] = ms["ot_score_a"].notna().astype(int)
    ms["duration_sec"] = duration_to_seconds(ms["Duration"])
    ms["total_rounds"] = ms["score_a"] + ms["score_b"]
    ms["round_diff"] = ms["score_a"] - ms["score_b"]
    ms["team_a_wins"] = (ms["score_a"] > ms["score_b"]).astype(int)
    ms = ms.drop(columns=["Duration"])

    # ---- 2. IDs (para ordenar temporalmente) -------------------------------
    ids = read_years("ids/tournaments_stages_matches_games_ids.csv")
    ids = ids[KEY + ["Tournament ID", "Stage ID", "Match ID", "Game ID"]].drop_duplicates(KEY)
    ids = ids.rename(columns={"Tournament ID": "tournament_id", "Stage ID": "stage_id",
                              "Match ID": "match_id", "Game ID": "game_id"})
    ms = ms.merge(ids, on=KEY, how="left")

    teams = pd.read_csv(RAW / "all_ids" / "all_teams_ids.csv").drop_duplicates("Team")
    team_id = dict(zip(teams["Team"], teams["Team ID"]))
    ms["team_a_id"] = ms["team_a"].map(team_id)
    ms["team_b_id"] = ms["team_b"].map(team_id)

    # ---- 3. Stats de jugadores -> promedio por equipo ----------------------
    ov = read_years("matches/overview.csv")
    ov = ov[ov["Side"] == "both"].copy()
    for col in ("Kill, Assist, Trade, Survive %", "Headshot %"):
        ov[col] = pct_to_float(ov[col])
    for col in OVERVIEW_STATS:
        ov[col] = pd.to_numeric(ov[col], errors="coerce")
    team_stats = (ov.groupby(KEY + ["Team"])[list(OVERVIEW_STATS)].mean()
                    .rename(columns=OVERVIEW_STATS).reset_index())
    ms = _merge_team_side(ms, team_stats, list(OVERVIEW_STATS.values()))

    # ---- 4. Economia -------------------------------------------------------
    eco = read_years("matches/eco_stats.csv")
    eco = eco[eco["Map"] != "All Maps"]
    eco_w = (eco.pivot_table(index=KEY + ["Team"], columns="Type", values="Won", aggfunc="sum")
                .rename(columns=ECO_TYPES).reset_index())
    eco_w = eco_w[KEY + ["Team"] + [c for c in ECO_TYPES.values() if c in eco_w]]
    ms = _merge_team_side(ms, eco_w, [c for c in ECO_TYPES.values() if c in eco_w])

    # ---- 5. Metodos de victoria de ronda -----------------------------------
    wl = read_years("matches/win_loss_methods_count.csv")
    wl = wl[wl["Map"] != "All Maps"][KEY + ["Team"] + list(WIN_METHODS)].rename(columns=WIN_METHODS)
    ms = _merge_team_side(ms, wl, list(WIN_METHODS.values()))

    # ---- 6. Draft: quien eligio el mapa ------------------------------------
    dr = read_years("matches/draft_phase.csv")
    picks = dr[dr["Action"] == "pick"][KEY + ["Team"]].drop_duplicates(KEY).rename(columns={"Team": "picked_by_team"})
    ms = ms.merge(picks, on=KEY, how="left")
    ms["map_picked_by"] = np.select(
        [ms["picked_by_team"] == ms["team_a"], ms["picked_by_team"] == ms["team_b"]],
        ["A", "B"], default="decider")
    ms = ms.drop(columns=["picked_by_team"])

    # ---- 7. Contexto y diferencias A - B -----------------------------------
    ms["region"] = ms["Tournament"].map(region_from_tournament)
    ms["stage_group"] = ms["Stage"].map(stage_group)
    ms["is_international"] = (ms["region"] == "International").astype(int)
    for stat in list(OVERVIEW_STATS.values()) + ["pistol_won"]:
        if f"{stat}_a" in ms and f"{stat}_b" in ms:
            ms[f"{stat}_diff"] = ms[f"{stat}_a"] - ms[f"{stat}_b"]

    ms = ms.rename(columns={"Tournament": "tournament", "Stage": "stage",
                            "Match Type": "match_type", "Match Name": "match_name", "Map": "map"})
    ms = ms.sort_values(["match_id", "game_id"]).reset_index(drop=True)

    first = ["year", "region", "tournament", "stage", "stage_group", "match_type", "match_name",
             "match_id", "game_id", "map", "map_picked_by", "team_a", "team_a_id", "team_b", "team_b_id",
             "score_a", "score_b", "total_rounds", "round_diff", "overtime", "duration_sec", "team_a_wins"]
    rest = [c for c in ms.columns if c not in first]
    return ms[first + rest]


def _merge_team_side(ms: pd.DataFrame, stats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Une stats por equipo dos veces: una para Team A (sufijo _a) y otra para Team B (_b)."""
    for side, team_col in (("a", "team_a"), ("b", "team_b")):
        s = stats.rename(columns={"Team": team_col, **{c: f"{c}_{side}" for c in cols}})
        ms = ms.merge(s, on=KEY + [team_col], how="left")
    return ms


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} mapas x {df.shape[1]} columnas -> {OUT.relative_to(ROOT)}")
    print(df.groupby("year").size().to_dict())
    print("team_a_wins balance:", df["team_a_wins"].mean().round(3))
