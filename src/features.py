"""Pre-match features + the column lists for Setups A-D.

Run from the repo root (after src.load_data):
    python -m src.features
Output: data/processed/features.csv

Every rolling feature uses shift(1) = "only matches BEFORE this one". That is what stops leakage.
Owners: Darareach (form, venue form, h2h, elo), Lyzing (rest days, shots), Cheypol (setup A stats)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.elo import add_elo
from src.h2h import add_h2h

OUT_FILE = Path("data/processed/features.csv")

TEAM_FEATS = ["form_pts5", "form_gf5", "form_ga5", "sot_ratio5", "rest_days",
              "venue_pts5", "ppg_season", "gf_season", "ga_season"]


def _both(cols):
    """home_, away_ and diff_ versions of each team feature."""
    return [f"{p}_{c}" for c in cols for p in ("home", "away", "diff")]


# ---- Setups (Section 6.4 of the gameplan) ---------------------------------------------
SETUP_A = _both(["ppg_season", "gf_season", "ga_season"])
SETUP_B = SETUP_A + ["elo_home", "elo_away", "elo_diff"] + _both(
    ["form_pts5", "form_gf5", "form_ga5", "venue_pts5", "rest_days", "sot_ratio5"])
SETUP_C = SETUP_B + ["h2h_score", "h2h_n"]
SETUP_D = SETUP_C + ["pois_pH", "pois_pD", "pois_pA"]
SETUPS = {"A": SETUP_A, "B": SETUP_B, "C": SETUP_C, "D": SETUP_D}


def to_long(m):
    """One row per team per match (home row + away row)."""
    h = pd.DataFrame({"match_id": m.match_id, "date": m.date, "season": m.season,
                      "team": m.home, "opp": m.away, "gf": m.hg, "ga": m.ag,
                      "sot_f": m.hst, "sot_a": m.ast, "is_home": 1})
    a = pd.DataFrame({"match_id": m.match_id, "date": m.date, "season": m.season,
                      "team": m.away, "opp": m.home, "gf": m.ag, "ga": m.hg,
                      "sot_f": m.ast, "sot_a": m.hst, "is_home": 0})
    long = pd.concat([h, a]).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    return long


def add_team_features(long):
    # per-game AVERAGE over the last 5 (works the same when a team has fewer than 5 games)
    past5 = lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    past_mean = lambda s: s.shift(1).expanding().mean()

    g = long.groupby("team")
    long["form_pts5"] = g["pts"].transform(past5)
    long["form_gf5"] = g["gf"].transform(past5)
    long["form_ga5"] = g["ga"].transform(past5)
    sot_f5 = g["sot_f"].transform(past5)
    sot_a5 = g["sot_a"].transform(past5)        # ratio of averages = ratio of sums
    long["sot_ratio5"] = sot_f5 / (sot_f5 + sot_a5)
    long["rest_days"] = g["date"].diff().dt.days.clip(upper=14)

    # last 5 HOME games for a home row, last 5 AWAY games for an away row
    long["venue_pts5"] = long.groupby(["team", "is_home"])["pts"].transform(past5)

    # Setup A: season-to-date averages before this match
    gs = long.groupby(["team", "season"])
    long["ppg_season"] = gs["pts"].transform(past_mean)
    long["gf_season"] = gs["gf"].transform(past_mean)
    long["ga_season"] = gs["ga"].transform(past_mean)
    return long


def fill_missing(long):
    """First match of a season has no season-to-date stats.
    Rule: use the team's full average from LAST season (known before the new season starts);
    teams with no last season (promoted) get the league-wide average of promoted-team-like values.
    Rest days on a team's first ever match = 7. Other early NaNs = fixed typical EPL values.

    NEVER fill with df[col].mean(): that average includes future matches = leakage.
    (tests/test_pipeline.py::test_no_leakage_from_future catches this.)
    """
    seasons = sorted(long["season"].unique())
    prev = dict(zip(seasons[1:], seasons[:-1]))
    full = long.groupby(["team", "season"])[["pts", "gf", "ga"]].mean()
    full.columns = ["ppg_season", "gf_season", "ga_season"]
    for col in full.columns:
        key = list(zip(long["team"], long["season"].map(prev)))
        last_season_val = pd.Series([full[col].get(k, np.nan) for k in key], index=long.index)
        long[col] = long[col].fillna(last_season_val)
    # promoted teams (no previous PL season): roughly a weak team
    long["ppg_season"] = long["ppg_season"].fillna(1.0)
    long["gf_season"] = long["gf_season"].fillna(1.1)
    long["ga_season"] = long["ga_season"].fillna(1.6)
    long["rest_days"] = long["rest_days"].fillna(7)
    typical = {"form_pts5": 1.37, "form_gf5": 1.37, "form_ga5": 1.37,
               "sot_ratio5": 0.5, "venue_pts5": 1.37}           # chosen in advance, not from data
    for col, val in typical.items():
        long[col] = long[col].fillna(val)
    return long


def build_features(m):
    long = fill_missing(add_team_features(to_long(m)))
    H = long[long.is_home == 1].set_index("match_id")[TEAM_FEATS].add_prefix("home_")
    A = long[long.is_home == 0].set_index("match_id")[TEAM_FEATS].add_prefix("away_")
    f = m.set_index("match_id").join(H).join(A)
    for c in TEAM_FEATS:
        f[f"diff_{c}"] = f[f"home_{c}"] - f[f"away_{c}"]
    f = f.reset_index()
    f = add_elo(f)
    f = add_h2h(f)
    return f.sort_values(["date", "home"]).reset_index(drop=True)


def main():
    from src.load_data import read_matches
    f = build_features(read_matches())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT_FILE, index=False)
    print(f"saved {len(f)} rows, {f.shape[1]} columns -> {OUT_FILE}")
    print("missing values in setup C columns:", int(f[SETUP_C].isna().sum().sum()))


if __name__ == "__main__":
    main()
