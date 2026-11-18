"""Elo ratings. The PRE-match rating is the feature; ratings update only after it is recorded.

Owner: Darareach
"""


def _run_elo(m, k, hfa, start, promoted, pull):
    first_season = m["season"].iloc[0]
    elo, last_season = {}, None
    pre_h, pre_a = [], []
    for row in m.itertuples():
        if row.season != last_season:                   # new season: regress to the mean
            elo = {t: r + pull * (start - r) for t, r in elo.items()}
            last_season = row.season
        for t in (row.home, row.away):
            if t not in elo:
                elo[t] = start if row.season == first_season else promoted
        rh, ra = elo[row.home], elo[row.away]
        pre_h.append(rh)                                # 1) record BEFORE updating
        pre_a.append(ra)
        exp_h = 1 / (1 + 10 ** (-(rh + hfa - ra) / 400))
        act_h = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        change = k * (act_h - exp_h)                    # 2) then update
        elo[row.home] = rh + change
        elo[row.away] = ra - change
    return pre_h, pre_a, elo


def add_elo(m, k=20, hfa=60, start=1500, promoted=1450, pull=0.2):
    """Add elo_home, elo_away, elo_diff to the matches table.

    k        how fast ratings move after each match
    hfa      home-field advantage (Elo points) used in the expected score
    start    rating for every team seen in the first season
    promoted rating for teams that first appear later (usually weaker)
    pull     at each new season, ratings move this share back toward `start`
    """
    m = m.sort_values(["date", "home"]).copy()          # MUST be in date order
    pre_h, pre_a, _ = _run_elo(m, k, hfa, start, promoted, pull)
    m["elo_home"], m["elo_away"] = pre_h, pre_a
    m["elo_diff"] = m["elo_home"] - m["elo_away"]
    return m


def final_ratings(m, k=20, hfa=60, start=1500, promoted=1450, pull=0.2):
    """Every team's rating AFTER the last match in the data (used by the app)."""
    m = m.sort_values(["date", "home"])
    return _run_elo(m, k, hfa, start, promoted, pull)[2]
