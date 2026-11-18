"""Fake Premier League seasons with the same columns as data/processed/matches.csv.

Used only by the tests, so the code can be checked without internet.
Team strengths are hidden; goals are Poisson. Do NOT use for real results.
"""
import numpy as np
import pandas as pd

from src.load_data import season_codes


def make_fake_matches(seed=0, first=2005, last=2024):
    rng = np.random.default_rng(seed)
    pool = [f"Team{i:02d}" for i in range(30)]
    strength = {t: rng.normal(0, 0.35) for t in pool}
    current = pool[:20]
    rows, mid = [], 0
    for s_idx, code in enumerate(season_codes(first, last)):
        year = first + s_idx
        if s_idx > 0:                                      # relegate 3 weakest-ish, promote 3
            ranked = sorted(current, key=lambda t: strength[t] + rng.normal(0, 0.2))
            out = ranked[:3]
            outside = [t for t in pool if t not in current]
            inn = list(rng.choice(outside, 3, replace=False))
            current = [t for t in current if t not in out] + inn
        fixtures = [(h, a) for h in current for a in current if h != a]
        rng.shuffle(fixtures)
        start = pd.Timestamp(f"{year}-08-10")
        for i, (h, a) in enumerate(fixtures):
            date = start + pd.Timedelta(days=int(i / 380 * 280))
            lam_h = np.exp(0.25 + 0.1 + strength[h] - strength[a])
            lam_a = np.exp(0.1 + strength[a] - strength[h])
            hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
            res = "H" if hg > ag else "A" if ag > hg else "D"
            hst, ast = hg + rng.poisson(3), ag + rng.poisson(2.5)
            ph = 1 / (1 + np.exp(-(strength[h] - strength[a] + 0.3) * 2)) * 0.75
            pa = 0.75 - ph
            pdraw = 0.25
            m = 1.05
            rows.append({"match_id": mid, "season": code, "date": date, "home": h, "away": a,
                         "hg": hg, "ag": ag, "result": res, "label": {"H": 0, "D": 1, "A": 2}[res],
                         "hs": hst + rng.poisson(6), "as_": ast + rng.poisson(5), "hst": hst, "ast": ast,
                         "odds_h": 1 / (ph * m), "odds_d": 1 / (pdraw * m), "odds_a": 1 / (pa * m)})
            mid += 1
    m = pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True)
    m["match_id"] = range(len(m))
    return m
