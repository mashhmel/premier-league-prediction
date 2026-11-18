"""Poisson goal model + walk-forward probabilities (for the Poisson baseline and Setup D).

Expected goals = exp(intercept + home + attack[team] + defence[opponent]), fitted as a
Poisson regression. Optional time-decay weights make recent matches count more.

Dixon-Coles (1997) upgrade: real football has slightly more 0-0 / 1-1 (and fewer 1-0 / 0-1)
than independent Poisson predicts. A correction factor tau with one parameter rho fixes the
four low-score cells. We use a two-step fit: (1) Poisson regression with time decay gives the
expected goals, (2) rho is chosen to maximise the (weighted) likelihood of the low scores
in the training window. Turn it on with walk_forward_probs(..., dixon_coles=True).

Owner: Chhoungkorng
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

OUT_FILE = Path("data/processed/poisson_probs.csv")
MAX_GOALS = 10


def _long_goals(df):
    return pd.concat([
        pd.DataFrame({"goals": df.hg.values, "team": df.home.values, "opp": df.away.values,
                      "home": 1, "date": df.date.values}),
        pd.DataFrame({"goals": df.ag.values, "team": df.away.values, "opp": df.home.values,
                      "home": 0, "date": df.date.values})], ignore_index=True)


def fit_poisson(train, xi=0.0, ref_date=None):
    """Fit on `train` matches. xi > 0 adds time decay: weight = exp(-xi * days_ago)
    (try xi between 0.0018 and 0.0065 per day)."""
    gl = _long_goals(train)
    weights = None
    if xi > 0:
        ref_date = ref_date or gl["date"].max()
        days = (pd.Timestamp(ref_date) - pd.to_datetime(gl["date"])).dt.days
        weights = np.exp(-xi * days.to_numpy())
    return smf.glm("goals ~ home + C(team) + C(opp)", data=gl,
                   family=sm.families.Poisson(), freq_weights=weights).fit()


def expected_goals(model, homes, aways):
    """Vectorised: expected home goals and away goals for lists of fixtures."""
    lam_h = model.predict(pd.DataFrame({"team": homes, "opp": aways, "home": 1}))
    lam_a = model.predict(pd.DataFrame({"team": aways, "opp": homes, "home": 0}))
    return np.asarray(lam_h), np.asarray(lam_a)


def dc_tau(x, y, lam_h, lam_a, rho):
    """Dixon-Coles correction for scores x-y (arrays OK). 1 for every score above 1-1."""
    x, y = np.asarray(x), np.asarray(y)
    t = np.ones(np.broadcast(x, y, lam_h, lam_a).shape)
    t = np.where((x == 0) & (y == 0), 1 - lam_h * lam_a * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam_h * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + lam_a * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def estimate_rho(train, model, xi=0.0, ref_date=None, bounds=(-0.2, 0.2)):
    """Step 2 of Dixon-Coles: pick rho that best explains the low scores in `train`
    (Poisson part fixed). Negative rho = more 0-0 and 1-1 than plain Poisson."""
    lh, la = expected_goals(model, train.home.values, train.away.values)
    w = np.ones(len(train))
    if xi > 0:
        ref_date = ref_date or train.date.max()
        w = np.exp(-xi * (pd.Timestamp(ref_date) - train.date).dt.days.to_numpy())
    x, y = train.hg.to_numpy(), train.ag.to_numpy()

    def neg_ll(rho):
        t = dc_tau(x, y, lh, la, rho)
        if np.any(t <= 0):
            return 1e12
        return -np.sum(w * np.log(t))

    return float(minimize_scalar(neg_ll, bounds=bounds, method="bounded").x)


def scoreline_grid(lam_h, lam_a, max_goals=MAX_GOALS, rho=None):
    """grid[i, j] = P(home scores i, away scores j). rho = Dixon-Coles correction (None = off)."""
    g = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    if rho:
        tau = dc_tau(g[:2, None], g[None, :2], lam_h, lam_a, rho)
        grid[:2, :2] *= np.clip(tau, 1e-6, None)      # tau can't go negative for big lambdas
        grid /= grid.sum()
    return grid


def outcome_probs_from_grid(grid):
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away                    # tiny mass lost above MAX_GOALS
    return p_home / total, p_draw / total, p_away / total


def top_scorelines(grid, n=5):
    cells = [(grid[i, j], f"{i}-{j}") for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    return sorted(cells, reverse=True)[:n]


def walk_forward_probs(m, window_days=730, xi=0.0, min_train=300, dixon_coles=False):
    """For every month: fit ONLY on matches before that month (last `window_days`),
    then predict that month. No leakage, so these probabilities can be features (Setup D).

    Teams the model has never seen (e.g. promoted team in August) cannot be predicted;
    those matches get NaN and `pois_missing = 1` (fill them later, see fill_missing_probs).

    Recommended: walk_forward_probs(m, xi=0.0019, dixon_coles=True)
    Output also has `pois_rho` (the rho used that month; 0 when Dixon-Coles is off).
    """
    m = m.sort_values("date")
    out = []
    for mon in m["date"].dt.to_period("M").unique():
        start = mon.to_timestamp()
        train = m[(m.date < start) & (m.date >= start - pd.Timedelta(days=window_days))]
        target = m[m.date.dt.to_period("M") == mon]
        if len(train) < min_train:
            continue
        model = fit_poisson(train, xi=xi, ref_date=start)
        rho = estimate_rho(train, model, xi=xi, ref_date=start) if dixon_coles else 0.0
        known = set(train.home) | set(train.away)
        ok = target[target.home.isin(known) & target.away.isin(known)]
        if len(ok):
            lh, la = expected_goals(model, ok.home.values, ok.away.values)
            good = np.isfinite(lh) & np.isfinite(la) & (lh < 10) & (la < 10)   # unstable fits -> missing
            ok, lh, la = ok[good], lh[good], la[good]
            for mid, a, b in zip(ok.match_id, lh, la):
                out.append((mid, *outcome_probs_from_grid(scoreline_grid(a, b, rho=rho)), a, b, rho, 0))
        for mid in target.match_id[~target.match_id.isin(ok.match_id)]:
            out.append((mid, np.nan, np.nan, np.nan, np.nan, np.nan, rho, 1))
    return pd.DataFrame(out, columns=["match_id", "pois_pH", "pois_pD", "pois_pA",
                                      "pois_xg_h", "pois_xg_a", "pois_rho", "pois_missing"])


def fill_missing_probs(probs, fallback=(0.45, 0.25, 0.30)):
    """Simple documented rule: unseen-team matches get historical H/D/A frequencies."""
    p = probs.copy()
    for col, val in zip(["pois_pH", "pois_pD", "pois_pA"], fallback):
        p[col] = p[col].fillna(val)
    return p


def main():
    from src.load_data import read_matches
    probs = walk_forward_probs(read_matches(), xi=0.0019, dixon_coles=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    probs.to_csv(OUT_FILE, index=False)
    print(f"saved {len(probs)} rows -> {OUT_FILE}; unseen-team matches: {int(probs.pois_missing.sum())}")


if __name__ == "__main__":
    main()
