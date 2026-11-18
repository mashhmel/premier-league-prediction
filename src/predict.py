"""Predict a NEW fixture (any two teams) with the trained app model.

Trick: append the fixture as a dummy match after the last real match and rebuild features.
Every feature only uses earlier matches (the leakage tests prove it), so the dummy result
never affects the fixture's own features.

After the final test, train the app model on ALL seasons (2010/11-2024/25):
    python -m src.predict train
Owner: Chhoungkorng
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import SETUPS, build_features
from src.load_data import read_matches
from src.poisson import (estimate_rho, expected_goals, fit_poisson, outcome_probs_from_grid,
                         scoreline_grid, top_scorelines)

APP_MODEL = Path("results/models/app_model.joblib")
APP_META = Path("results/models/app_model.json")
XI = 0.0019


def poisson_for_fixture(m, home, away, date):
    """Dixon-Coles probabilities + scoreline grid for one fixture, fitted on the 2 years before `date`."""
    train = m[(m.date < date) & (m.date >= date - pd.Timedelta(days=730))]
    model = fit_poisson(train, xi=XI, ref_date=date)
    rho = estimate_rho(train, model, xi=XI, ref_date=date)
    lh, la = expected_goals(model, [home], [away])
    grid = scoreline_grid(lh[0], la[0], rho=rho)
    return outcome_probs_from_grid(grid), grid, (float(lh[0]), float(la[0])), rho


def fixture_features(m, home, away, date=None):
    """One feature row (all setups' columns) for home vs away on `date` (default: 7 days after the data ends)."""
    date = pd.Timestamp(date) if date is not None else m.date.max() + pd.Timedelta(days=7)
    dummy = {c: np.nan for c in m.columns}
    dummy.update({"match_id": int(m.match_id.max()) + 1, "season": m.season.max(), "date": date,
                  "home": home, "away": away, "hg": 0, "ag": 0, "result": "D", "label": 1,
                  "hst": 0, "ast": 0})
    f = build_features(pd.concat([m, pd.DataFrame([dummy])], ignore_index=True))
    row = f[f.match_id == dummy["match_id"]].copy()
    (pH, pD, pA), grid, xg, rho = poisson_for_fixture(m, home, away, date)
    row["pois_pH"], row["pois_pD"], row["pois_pA"] = pH, pD, pA
    return row, grid, xg, rho


def train_app_model(model_name="LR", setup="D", params=None, path_probs="data/processed/poisson_probs.csv"):
    """Train the chosen model on every season 2010/11-2024/25 and save it for the app."""
    from src.models import make_model
    from src.poisson import fill_missing_probs
    m = read_matches()
    f = build_features(m).merge(fill_missing_probs(pd.read_csv(path_probs)), on="match_id", how="left")
    f = f[f.season >= "1011"]
    cols = SETUPS[setup]
    model = make_model(model_name, **(params or {})).fit(f[cols], f["label"])
    APP_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, APP_MODEL)
    APP_META.write_text(json.dumps({"model": model_name, "setup": setup, "params": params or {},
                                    "trained_on": "2010/11-2024/25", "columns": cols}, indent=2))
    print(f"saved {model_name} setup {setup} -> {APP_MODEL}")
    return model


def predict_fixture(home, away, m=None):
    m = m if m is not None else read_matches()
    row, grid, xg, rho = fixture_features(m, home, away)
    meta = json.loads(APP_META.read_text()) if APP_META.exists() else None
    if meta and APP_MODEL.exists():
        model = joblib.load(APP_MODEL)
        probs = model.predict_proba(row[meta["columns"]])[0]
        source = f"{meta['model']} (setup {meta['setup']})"
    else:
        probs = outcome_probs_from_grid(grid)
        source = "Poisson (Dixon-Coles) - app model not trained yet"
    return {"probs": [float(p) for p in probs], "source": source, "xg": xg, "rho": rho,
            "top_scores": top_scorelines(grid), "grid": grid,
            "elo": (float(row.elo_home.iloc[0]), float(row.elo_away.iloc[0])),
            "form": (float(row.home_form_pts5.iloc[0]), float(row.away_form_pts5.iloc[0]))}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # default = best model from tuning; edit if your final results pick a different one
        params = None
        tp = Path("results/tuning_results.csv")
        if tp.exists():
            from src.models import load_tuned
            t = load_tuned(tp)
            row = t[(t.model == "LR") & (t.setup == "D")]
            params = row.params.iloc[0] if len(row) else None
        train_app_model("LR", "D", params)
