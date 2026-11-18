"""ML models (Logistic Regression, Random Forest, XGBoost) for any setup.

A setup is just a list of column names (see src/features.py SETUPS).
Owner: Cheypol
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate, split

# small grids for week 3 tuning (choose by validation RPS)
GRIDS = {
    "LR": [{"C": c} for c in (0.01, 0.1, 1, 10)],
    "RF": [{"max_depth": d, "min_samples_leaf": s} for d in (4, 6, 8, None) for s in (5, 20, 50)],
    "XGB": [{"max_depth": d, "learning_rate": lr, "n_estimators": n, "subsample": 0.8}
            for d in (2, 3, 4) for lr in (0.03, 0.1) for n in (200, 400)],
}


def make_model(name, **params):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, **params))
    if name == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, **params))
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                             random_state=42, n_jobs=-1, **params)
    raise ValueError(name)


def fit_predict(name, train, test, cols, **params):
    model = make_model(name, **params).fit(train[cols], train["label"])
    return model, model.predict_proba(test[cols])          # columns: H, D, A


def run_validation(df, setups, models=("LR", "RF", "XGB")):
    """Default settings, train -> validation. One row per (model, setup)."""
    train, val, _ = split(df)
    rows = []
    for s, cols in setups.items():
        for name in models:
            _, P = fit_predict(name, train, val, cols)
            rows.append({"model": name, "setup": s, "split": "val", **evaluate(val.label, P)})
    return pd.DataFrame(rows)


def tune(df, name, cols, grid=None):
    """Try each setting on validation; return (best_params, table sorted by RPS)."""
    train, val, _ = split(df)
    rows = []
    for params in grid or GRIDS[name]:
        _, P = fit_predict(name, train, val, cols, **params)
        rows.append({"params": params, **evaluate(val.label, P)})
    table = pd.DataFrame(rows).sort_values("rps")
    return table.iloc[0]["params"], table


def final_test(df, name, cols, **params):
    """RUN ONCE (end of week 3): retrain on train + val, score on test."""
    train, val, test = split(df)
    model, P = fit_predict(name, pd.concat([train, val]), test, cols, **params)
    return model, P, evaluate(test.label, P)


# ---- 5-season rolling validation (more reliable than one season) -------------------------
ROLLING_SEASONS = ["1819", "1920", "2021", "2122", "2223"]


def rolling_eval(df, name, cols, params=None, seasons=ROLLING_SEASONS):
    """For each season: train on every earlier training season (from 2010/11), predict it.
    Returns metrics over all predicted matches (~1,900). Test seasons are never touched."""
    import numpy as np
    Ps, ys = [], []
    for vs in seasons:
        tr = df[(df.season >= "1011") & (df.season < vs)]
        te = df[df.season == vs]
        _, P = fit_predict(name, tr, te, cols, **(params or {}))
        Ps.append(P)
        ys.append(te["label"].to_numpy())
    return evaluate(np.concatenate(ys), np.vstack(Ps))


def rolling_validation(df, setups, models=("LR", "RF", "XGB")):
    """Default settings for every model x setup, scored on the 5 rolling seasons."""
    rows = [{"model": n, "setup": s, **rolling_eval(df, n, cols)}
            for s, cols in setups.items() for n in models]
    return pd.DataFrame(rows)


EXTRA_GRIDS = {   # used when the best setting sat at the edge of GRIDS
    "LR": [{"C": c} for c in (0.001, 0.003)],
    "XGB": [{"max_depth": d, "learning_rate": 0.01, "n_estimators": n, "subsample": 0.8}
            for d in (1, 2) for n in (300, 600)],
}


def tune_rolling(df, setups, models=("LR", "RF", "XGB"), grids=None, extra=True, verbose=True):
    """Pick the best settings for every model x setup by rolling-validation RPS.
    Returns one row per model x setup: model, setup, params, metrics."""
    grids = grids or GRIDS
    best = []
    for s, cols in setups.items():
        for name in models:
            grid = list(grids[name]) + (EXTRA_GRIDS.get(name, []) if extra else [])
            trials = [{"params": p, **rolling_eval(df, name, cols, p)} for p in grid]
            top = min(trials, key=lambda r: r["rps"])
            best.append({"model": name, "setup": s, **top})
            if verbose:
                print(name, s, top["params"], top["rps"])
    return pd.DataFrame(best)


def load_tuned(path="results/tuning_results.csv"):
    """Reload saved tuning results (params come back from text to dicts)."""
    import ast
    t = pd.read_csv(path)
    t["params"] = t["params"].apply(lambda p: p if isinstance(p, dict) else ast.literal_eval(p))
    return t


# ---- FINAL TEST (run once) -------------------------------------------------------------------
def run_final_test(df, tuned, setups, out_dir="results", force=False):
    """Train every tuned model on 2010/11-2022/23, score ONCE on 2023/24-2024/25.

    Refuses to run again if results/results_table.csv already exists (protects the
    "test once" rule). Re-running with identical settings gives identical numbers, so
    force=True is only for reproducing after a crash/restart - never for trying new settings.
    Saves: results_table.csv, test_predictions.npz, models/<model>_<setup>.joblib
    """
    from pathlib import Path
    import joblib
    import numpy as np
    from src.evaluate import always_home, bookmaker_probs, frequencies
    out = Path(out_dir)
    table_path = out / "results_table.csv"
    if table_path.exists() and not force:
        raise RuntimeError(f"{table_path} already exists - the final test has been run. "
                           "Use force=True ONLY to reproduce it with the same settings.")
    (out / "models").mkdir(parents=True, exist_ok=True)
    train, val, test = split(df)
    rows, preds = [], {}
    for r in tuned.to_dict("records"):
        model, P, scores = final_test(df, r["model"], setups[r["setup"]], **r["params"])
        rows.append({"model": r["model"], "setup": r["setup"], **scores, "params": str(r["params"])})
        preds[f"{r['model']}_{r['setup']}"] = P
        joblib.dump(model, out / "models" / f"{r['model']}_{r['setup']}.joblib")
    P_pois = test[["pois_pH", "pois_pD", "pois_pA"]].to_numpy()
    P_book = bookmaker_probs(test)
    P_freq = frequencies(pd.concat([train, val])["label"], len(test))
    for name, P in [("Poisson (Dixon-Coles)", P_pois), ("Bookmaker (Bet365)", P_book),
                    ("Frequencies", P_freq), ("Always home", always_home(len(test)))]:
        rows.append({"model": name, "setup": "-", **evaluate(test["label"], P), "params": ""})
        preds[name] = P
    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)
    np.savez(out / "test_predictions.npz", labels=test["label"].to_numpy(),
             match_id=test["match_id"].to_numpy(), **preds)
    return table


def extend_tuning(df, tuned, setups, extra=None, verbose=True):
    """Try EXTRA_GRIDS on top of already-saved tuning; keep whichever is better.
    Safe to run more than once (only ever replaces a setting with a better one)."""
    extra = extra or EXTRA_GRIDS
    rows = []
    for r in tuned.to_dict("records"):
        for p in extra.get(r["model"], []):
            if p == r["params"]:
                continue
            sc = rolling_eval(df, r["model"], setups[r["setup"]], p)
            if sc["rps"] < r["rps"]:
                if verbose:
                    print(f"{r['model']} {r['setup']}: {r['params']} -> {p}  rps {r['rps']} -> {sc['rps']}")
                r = {**r, "params": p, **sc}
        rows.append(r)
    return pd.DataFrame(rows)
