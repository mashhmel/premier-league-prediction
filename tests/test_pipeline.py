"""Run from the repo root:  python -m pytest -q

These tests use FAKE data (tests/fake_data.py) so they run offline.
The leakage tests are the important ones: if they fail, a feature is peeking at the future.
"""
import numpy as np
import pytest

from src.evaluate import brier, evaluate, rps, split
from src.features import SETUPS, build_features
from src.poisson import dc_tau, outcome_probs_from_grid, scoreline_grid, walk_forward_probs
from tests.fake_data import make_fake_matches


@pytest.fixture(scope="module")
def matches():
    return make_fake_matches()


@pytest.fixture(scope="module")
def features(matches):
    return build_features(matches)


# ---- metrics -----------------------------------------------------------------------------
def test_rps_perfect_is_zero():
    assert rps([0, 1, 2], np.eye(3)) == 0


def test_rps_values():
    # home win happened; predicting a draw is "less wrong" than predicting an away win
    assert rps([0], [[0, 1, 0]]) < rps([0], [[0, 0, 1]])
    assert rps([0], [[0, 0, 1]]) == pytest.approx(1.0)


def test_brier_range():
    assert brier([0], [[1, 0, 0]]) == 0
    assert brier([0], [[0, 0, 1]]) == pytest.approx(2.0)


# ---- data shape ------------------------------------------------------------------------
def test_shape(matches, features):
    assert len(features) == len(matches) == 20 * 380
    assert features["match_id"].is_unique


def test_no_missing_in_setup_c(features):
    assert features[SETUPS["C"]].isna().sum().sum() == 0


def test_splits_are_by_season(features):
    train, val, test = split(features)
    assert train.date.max() < val.date.min() < test.date.min()
    assert len(test) == 760 and len(val) == 380


# ---- LEAKAGE -----------------------------------------------------------------------------
def _flip(result):
    return {"H": "A", "A": "H", "D": "H"}[result]


def test_no_leakage_from_own_result(matches, features):
    """Changing a match's own result/score must NOT change that match's features."""
    target = len(matches) // 2
    m2 = matches.copy()
    m2.loc[target, ["hg", "ag"]] = m2.loc[target, ["ag", "hg"]].values + [3, 0]
    m2.loc[target, "result"] = _flip(m2.loc[target, "result"])
    m2.loc[target, ["hst", "ast"]] = [20, 0]
    f2 = build_features(m2)
    cols = SETUPS["C"]
    row1 = features.loc[features.match_id == target, cols].to_numpy()
    row2 = f2.loc[f2.match_id == target, cols].to_numpy()
    assert np.allclose(row1, row2, equal_nan=True)


def test_no_leakage_from_future(matches, features):
    """Changing every result in the LAST season must not change any earlier match's features."""
    m2 = matches.copy()
    last = m2.season == "2425"
    m2.loc[last, ["hg", "ag"]] = m2.loc[last, ["ag", "hg"]].values
    m2.loc[last, "result"] = m2.loc[last, "result"].map(_flip)
    f2 = build_features(m2)
    early = features.season != "2425"
    cols = SETUPS["C"]
    a = features.loc[early].sort_values("match_id")[cols].to_numpy()
    b = f2.loc[f2.season != "2425"].sort_values("match_id")[cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_poisson_walk_forward_no_leakage(matches):
    """Poisson probs for a month must not change if a LATER month's results change."""
    sub = matches[matches.season.isin(["2223", "2324"])].copy()
    p1 = walk_forward_probs(sub, min_train=100)
    sub2 = sub.copy()
    late = sub2.date >= sub2.date.max() - np.timedelta64(30, "D")
    sub2.loc[late, ["hg", "ag"]] = sub2.loc[late, ["ag", "hg"]].values
    p2 = walk_forward_probs(sub2, min_train=100)
    early_ids = sub.loc[sub.date < sub.date.max() - np.timedelta64(60, "D"), "match_id"]
    a = p1[p1.match_id.isin(early_ids)].sort_values("match_id")["pois_pH"].to_numpy()
    b = p2[p2.match_id.isin(early_ids)].sort_values("match_id")["pois_pH"].to_numpy()
    assert len(a) > 0 and np.allclose(a, b, equal_nan=True)


# ---- Poisson ---------------------------------------------------------------------------
def test_poisson_probs_sum_to_one():
    ph, pd_, pa = outcome_probs_from_grid(scoreline_grid(1.6, 1.1))
    assert ph + pd_ + pa == pytest.approx(1.0)
    assert ph > pa                                        # stronger expected attack wins more


# ---- models run end to end -------------------------------------------------------------
def test_models_run(features):
    from src.models import fit_predict
    train, val, _ = split(features)
    for name in ("LR", "RF"):
        _, P = fit_predict(name, train, val, SETUPS["B"])
        assert P.shape == (len(val), 3)
        assert np.allclose(P.sum(axis=1), 1)
        assert evaluate(val.label, P)["accuracy"] > 0.33   # better than random on fake data


# ---- Dixon-Coles -------------------------------------------------------------------------
def test_dixon_coles_grid():
    plain = scoreline_grid(1.4, 1.1)
    dc = scoreline_grid(1.4, 1.1, rho=-0.1)
    assert dc.sum() == pytest.approx(1.0)
    assert dc[0, 0] > plain[0, 0] and dc[1, 1] > plain[1, 1]     # negative rho: more 0-0, 1-1
    assert dc[1, 0] < plain[1, 0] and dc[0, 1] < plain[0, 1]
    assert np.allclose(dc_tau(2, 3, 1.4, 1.1, -0.1), 1.0)         # high scores untouched


def test_dixon_coles_walk_forward(matches):
    sub = matches[matches.season.isin(["2122", "2223", "2324"])]
    p = walk_forward_probs(sub, xi=0.0019, dixon_coles=True, min_train=200)
    ok = p.dropna(subset=["pois_pH"])
    assert len(ok) > 0
    assert np.allclose(ok[["pois_pH", "pois_pD", "pois_pA"]].sum(axis=1), 1)
    assert ok["pois_rho"].between(-0.2, 0.2).all()


# ---- final-test guard, bootstrap, fixture prediction ---------------------------------------
def test_bootstrap_identical_is_zero(features):
    from src.analysis import bootstrap_diff
    y = features.label.values[:300]
    P = np.tile([0.45, 0.25, 0.30], (300, 1))
    r = bootstrap_diff(y, P, P, n=200)
    assert r["diff"] == 0 and not r["clear"]


def test_final_test_refuses_second_run(features, tmp_path):
    import pandas as pd
    from src.models import run_final_test
    from src.poisson import fill_missing_probs
    f = features.copy()
    f["pois_pH"], f["pois_pD"], f["pois_pA"] = 0.45, 0.25, 0.30
    tuned = pd.DataFrame([{"model": "LR", "setup": "A", "params": {"C": 0.1}}])
    run_final_test(f, tuned, SETUPS, out_dir=tmp_path)
    with pytest.raises(RuntimeError):
        run_final_test(f, tuned, SETUPS, out_dir=tmp_path)


def test_fixture_features_ignore_dummy_result(matches):
    """The made-up result of a future fixture must not change its own features."""
    import pandas as pd
    from src.features import build_features
    home, away = matches.home.iloc[-1], matches.away.iloc[-1]
    date = matches.date.max() + pd.Timedelta(days=7)
    rows = []
    for res, hg, ag in [("D", 0, 0), ("H", 5, 0)]:
        d = {c: np.nan for c in matches.columns}
        d.update({"match_id": 99999, "season": matches.season.max(), "date": date, "home": home,
                  "away": away, "hg": hg, "ag": ag, "result": res, "label": 1, "hst": 0, "ast": 0})
        f = build_features(pd.concat([matches, pd.DataFrame([d])], ignore_index=True))
        rows.append(f.loc[f.match_id == 99999, SETUPS["C"]].to_numpy())
    assert np.allclose(rows[0], rows[1], equal_nan=True)
