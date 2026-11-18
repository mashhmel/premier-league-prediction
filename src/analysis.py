"""Explaining the results: bootstrap ranges, charts and tables for the report and slides.

Every plot_* function saves a PNG in results/figures/ (dpi 200) and returns the path.
Owners: Cheypol (bootstrap, SHAP, confusion, calibration), Chhoungkorng (draws, Poisson example)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import rps  # noqa: E402

FIG_DIR = Path("results/figures")
# reference categorical palette (fixed order) + neutral for benchmarks
COLORS = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a"}
GREY, INK, MUTED, GRID = "#8a8a85", "#1f1f1e", "#5f5f5a", "#e6e6e1"
LABELS = ["Home win", "Draw", "Away win"]
MODEL_NAMES = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bdbdb6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def load_predictions(path="results/test_predictions.npz"):
    d = np.load(path, allow_pickle=False)
    preds = {k: d[k] for k in d.files if k not in ("labels", "match_id")}
    return d["labels"], d["match_id"], preds


# ---- 1. Is a difference real or luck? (bootstrap) -------------------------------------
def per_match_rps(y, P):
    y, P = np.asarray(y, dtype=int), np.asarray(P, dtype=float)
    O = np.eye(3)[y]
    cp, co = np.cumsum(P, axis=1), np.cumsum(O, axis=1)
    return np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2


def bootstrap_diff(y, P_new, P_old, n=1000, seed=42):
    """RPS(new) - RPS(old) on resampled test sets. Negative = new is better.
    Returns mean difference, 95% range, and how often new was better."""
    rng = np.random.default_rng(seed)
    d = per_match_rps(y, P_new) - per_match_rps(y, P_old)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "p_better": round(float((boots < 0).mean()), 3),
            "clear": bool(hi < 0 or lo > 0)}


def research_question_table(y, preds, models=("LR", "RF", "XGB"), n=1000):
    """RQ1 (B vs A), RQ2 (C vs B), RQ3 (D vs C) for each model, with bootstrap ranges."""
    rows = []
    for m in models:
        for rq, new, old in [("RQ1 features", "B", "A"), ("RQ2 head-to-head", "C", "B"),
                             ("RQ3 hybrid", "D", "C")]:
            rows.append({"question": rq, "model": m, "compare": f"{new} vs {old}",
                         **bootstrap_diff(y, preds[f"{m}_{new}"], preds[f"{m}_{old}"], n=n)})
    return pd.DataFrame(rows)


# ---- 2. Setup comparison chart (answers RQ1-RQ3 in one picture) ------------------------
def plot_setup_comparison(table, title="Test seasons 2023/24-2024/25", metric="rps", fname="setup_comparison.png",
                          benchmarks=("Bookmaker (Bet365)", "Poisson (Dixon-Coles)")):
    """table: results_table.csv (model, setup, metric...). One line per model across setups A-D,
    benchmarks as grey reference lines. (Frequencies is far worse; it is left out to keep the scale readable.)"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    setups = ["A", "B", "C", "D"]
    x = np.arange(4)
    for m, c in COLORS.items():
        sub = table[table.model == m].set_index("setup").reindex(setups)
        if sub[metric].isna().all():
            continue
        ax.plot(x, sub[metric], color=c, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=MODEL_NAMES[m])
    styles = ["--", ":"]
    for i, b in enumerate(benchmarks):
        row = table[table.model == b]
        if len(row):
            v = float(row[metric].iloc[0])
            ax.axhline(v, color=GREY, linewidth=1.2, linestyle=styles[i % 2], label=f"{b}: {v:.4f}")
    ax.set_xticks(x, ["A\nbasic stats", "B\n+ engineered", "C\n+ head-to-head", "D\n+ hybrid"])
    ax.set_ylabel("RPS (lower is better)" if metric == "rps" else metric, color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    _style(ax)
    return _save(fig, fname)


# ---- 3. Confusion matrix ------------------------------------------------------------------
def plot_confusion(y, P, title, fname="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, np.asarray(P).argmax(axis=1), labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK)
    ax.set_xticks(range(3), LABELS, fontsize=9)
    ax.set_yticks(range(3), LABELS, fontsize=9)
    ax.set_xlabel("Predicted (top pick)", color=MUTED, fontsize=9)
    ax.set_ylabel("Actual result", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    return _save(fig, fname), cm


# ---- 4. Draw analysis ------------------------------------------------------------------------
def draw_analysis(y, P, elo_diff):
    """How often draws happen vs how often the model picks them, by Elo gap."""
    y, P, elo_diff = np.asarray(y), np.asarray(P), np.asarray(elo_diff)
    pick = P.argmax(axis=1)
    bins = [0, 50, 100, 200, 10_000]
    names = ["0-50", "50-100", "100-200", "200+"]
    gap = np.abs(elo_diff)
    rows = []
    for lo, hi, nm in zip(bins[:-1], bins[1:], names):
        mask = (gap >= lo) & (gap < hi)
        if mask.sum() == 0:
            continue
        rows.append({"elo_gap": nm, "matches": int(mask.sum()),
                     "actual_draw_rate": round(float((y[mask] == 1).mean()), 3),
                     "avg_predicted_draw": round(float(P[mask, 1].mean()), 3),
                     "picked_draw_share": round(float((pick[mask] == 1).mean()), 3)})
    summary = {"draw_rate": round(float((y == 1).mean()), 3),
               "draw_picked": round(float((pick == 1).mean()), 3),
               "max_draw_prob": round(float(P[:, 1].max()), 3)}
    return pd.DataFrame(rows), summary


def plot_draws(draw_table, fname="draw_analysis.png"):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(draw_table))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, draw_table.actual_draw_rate, w, color="#2a78d6", label="Actual draw rate")
    ax.bar(x + w / 2 + 0.01, draw_table.avg_predicted_draw, w, color="#eb6834", label="Model's average draw probability")
    ax.set_xticks(x, [f"{g}\n({n} matches)" for g, n in zip(draw_table.elo_gap, draw_table.matches)], fontsize=8)
    ax.set_xlabel("Elo gap between the teams", color=MUTED, fontsize=9)
    ax.set_ylabel("Share of matches", color=MUTED, fontsize=9)
    ax.set_title("Draws: how often they happen vs what the model expects", loc="left", fontsize=10, color=INK)
    top = max(draw_table.actual_draw_rate.max(), draw_table.avg_predicted_draw.max())
    ax.set_ylim(0, top * 1.35)                       # headroom so the legend never covers a bar
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    _style(ax)
    return _save(fig, fname)


# ---- 5. Calibration -----------------------------------------------------------------------
def calibration_table(y, P, outcome=0, bins=10):
    y, P = np.asarray(y), np.asarray(P)
    p = P[:, outcome]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.sum() >= 10:
            rows.append({"predicted": float(p[mask].mean()), "actual": float((y[mask] == outcome).mean()),
                         "n": int(mask.sum())})
    return pd.DataFrame(rows)


def plot_calibration(y, curves, outcome=0, fname="calibration_home.png"):
    """curves: {label: (P, color)}. Bins with fewer than 10 matches are skipped."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=1, linestyle="--", label="Perfect calibration")
    for label, (P, color) in curves.items():
        t = calibration_table(y, P, outcome)
        ax.plot(t.predicted, t.actual, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted {LABELS[outcome].lower()} probability", color=MUTED, fontsize=9)
    ax.set_ylabel(f"Actual {LABELS[outcome].lower()} rate", color=MUTED, fontsize=9)
    ax.set_title("Calibration on the test seasons", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    return _save(fig, fname)


# ---- 6. Feature importance -----------------------------------------------------------------
def lr_coefficients(model, cols):
    """Home-win vs away-win weight per feature from a fitted LR pipeline (standardised inputs)."""
    lr = model[-1]
    coefs = pd.DataFrame(lr.coef_.T, index=cols, columns=["home", "draw", "away"])
    coefs["home_minus_away"] = coefs["home"] - coefs["away"]
    coefs["importance"] = coefs[["home", "draw", "away"]].abs().max(axis=1)
    return coefs.sort_values("importance", ascending=False)


def plot_lr_coefficients(coefs, top=12, fname="lr_coefficients.png"):
    c = coefs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(c) + 1.2))
    colors = ["#2a78d6" if v > 0 else "#eb6834" for v in c["home_minus_away"]]
    ax.barh(c.index, c["home_minus_away"], color=colors, height=0.6)
    ax.axvline(0, color="#bdbdb6", linewidth=1)
    ax.set_xlabel("Weight toward home win (+) or away win (−), standardised", color=MUTED, fontsize=9)
    ax.set_title("What Logistic Regression relies on", loc="left", fontsize=10, color=INK)
    ax.tick_params(axis="y", labelsize=8)
    _style(ax)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID)
    return _save(fig, fname)


def shap_importance(xgb_model, X, fname="shap_xgb.png", class_index=0, top=12):
    """Mean |SHAP| per feature for one class (0 = home win). Saves a bar chart + returns the table."""
    import shap
    sv = shap.TreeExplainer(xgb_model).shap_values(X)
    vals = sv[class_index] if isinstance(sv, list) else (sv[:, :, class_index] if np.ndim(sv) == 3 else sv)
    imp = pd.Series(np.abs(vals).mean(axis=0), index=X.columns).sort_values(ascending=False)
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(t) + 1.2))
    ax.barh(t.index, t.values, color="#1baf7a", height=0.6)
    ax.set_xlabel(f"Mean |SHAP value| ({LABELS[class_index].lower()})", color=MUTED, fontsize=9)
    ax.set_title("What XGBoost relies on (SHAP)", loc="left", fontsize=10, color=INK)
    ax.tick_params(axis="y", labelsize=8)
    _style(ax)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID)
    path = _save(fig, fname)
    rank = {f: i + 1 for i, f in enumerate(imp.index)}
    return imp, path, rank


# ---- 7. Extra: home advantage per season (2020/21 had no crowds) -------------------------------
def home_advantage_by_season(m):
    t = m.groupby("season")["result"].value_counts(normalize=True).unstack().fillna(0)
    return t[["H", "D", "A"]].round(3)


def plot_home_advantage(m, fname="home_advantage.png"):
    t = home_advantage_by_season(m)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(t))
    ax.plot(x, t["H"], color="#2a78d6", linewidth=2, marker="o", markersize=5, label="Home win rate")
    ax.plot(x, t["A"], color="#eb6834", linewidth=2, marker="o", markersize=5, label="Away win rate")
    top = max(t["H"].max(), t["A"].max())
    if "2021" in t.index:
        i = list(t.index).index("2021")
        ax.axvspan(i - 0.4, i + 0.4, color="#ededE8", zorder=0)      # shaded band behind the lines
        ax.text(i, top + 0.035, "2020/21\nno crowds", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_ylim(min(t["H"].min(), t["A"].min()) - 0.03, top + 0.10)
    ax.set_xticks(x, [f"{s[:2]}/{s[2:]}" for s in t.index], fontsize=7, rotation=45)
    ax.set_ylabel("Share of matches", color=MUTED, fontsize=9)
    ax.set_title("Home advantage by season", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _style(ax)
    return _save(fig, fname), t


# ---- 8. Dixon-Coles scoreline example ----------------------------------------------------------
def plot_scoreline_grid(grid, home, away, fname="scoreline_example.png", n=6):
    g = grid[:n, :n] * 100
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(g, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if g[i, j] > g.max() * 0.6 else INK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel(f"{away} goals", color=MUTED, fontsize=9)
    ax.set_ylabel(f"{home} goals", color=MUTED, fontsize=9)
    ax.set_title(f"{home} vs {away}: scoreline chances (%)", loc="left", fontsize=10, color=INK)
    return _save(fig, fname)
