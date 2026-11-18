# Is Head-to-Head History Overrated?

**Predicting Premier League match results with engineered features and a hybrid Poisson–machine learning model.**
ITM-390 Machine Learning group project · Advisor: Kuntha Pin

| Member | Main role | Owns |
|---|---|---|
| Lyzing Seang (leader) | Data, rest days + shots features, report lead | `src/load_data.py` |
| Darareach Nhoeun | Elo, form, home/away form, head-to-head | `src/elo.py`, `src/h2h.py`, `src/features.py` |
| Cheypol Khin | Evaluation, ML models, setups A–D, tuning, SHAP | `src/evaluate.py`, `src/models.py`, `src/analysis.py` |
| Chhoungkorng Tang | Poisson / Dixon–Coles, hybrid, Streamlit app | `src/poisson.py`, `src/predict.py`, `app/app.py` |

---

## Results (final test: 2023/24 + 2024/25, 760 matches, run once)

| Model | A: basic stats | B: + engineered | C: + head-to-head | D: + hybrid |
|---|---|---|---|---|
| Logistic Regression | 0.2048 | **0.1951** | 0.1953 | **0.1951** |
| Random Forest | 0.2049 | 0.1976 | 0.1980 | 0.1972 |
| XGBoost | 0.2047 | 0.1974 | 0.1974 | 0.1969 |

RPS (Ranked Probability Score), lower is better.

| Benchmark | Accuracy | RPS |
|---|---|---|
| Bookmaker (Bet365) | 56.8% | 0.1906 |
| **Best model: Logistic Regression (B / D)** | **56.5%** | **0.1951** |
| Poisson (Dixon–Coles) | 54.7% | 0.1979 |
| Historical frequencies | 43.4% | 0.2348 |
| Always home win | 43.4% | 0.4507 |

**Answers to the research questions** (bootstrap with 1,000 resamples of the test matches):

- **RQ1: does feature engineering help? Yes, clearly.** A → B improves every model (RPS −0.007 to −0.010), and in 100% of bootstrap resamples.
- **RQ2: is head-to-head history overrated? Yes.** B → C gives no improvement for any model (LR +0.0002, RF +0.0003, XGB 0.0000). In XGBoost's SHAP ranking, `h2h_score` is 31st and `h2h_n` is 32nd out of 32 features.
- **RQ3: does the Poisson + ML hybrid help? Only a little, and not clearly.** C → D improves RPS by 0.0002–0.0008, better in 65–83% of resamples, but the 95% range crosses zero. Every hybrid beats Poisson alone.
- **RQ4: how close to the bookmakers? Very close.** Best model 56.5% accuracy vs 56.8%, RPS 0.1951 vs 0.1906, using only past results and shots.

Other findings:

- After tuning, all three models are nearly equal (the features matter more than the algorithm), and XGBoost preferred the simplest possible trees (depth 1).
- The model almost never picks "draw" as its top result (1 of 760 matches), even though 23% of matches were draws. Draws are never the single most likely outcome.
- Dixon–Coles rho = −0.044 (more 0–0 and 1–1 than plain Poisson predicts).
- In 2020/21 (no crowds), away wins (40.3%) outnumbered home wins (37.9%), the only season where this happened.

Full tables: `results/results_table.csv`, `results/rq_bootstrap.csv`, `results/tuning_results.csv`. Charts: `results/figures/`.

---

## Quick start (Windows)

```
git clone https://github.com/mashhmel/FootballPrediction.git
cd FootballPrediction
python -m pip install -r requirements.txt
python -m pytest -q                        # should say 16 passed
```

**Clone outside OneDrive** (e.g. `C:\Users\<you>\Projects`). OneDrive syncing can corrupt Git files.
**Always run commands from the repo root** (the folder with this README), otherwise `from src...` imports fail.

### Run the app
```
python -m streamlit run app/app.py
```
It opens in your browser (`http://localhost:8501`). Pick a home and away team, then click **Predict**. You get win/draw/loss chances from the best model (Logistic Regression, setup D, trained on all seasons), the most likely scorelines from Dixon–Coles, and the clues behind the prediction (Elo, form, expected goals). Press `Ctrl + C` in the terminal to stop it.

If `results/models/app_model.joblib` is missing, train it first: `python -m src.predict train`.

### Reproduce everything
The data is already in `data/`. To re-download it: `python -m src.load_data`.

| Notebook | What it does |
|---|---|
| `notebooks/01_week1_quickstart.ipynb` | Data checks, benchmarks, first Poisson + Setup A results |
| `notebooks/02_final_experiment.ipynb` | Dixon–Coles, 5-season tuning, **final test**, bootstrap, all charts, app model |

`02_final_experiment` loads saved tuning results, so it only takes a few minutes. The final test refuses to run a second time (the "test once" rule). `run_final_test(..., force=True)` re-runs it with the same frozen settings and gives identical numbers (fixed random seed 42).

---

## Method in short

**Data:** football-data.co.uk, Premier League (`E0`), 20 seasons 2005/06–2024/25, 7,600 matches. 380 matches and 20 teams in every season, 0 missing values.

**Splits (by season, never random):**

| Split | Seasons | Use |
|---|---|---|
| Warm-up | 2005/06 – 2009/10 | Build Elo/form history only |
| Train | 2010/11 – 2022/23 | Fit models |
| Validation (rolling) | 2018/19 – 2022/23 | Each season predicted from all earlier ones; used for tuning |
| Test | 2023/24 – 2024/25 | **Run once** with frozen settings |

**Feature setups:**

| Setup | Adds | Research question |
|---|---|---|
| A | Season-to-date points / goals per game | Baseline |
| B | Elo, form (avg points per game, last 5), home/away form, rest days, shots-on-target ratio | RQ1 |
| C | Smoothed head-to-head: (wins + 0.5 × draws + 1.5) / (meetings + 3), last 6 meetings | RQ2 |
| D | Walk-forward Dixon–Coles win/draw/loss probabilities (hybrid) | RQ3 |

**Models:** Poisson / Dixon–Coles (time decay ξ = 0.0019, two-step rho fit), Logistic Regression, Random Forest, XGBoost. Tuned on rolling validation by RPS. The frozen settings are in `results/tuning_results.csv`.

**Metrics:** accuracy, macro F1, Brier score, RPS (main metric).

## Leakage rules

1. Sort by date before any rolling or loop feature.
2. Every rolling feature uses `shift(1)`. Elo and head-to-head record the value **before** updating.
3. Never fill missing values with an average over the whole dataset (it includes future matches). Use fixed values chosen in advance.
4. Poisson probabilities used as features come from `walk_forward_probs` only (fitted on earlier months).
5. `tests/test_pipeline.py` changes future results and checks that past features don't move (16 tests). Three matches were also checked by hand.

## Repo layout

```
data/raw/            20 downloaded season files (never edit by hand)
data/processed/      matches.csv, features.csv, poisson_probs.csv
src/                 load_data, features, elo, h2h, poisson, evaluate, models, analysis, predict
notebooks/           01_week1_quickstart, 02_final_experiment
app/app.py           Streamlit app
results/             results_table.csv, rq_bootstrap.csv, tuning_results.csv, figures/, models/
tests/               fake-data tests incl. leakage tests
report/, slides/     final write-up
```

## Team workflow

- `git pull` before you start, `git push` when you stop. **Never `git push --force`.**
- Shared code goes in `src/`. Experiments go in your own notebook.
- Run `python -m pytest -q` after changing any code in `src/`.

## Status

- [x] Data loader, features, Elo, head-to-head, leakage tests
- [x] Real data downloaded + sanity checks + 3 hand-checked matches
- [x] Poisson → time decay → Dixon–Coles
- [x] Setups A–D, 5-season rolling validation, tuning (with grid extension)
- [x] Final test (run once), bootstrap, 8 charts
- [x] App with the best model
- [ ] Report
- [ ] Presentation

*Starter code was drafted with AI assistance (Claude). Mention this in the course's AI-use disclosure if one is required.*
