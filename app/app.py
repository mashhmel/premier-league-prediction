"""Streamlit prediction app.

Run from the repo root:
    python -m src.predict train     # once, after the final test (saves the app model)
    streamlit run app/app.py

Win/draw/loss chances come from the best model (Logistic Regression, setup D) trained on all
seasons; scorelines come from the Dixon-Coles model. Without a trained app model it falls
back to Dixon-Coles for everything.
Owner: Chhoungkorng
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.load_data import read_matches   # noqa: E402
from src.predict import predict_fixture  # noqa: E402


@st.cache_data
def load():
    m = read_matches()
    teams = sorted(set(m.loc[m.season == m.season.max(), "home"]))
    return m, teams


st.title("Premier League match predictor")
m, teams = load()
season = m.season.max()
st.caption(f"Teams from {season[:2]}/{season[2:]}. Prediction for the next match after "
           f"{m.date.max():%d %b %Y}, using only information known before kickoff.")

c1, c2 = st.columns(2)
home = c1.selectbox("Home team", teams)
away = c2.selectbox("Away team", [t for t in teams if t != home])

if st.button("Predict", type="primary"):
    with st.spinner("Building features and fitting the goals model..."):
        r = predict_fixture(home, away, m)
    pH, pD, pA = r["probs"]
    st.subheader("Result probabilities")
    a, b, c = st.columns(3)
    a.metric(f"{home} win", f"{pH:.0%}")
    b.metric("Draw", f"{pD:.0%}")
    c.metric(f"{away} win", f"{pA:.0%}")
    st.bar_chart(pd.DataFrame({"probability": [pH, pD, pA]}, index=["Home win", "Draw", "Away win"]))
    st.caption(f"Model: {r['source']}")

    st.subheader("Most likely scorelines (Dixon-Coles)")
    top = pd.DataFrame(r["top_scores"], columns=["probability", "score"])
    top["probability"] = (top["probability"] * 100).round(1).astype(str) + "%"
    st.table(top[["score", "probability"]])

    with st.expander("Why? The clues behind this prediction"):
        st.write(f"Expected goals: **{home} {r['xg'][0]:.2f} – {r['xg'][1]:.2f} {away}**")
        st.write(f"Elo rating: {home} {r['elo'][0]:.0f}, {away} {r['elo'][1]:.0f}")
        st.write(f"Form (avg points, last 5): {home} {r['form'][0]:.2f}, {away} {r['form'][1]:.2f}")
        st.write(f"Dixon-Coles rho: {r['rho']:.3f}")

st.caption("For learning and fan use only. Not betting advice.")
