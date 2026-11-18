"""Smoothed head-to-head score.

From the HOME team's point of view, over the last `last_n` league meetings (either venue):
    h2h_score = (wins + 0.5 * draws + k * 0.5) / (n + k)
k "imaginary" neutral meetings pull small samples toward 0.5.
    never met               -> 0.50
    met once, home team won -> (1 + 1.5) / 4 = 0.625   (not 1.0)

Owner: Darareach
"""
from collections import defaultdict, deque


def add_h2h(m, last_n=6, k=3):
    m = m.sort_values(["date", "home"]).copy()          # MUST be in date order
    hist = defaultdict(lambda: deque(maxlen=last_n))     # pair -> recent winners ("D" = draw)
    scores, counts = [], []
    for row in m.itertuples():
        past = hist[tuple(sorted((row.home, row.away)))]
        w = sum(1 for r in past if r == row.home)
        d = sum(1 for r in past if r == "D")
        n = len(past)
        scores.append((w + 0.5 * d + k * 0.5) / (n + k))   # 1) record BEFORE updating
        counts.append(n)
        winner = row.home if row.result == "H" else row.away if row.result == "A" else "D"
        past.append(winner)                                # 2) then update
    m["h2h_score"], m["h2h_n"] = scores, counts
    return m
