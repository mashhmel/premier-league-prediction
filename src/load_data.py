"""Download, merge and clean Premier League match data from football-data.co.uk.

Run from the repo root:
    python -m src.load_data

Output: data/processed/matches.csv  (one row per match, sorted by date)

Owner: Lyzing
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/matches.csv")
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

FIRST_SEASON = 2005   # 2005/06 (warm-up seasons 2005/06 - 2009/10 are never trained on)
LAST_SEASON = 2024    # 2024/25

# columns we keep from the raw files -> our agreed names (see README)
RENAME = {
    "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "hg", "FTAG": "ag", "FTR": "result",
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "B365H": "odds_h", "B365D": "odds_d", "B365A": "odds_a",
}


def season_codes(first=FIRST_SEASON, last=LAST_SEASON):
    """2005 -> '0506', 2010 -> '1011', 2024 -> '2425'."""
    return [f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(first, last + 1)]


def download_raw(codes=None, force=False):
    """Download each season once and keep a raw copy in data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for code in codes or season_codes():
        path = RAW_DIR / f"E0_{code}.csv"
        if path.exists() and not force:
            continue
        df = pd.read_csv(BASE_URL.format(code=code), encoding="latin1", on_bad_lines="skip")
        df.to_csv(path, index=False)
        print(f"downloaded {code}: {len(df)} rows")


def load_raw(codes=None):
    frames = []
    for code in codes or season_codes():
        df = pd.read_csv(RAW_DIR / f"E0_{code}.csv", encoding="latin1", on_bad_lines="skip")
        df = df.reindex(columns=list(RENAME))            # missing columns become NaN
        df["season"] = code
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(raw):
    m = raw.rename(columns=RENAME)
    m = m.dropna(subset=["home", "away", "result"])      # drop empty trailing rows
    # older files use dd/mm/yy, newer use dd/mm/yyyy
    m["date"] = pd.to_datetime(m["date"], dayfirst=True, format="mixed")
    m["home"] = m["home"].str.strip()
    m["away"] = m["away"].str.strip()
    m[["hg", "ag"]] = m[["hg", "ag"]].astype(int)
    m["label"] = m["result"].map({"H": 0, "D": 1, "A": 2})   # ORDER MATTERS (RPS)
    m = m.sort_values(["date", "home"]).reset_index(drop=True)
    m["match_id"] = range(len(m))
    cols = ["match_id", "season", "date", "home", "away", "hg", "ag", "result", "label",
            "hs", "as_", "hst", "ast", "odds_h", "odds_d", "odds_a"]
    return m[cols]


def sanity_check(m):
    """Print the checks from the gameplan. Look at the output, don't just run it."""
    per_season = m.groupby("season").size()
    bad = per_season[per_season != 380]
    print("matches per season (expect 380):", "all OK" if bad.empty else bad.to_dict())
    teams = m.groupby("season")["home"].nunique()
    print("teams per season (expect 20):", "all OK" if (teams == 20).all() else teams[teams != 20].to_dict())
    print("missing results:", int(m["label"].isna().sum()))
    print("missing shots-on-target:", int(m[["hst", "ast"]].isna().any(axis=1).sum()))
    print("missing odds:", int(m[["odds_h", "odds_d", "odds_a"]].isna().any(axis=1).sum()))
    print("all team names (check for duplicates/typos):")
    print(sorted(set(m["home"]) | set(m["away"])))


def main():
    download_raw()
    m = clean(load_raw())
    sanity_check(m)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_FILE, index=False)
    print(f"saved {len(m)} matches -> {OUT_FILE}")


def read_matches(path=OUT_FILE):
    """Use this everywhere else to load the clean data with correct types."""
    return pd.read_csv(path, parse_dates=["date"], dtype={"season": str})


if __name__ == "__main__":
    main()
