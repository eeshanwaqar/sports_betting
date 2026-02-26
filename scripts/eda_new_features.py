"""
EDA: Rest Days & Live League Position — Correlation Analysis
Run: python scripts/eda_new_features.py
"""
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# Load data
df = pd.read_csv("data/raw/matches.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded {len(df)} matches")

# === 1. REST DAYS ===
last_match = {}
home_rest, away_rest = [], []
for _, row in df.iterrows():
    home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
    home_rest.append((date - last_match[home]).days if home in last_match else np.nan)
    away_rest.append((date - last_match[away]).days if away in last_match else np.nan)
    last_match[home] = date
    last_match[away] = date

df["home_rest_days"] = home_rest
df["away_rest_days"] = away_rest
df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
df["target_num"] = df["FTR"].map({"H": 1, "D": 0, "A": -1})

df_rest = df.dropna(subset=["home_rest_days", "away_rest_days"]).copy()
print(f"Matches with rest data: {len(df_rest)}")

print()
print("=" * 60)
print("REST DAYS ANALYSIS")
print("=" * 60)
for feat in ["home_rest_days", "away_rest_days", "rest_diff"]:
    valid = df_rest[[feat, "target_num"]].dropna()
    corr = valid[feat].corr(valid["target_num"])
    stat, p = scipy_stats.pearsonr(valid[feat], valid["target_num"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {feat:25s}: r = {corr:+.4f}  (p = {p:.4f}) {sig}")

# ANOVA
h_r = df_rest[df_rest["FTR"] == "H"]["rest_diff"].dropna()
d_r = df_rest[df_rest["FTR"] == "D"]["rest_diff"].dropna()
a_r = df_rest[df_rest["FTR"] == "A"]["rest_diff"].dropna()
f_stat, anova_p = scipy_stats.f_oneway(h_r, d_r, a_r)
print(f"  ANOVA: F={f_stat:.3f}, p={anova_p:.4f}")

print()
print("Rest diff by outcome:")
for r in ["H", "D", "A"]:
    sub = df_rest[df_rest["FTR"] == r]["rest_diff"]
    print(f"  {r}: mean={sub.mean():.2f}, median={sub.median():.1f}")

# === 2. LIVE LEAGUE POSITION ===
print()
print("=" * 60)
print("LIVE LEAGUE POSITION ANALYSIS")
print("=" * 60)

season_stats = {}
home_positions, away_positions = [], []


def get_table_position(season, team):
    season_teams = {
        k[1]: v
        for k, v in season_stats.items()
        if k[0] == season and v["played"] > 0
    }
    if team not in season_teams or len(season_teams) < 2:
        return np.nan
    table = sorted(
        season_teams.items(),
        key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
        reverse=True,
    )
    for pos, (t, _) in enumerate(table, 1):
        if t == team:
            return pos
    return np.nan


for _, row in df.iterrows():
    home, away, season = row["HomeTeam"], row["AwayTeam"], row["Season"]
    for team in [home, away]:
        if (season, team) not in season_stats:
            season_stats[(season, team)] = {
                "points": 0,
                "gd": 0,
                "gf": 0,
                "played": 0,
            }

    home_positions.append(get_table_position(season, home))
    away_positions.append(get_table_position(season, away))

    hg, ag = int(row["FTHG"]), int(row["FTAG"])
    result = row["FTR"]
    season_stats[(season, home)]["gf"] += hg
    season_stats[(season, home)]["gd"] += hg - ag
    season_stats[(season, home)]["played"] += 1
    season_stats[(season, away)]["gf"] += ag
    season_stats[(season, away)]["gd"] += ag - hg
    season_stats[(season, away)]["played"] += 1
    if result == "H":
        season_stats[(season, home)]["points"] += 3
    elif result == "A":
        season_stats[(season, away)]["points"] += 3
    else:
        season_stats[(season, home)]["points"] += 1
        season_stats[(season, away)]["points"] += 1

df["home_league_pos"] = home_positions
df["away_league_pos"] = away_positions
df["league_pos_diff"] = df["home_league_pos"] - df["away_league_pos"]

df_pos = df.dropna(subset=["home_league_pos", "away_league_pos"]).copy()
df_pos["target_num"] = df_pos["FTR"].map({"H": 1, "D": 0, "A": -1})
print(f"Matches with position data: {len(df_pos)}")

for feat in ["home_league_pos", "away_league_pos", "league_pos_diff"]:
    valid = df_pos[[feat, "target_num"]].dropna()
    corr = valid[feat].corr(valid["target_num"])
    stat, p = scipy_stats.pearsonr(valid[feat], valid["target_num"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {feat:25s}: r = {corr:+.4f}  (p = {p:.8f}) {sig}")

# ANOVA
h_p = df_pos[df_pos["FTR"] == "H"]["league_pos_diff"].dropna()
d_p = df_pos[df_pos["FTR"] == "D"]["league_pos_diff"].dropna()
a_p = df_pos[df_pos["FTR"] == "A"]["league_pos_diff"].dropna()
f_stat, anova_p = scipy_stats.f_oneway(h_p, d_p, a_p)
print(f"  ANOVA: F={f_stat:.3f}, p={anova_p:.8f}")

print()
print("Position diff by outcome:")
for r in ["H", "D", "A"]:
    sub = df_pos[df_pos["FTR"] == r]["league_pos_diff"]
    print(f"  {r}: mean={sub.mean():.2f}, median={sub.median():.1f}")

# Sanity check
print()
print("Sanity check - avg live position:")
for team in [
    "Man United",
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Man City",
    "Sunderland",
    "Wigan",
]:
    hp = df_pos[df_pos["HomeTeam"] == team]["home_league_pos"]
    ap = df_pos[df_pos["AwayTeam"] == team]["away_league_pos"]
    all_p = pd.concat([hp, ap])
    if len(all_p) > 0:
        print(f"  {team:20s}: avg pos = {all_p.mean():.1f}")

# === 3. COMPARISON WITH EXISTING ===
print()
print("=" * 60)
print("COMPARISON WITH EXISTING FEATURES")
print("=" * 60)
model_df = pd.read_csv("data/features/model_ready.csv")
model_df["target_num"] = model_df["target"].map({"H": 1, "D": 0, "A": -1})
numeric_cols = model_df.select_dtypes(include=[np.number]).columns
exclude = ["target_num", "home_goals", "away_goals"]
feature_cols = [c for c in numeric_cols if c not in exclude]
existing_corrs = (
    model_df[list(feature_cols) + ["target_num"]]
    .corr()["target_num"]
    .drop("target_num")
    .dropna()
)
existing_corrs = existing_corrs.sort_values(key=abs, ascending=False)

rest_corr = df_rest["rest_diff"].corr(df_rest["target_num"])
pos_corr = df_pos["league_pos_diff"].corr(df_pos["target_num"])

full = pd.concat(
    [
        existing_corrs,
        pd.Series({"rest_diff": rest_corr, "league_pos_diff": pos_corr}),
    ]
)
full = full.sort_values(key=abs, ascending=False)
rest_rank = list(full.index).index("rest_diff") + 1
pos_rank = list(full.index).index("league_pos_diff") + 1

print(f"rest_diff:        rank #{rest_rank}/{len(full)} (|r| = {abs(rest_corr):.4f})")
print(f"league_pos_diff:  rank #{pos_rank}/{len(full)} (|r| = {abs(pos_corr):.4f})")
print(
    f"Top existing:     {existing_corrs.index[0]} (|r| = {abs(existing_corrs.iloc[0]):.4f})"
)
print(
    f"Weakest existing: {existing_corrs.index[-1]} (|r| = {abs(existing_corrs.iloc[-1]):.4f})"
)

# Top 20 features including new ones
print()
print("Top 20 features (existing + new):")
for i, (feat, corr_val) in enumerate(full.head(20).items(), 1):
    marker = " <-- NEW" if feat in ("rest_diff", "league_pos_diff") else ""
    print(f"  {i:2d}. {feat:30s}: r = {corr_val:+.4f}{marker}")

# === 4. MULTICOLLINEARITY CHECK ===
print()
print("=" * 60)
print("MULTICOLLINEARITY CHECK")
print("=" * 60)
df_pos["match_key"] = (
    df_pos["HomeTeam"] + "_" + df_pos["AwayTeam"] + "_" + df_pos["Date"].astype(str)
)
model_df["match_key"] = (
    model_df["home_team"]
    + "_"
    + model_df["away_team"]
    + "_"
    + model_df["date"].astype(str)
)
merged = model_df.merge(
    df_pos[["match_key", "league_pos_diff", "rest_diff"]].dropna(),
    on="match_key",
    how="inner",
)
print(f"Merged rows: {len(merged)}")

check_cols = [
    "elo_diff",
    "diff_season_win_rate",
    "diff_season_ppg",
    "diff_form5_points",
    "odds_home_away_diff",
]
print(f"{'Feature':30s} {'league_pos_diff':>16s} {'rest_diff':>12s}")
print("-" * 60)
for col in check_cols:
    if col in merged.columns:
        pc = merged["league_pos_diff"].corr(merged[col])
        rc = merged["rest_diff"].corr(merged[col])
        print(f"{col:30s} {pc:+.4f}          {rc:+.4f}")

print()
print("(|r| > 0.7 = high redundancy, |r| < 0.3 = independent info)")
