"""
Constants - Application-wide constants and enums.
"""

from enum import Enum
from typing import List


# --- Match Outcomes ---
class MatchResult(str, Enum):
    HOME = "H"
    DRAW = "D"
    AWAY = "A"


VALID_RESULTS: List[str] = [MatchResult.HOME, MatchResult.DRAW, MatchResult.AWAY]

# Points mapping
POINTS_MAP = {"W": 3, "D": 1, "L": 0}

# Result mapping (from FTR perspective of home/away)
HOME_RESULT_MAP = {"H": "W", "D": "D", "A": "L"}
AWAY_RESULT_MAP = {"H": "L", "D": "D", "A": "W"}

# --- Column Names ---
ESSENTIAL_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]

ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

META_COLUMNS = ["match_id", "date", "home_team", "away_team", "season"]

TARGET_COLUMNS = ["target", "home_goals", "away_goals"]

# --- Feature Defaults ---
DEFAULT_FORM_WINDOWS = [3, 5, 10]
DEFAULT_EXP_DECAY = 0.7
DEFAULT_VENUE_WINDOW = 5
DEFAULT_H2H_WINDOW = 10
DEFAULT_SHOOTING_WINDOW = 10

# --- Elo Defaults ---
DEFAULT_ELO_K = 20
DEFAULT_ELO_HOME_ADV = 100
DEFAULT_INITIAL_ELO = 1500

# --- Model Defaults ---
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_ODDS_MARGIN = 0.05
