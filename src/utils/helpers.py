"""
Helpers - Common utility functions used across modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_season_start(date: pd.Timestamp) -> pd.Timestamp:
    """
    Get approximate EPL season start date for a given date.
    Seasons start in August and end in May.
    """
    if date.month >= 8:
        return pd.Timestamp(f"{date.year}-08-01")
    return pd.Timestamp(f"{date.year - 1}-08-01")


def get_season_label(date: pd.Timestamp) -> str:
    """Return season label like '2017-18' from a date."""
    if date.month >= 8:
        return f"{date.year}-{str(date.year + 1)[-2:]}"
    return f"{date.year - 1}-{str(date.year)[-2:]}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning default if denominator is zero."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator
