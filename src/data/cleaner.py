"""
Data Cleaner - Clean and standardize EPL match data.

Maps to: notebooks/01_data_loading.ipynb (cleaning cells)

Key operations:
- Parse dates
- Filter invalid FTR values (e.g. 'NH')
- Drop rows with missing essential data
- Standardize data types
"""

import pandas as pd

from src.utils.constants import ESSENTIAL_COLUMNS, STAT_COLUMNS, VALID_RESULTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw match data.

    This mirrors the cleaning logic from notebook 01:
    1. Parse dates
    2. Filter valid FTR only (H, D, A)
    3. Drop rows missing essential columns
    4. Standardize types
    5. Fill missing stats with 0
    6. Sort chronologically

    Args:
        df: Raw combined DataFrame from loader.

    Returns:
        Cleaned, sorted DataFrame.
    """
    logger.info(f"Cleaning {len(df)} raw matches...")
    df = df.copy()

    # 1. Parse dates (handles both DD/MM/YY and DD/MM/YYYY formats)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # 2. Filter valid match results only (removes 'NH' and other anomalies)
    valid_ftr = df["FTR"].isin(VALID_RESULTS)
    removed_ftr = (~valid_ftr).sum()
    if removed_ftr > 0:
        logger.warning(f"Removed {removed_ftr} rows with invalid FTR values")
    df = df[valid_ftr]

    # 3. Drop rows missing essential data
    before = len(df)
    df = df.dropna(subset=ESSENTIAL_COLUMNS)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing essential data")

    # 4. Standardize team names
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()

    # 5. Fix numeric types
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # 6. Fill missing statistics with 0
    for col in STAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # 7. Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(
        f"Cleaning complete: {len(df)} matches, "
        f"{df['Date'].min().date()} to {df['Date'].max().date()}, "
        f"{df['HomeTeam'].nunique()} teams"
    )
    return df
