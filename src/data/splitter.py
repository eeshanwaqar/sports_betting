"""
Data Splitter - Time-based train/test split utilities.

Maps to: notebooks/04_modeling.ipynb (split cell)

Key principle: Always split by time to prevent data leakage.
Train on past, test on future.
"""

from typing import Tuple

import pandas as pd

from src.utils.constants import DEFAULT_TEST_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def time_based_split(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically: first (1 - test_size) for training, rest for testing.

    Args:
        df: DataFrame with a date column, assumed sorted.
        test_size: Fraction of data to use as test set.
        date_col: Name of the date column.

    Returns:
        Tuple of (train_df, test_df).
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    logger.info(
        f"Time split: {len(train_df)} train "
        f"({train_df[date_col].min().date()} to {train_df[date_col].max().date()}), "
        f"{len(test_df)} test "
        f"({test_df[date_col].min().date()} to {test_df[date_col].max().date()})"
    )
    return train_df, test_df
