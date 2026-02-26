"""
Data Validator - Validate data quality after loading/cleaning.

Checks for common data issues before proceeding to feature engineering.
"""

import pandas as pd
from typing import List

from src.utils.logger import get_logger
from src.utils.constants import ESSENTIAL_COLUMNS, VALID_RESULTS

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


def validate_matches(df: pd.DataFrame, strict: bool = False) -> List[str]:
    """
    Validate match data quality.

    Args:
        df: Cleaned match DataFrame.
        strict: If True, raise ValidationError on failures. Otherwise log warnings.

    Returns:
        List of warning/error messages.
    """
    issues: List[str] = []

    # Check essential columns exist
    for col in ESSENTIAL_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing essential column: {col}")

    # Check for NaN in essential columns
    for col in ESSENTIAL_COLUMNS:
        if col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                issues.append(f"{col} has {na_count} null values")

    # Check FTR values
    if "FTR" in df.columns:
        invalid = df[~df["FTR"].isin(VALID_RESULTS)]
        if len(invalid) > 0:
            issues.append(f"{len(invalid)} rows have invalid FTR values: {invalid['FTR'].unique()}")

    # Check date ordering
    if "Date" in df.columns:
        if not df["Date"].is_monotonic_increasing:
            issues.append("Data is not sorted by date")

    # Check for duplicate matches
    if all(c in df.columns for c in ["Date", "HomeTeam", "AwayTeam"]):
        dupes = df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum()
        if dupes > 0:
            issues.append(f"{dupes} duplicate matches found")

    # Check goal values are non-negative
    for col in ["FTHG", "FTAG"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"{col} has {neg} negative values")

    # Report
    if issues:
        for issue in issues:
            if strict:
                logger.error(f"VALIDATION FAILED: {issue}")
            else:
                logger.warning(f"Validation warning: {issue}")
        if strict:
            raise ValidationError(f"Data validation failed with {len(issues)} issues")
    else:
        logger.info("Data validation passed - no issues found")

    return issues
