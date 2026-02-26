"""
Data Loader - Load EPL match data from CSV files.

Maps to: notebooks/01_data_loading.ipynb
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import AppConfig
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


def load_season_files(data_path: str = "archive/Datasets") -> pd.DataFrame:
    """
    Load all season CSV files and combine into a single DataFrame.

    Filters out non-match files (standings, final_dataset, test files).
    Mirrors the logic from notebook 01_data_loading.

    Args:
        data_path: Path to directory containing season CSV files.

    Returns:
        Combined DataFrame with all season matches.
    """
    data_dir = Path(data_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Filter to season files only (start with digits, exclude known non-match files)
    season_files = sorted([
        f for f in data_dir.glob("*.csv")
        if f.stem[0:2].isdigit()
        and "Standing" not in f.name
        and "final" not in f.name
        and "test" not in f.name
    ])

    if not season_files:
        raise FileNotFoundError(f"No season CSV files found in {data_dir}")

    logger.info(f"Found {len(season_files)} season files in {data_dir}")

    all_seasons = []
    for file_path in season_files:
        season_label = file_path.stem
        df = pd.read_csv(file_path)
        df["Season"] = season_label
        all_seasons.append(df)
        logger.debug(f"  Loaded {season_label}: {len(df)} matches")

    combined = pd.concat(all_seasons, ignore_index=True)
    logger.info(f"Total matches loaded: {len(combined)}")
    return combined


def load_standings(data_path: str = "archive/Datasets") -> pd.DataFrame:
    """
    Load EPL standings data and reshape to long format.

    Args:
        data_path: Path to directory containing standings file.

    Returns:
        DataFrame with columns [Season, Team, Position].
    """
    standings_file = Path(data_path) / "EPLStandings.csv"

    if not standings_file.exists():
        logger.warning(f"Standings file not found: {standings_file}")
        return pd.DataFrame(columns=["Season", "Team", "Position"])

    df = pd.read_csv(standings_file)

    # Reshape: columns are seasons, rows are positions
    records = []
    for col in df.columns:
        if col.lower() in ("pos", "position", ""):
            continue
        for pos_idx, team in enumerate(df[col].dropna(), start=1):
            records.append({"Season": col, "Team": team, "Position": pos_idx})

    standings = pd.DataFrame(records)
    logger.info(f"Loaded standings: {len(standings)} team-seasons")
    return standings


def load_raw_matches(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the cleaned matches.csv from the raw directory.

    This is the output of the data loading pipeline (post-cleaning).
    """
    path = Path(raw_dir) / "matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"matches.csv not found at {path}. Run data pipeline first.")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} matches from {path}")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to CSV, creating directories as needed."""
    ensure_dir(str(Path(output_path).parent))
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def run_pipeline(config: Optional[AppConfig] = None) -> pd.DataFrame:
    """
    Execute the full data loading pipeline.

    Loads season files → cleans → saves to data/raw/matches.csv.
    """
    from src.data.cleaner import clean_data

    if config is None:
        from src.utils.config import load_config
        config = load_config()

    # Load raw season files
    raw_df = load_season_files(config.data.source_path)

    # Clean
    clean_df = clean_data(raw_df)

    # Save
    save_data(clean_df, f"{config.data.raw}/matches.csv")

    # Load and save standings
    standings = load_standings(config.data.source_path)
    if len(standings) > 0:
        save_data(standings, f"{config.data.raw}/standings.csv")

    return clean_df
