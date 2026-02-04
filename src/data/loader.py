"""
Data Loader - Load EPL match data from CSV files
"""

import pandas as pd
import glob
from pathlib import Path
from typing import List
import yaml


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_all_matches(data_path: str = 'archive/Datasets') -> pd.DataFrame:
    """
    Load all match data from CSV files and combine into single DataFrame
    
    Args:
        data_path: Path to directory containing CSV files
        
    Returns:
        Combined DataFrame with all matches
    """
    # Find all CSV files (excluding EPLStandings and explanation files)
    csv_files = glob.glob(f"{data_path}/*.csv")
    csv_files = [f for f in csv_files if 'EPLStandings' not in f and 'final' not in f]
    
    all_matches = []
    
    print(f"Loading {len(csv_files)} season files...")
    
    for file_path in sorted(csv_files):
        # Extract season from filename (e.g., "2019-20.csv" -> "2019-20")
        season = Path(file_path).stem
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Add season column
        df['Season'] = season
        
        all_matches.append(df)
        print(f"  ✓ Loaded {season}: {len(df)} matches")
    
    # Combine all seasons
    combined_df = pd.concat(all_matches, ignore_index=True)
    
    print(f"\nTotal matches loaded: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"Columns: {len(combined_df.columns)}")
    
    return combined_df


def load_standings(data_path: str = 'archive/Datasets') -> pd.DataFrame:
    """
    Load EPL standings data
    
    Args:
        data_path: Path to directory containing standings file
        
    Returns:
        DataFrame with standings data
    """
    standings_file = f"{data_path}/EPLStandings.csv"
    df = pd.read_csv(standings_file)
    print(f"Loaded standings for {len(df)} team-seasons")
    return df


def save_data(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to CSV
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    
    # Load all matches
    matches_df = load_all_matches()
    
    # Save to raw data folder
    output_path = f"{config['data']['raw']}/all_matches.csv"
    save_data(matches_df, output_path)
    
    # Load standings
    standings_df = load_standings()
    standings_output = f"{config['data']['raw']}/standings.csv"
    save_data(standings_df, standings_output)
