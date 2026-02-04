"""
Data Cleaner - Clean and standardize EPL match data
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize match data
    
    Args:
        df: Raw match DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("Cleaning data...")
    df = df.copy()
    
    # 1. Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
    
    # 2. Handle missing values in key columns
    # Keep only rows with essential data
    essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    df = df.dropna(subset=essential_cols)
    
    # 3. Standardize team names (remove extra spaces)
    df['HomeTeam'] = df['HomeTeam'].str.strip()
    df['AwayTeam'] = df['AwayTeam'].str.strip()
    
    # 4. Fix data types
    df['FTHG'] = df['FTHG'].astype(int)  # Full Time Home Goals
    df['FTAG'] = df['FTAG'].astype(int)  # Full Time Away Goals
    
    # 5. Fill missing statistics with 0
    stat_columns = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    for col in stat_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    # 6. Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 7. Add additional useful columns
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['GoalDifference'] = df['FTHG'] - df['FTAG']
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    print(f"✓ Cleaned {len(df)} matches")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Unique teams: {df['HomeTeam'].nunique()}")
    print(f"  Columns: {len(df.columns)}")
    
    return df


def get_data_summary(df: pd.DataFrame):
    """
    Print summary statistics of the dataset
    """
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    print(f"\nTotal matches: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Seasons: {df['Season'].nunique()}")
    print(f"Teams: {df['HomeTeam'].nunique()}")
    
    print("\n--- Match Results ---")
    result_counts = df['FTR'].value_counts()
    print(f"Home wins (H): {result_counts.get('H', 0)} ({result_counts.get('H', 0)/len(df)*100:.1f}%)")
    print(f"Draws (D):     {result_counts.get('D', 0)} ({result_counts.get('D', 0)/len(df)*100:.1f}%)")
    print(f"Away wins (A): {result_counts.get('A', 0)} ({result_counts.get('A', 0)/len(df)*100:.1f}%)")
    
    print("\n--- Goals Statistics ---")
    print(f"Average home goals: {df['FTHG'].mean():.2f}")
    print(f"Average away goals: {df['FTAG'].mean():.2f}")
    print(f"Average total goals: {df['TotalGoals'].mean():.2f}")
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load raw data
    raw_path = f"{config['data']['raw']}/all_matches.csv"
    df = pd.read_csv(raw_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Show summary
    get_data_summary(df_clean)
    
    # Save cleaned data
    output_path = f"{config['data']['processed']}/clean_matches.csv"
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
