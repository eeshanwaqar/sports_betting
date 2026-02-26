"""
Make Predictions - Predict match outcomes using the production inference pipeline.

Usage:
    python scripts/predict.py --home Arsenal --away Chelsea
    python scripts/predict.py --home "Man City" --away Liverpool --date 2024-03-15
    python scripts/predict.py --list-teams
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.inference.predictor import MatchPredictor


def print_prediction(result: dict) -> None:
    """Pretty-print a prediction result dict."""
    probs = result["probabilities"]
    odds = result["odds"]

    print(f"\n  Predicted outcome: {result['prediction']}")
    print(f"  Confidence:        {result['confidence']:.1%}")
    print()
    print("  Probabilities            Odds")
    print(f"  Home Win (H): {probs['home_win']:.1%}       {odds['home_win']:.2f}")
    print(f"  Draw    (D):  {probs['draw']:.1%}       {odds['draw']:.2f}")
    print(f"  Away Win(A):  {probs['away_win']:.1%}       {odds['away_win']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict EPL match outcome")
    parser.add_argument("--home", type=str, help="Home team name")
    parser.add_argument("--away", type=str, help="Away team name")
    parser.add_argument("--date", type=str, default=None, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--list-teams", action="store_true", help="List available teams")
    args = parser.parse_args()

    print("=" * 60)
    print("EPL BETTING ODDS PREDICTOR")
    print("=" * 60)

    config = load_config()
    print("\nLoading model and historical data ...")
    predictor = MatchPredictor(config)

    if args.list_teams:
        teams = predictor.get_available_teams()
        print(f"\nAvailable teams ({len(teams)}):")
        for team in teams:
            print(f"  - {team}")
        return

    if not args.home or not args.away:
        parser.error("--home and --away are required (or use --list-teams)")

    # Validate teams
    available = predictor.get_available_teams()
    for team, label in [(args.home, "Home"), (args.away, "Away")]:
        if team not in available:
            print(f"\nERROR: {label} team '{team}' not found.")
            print("Use --list-teams to see available names.")
            sys.exit(1)

    if args.home == args.away:
        print("\nERROR: Home and away teams must be different.")
        sys.exit(1)

    match_date = None
    if args.date:
        match_date = datetime.strptime(args.date, "%Y-%m-%d")

    print(f"\nPredicting: {args.home} vs {args.away}")
    print("-" * 40)

    result = predictor.predict(args.home, args.away, match_date)
    print_prediction(result)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
