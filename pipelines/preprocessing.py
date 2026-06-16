"""Preprocessing pipeline for the World Cup 2026 Goal Prediction Challenge.

Handles entity cleaning, name mappings, and historical Elo rating computations.
"""

import os
import pandas as pd
from typing import Dict, Tuple

# Country name mapping to align historical data with the 2026 test set
COUNTRY_MAPPING = {
    "Czechoslovakia": "Czechia",
    "Turkey": "Turkiye",
    "Ivory Coast": "Cote d'Ivoire",
    "Zaire": "DR Congo",
}


def clean_countries(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replaces historical country names with modern aligned names in specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(COUNTRY_MAPPING)
    return df


def run_elo_simulation(
    matches_df: pd.DataFrame, k_factor: float = 32.0, home_advantage: float = 100.0
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Runs a match-by-match Elo rating simulation for Men's matches.

    Snapshots team ratings prior to each tournament (before their first match of that tournament).

    Args:
        matches_df: DataFrame of raw matches.
        k_factor: K-factor for Elo rating updates.
        home_advantage: Temporary Elo boost for the host team in expected score calculation.

    Returns:
        final_ratings: Dict of final ratings at the end of 2022.
        tournament_priors: Dict of prior ratings keyed by (tournament_id, team_name).
    """
    # Filter to only include Men's matches
    mens_matches = matches_df[
        matches_df["tournament_name"].str.contains("Men", na=False)
        & ~matches_df["tournament_name"].str.contains("Women", na=False)
    ].copy()

    # Clean country names
    mens_matches = clean_countries(
        mens_matches, ["home_team_name", "away_team_name", "country_name"]
    )

    # Sort matches chronologically
    mens_matches["match_date"] = pd.to_datetime(mens_matches["match_date"])
    mens_matches = mens_matches.sort_values(by=["match_date", "match_id"]).reset_index(drop=True)

    ratings: Dict[str, float] = {}
    tournament_priors: Dict[Tuple[str, str], float] = {}

    current_tournament = None
    snapshotted_teams = set()

    for _, row in mens_matches.iterrows():
        tour_id = row["tournament_id"]
        home = row["home_team_name"]
        away = row["away_team_name"]
        host = row["country_name"]

        # Initialize ratings if team is new
        r_home = ratings.setdefault(home, 1500.0)
        r_away = ratings.setdefault(away, 1500.0)

        # Snapshot prior Elo at the start of a tournament
        if tour_id != current_tournament:
            current_tournament = tour_id
            snapshotted_teams = set()

        if home not in snapshotted_teams:
            tournament_priors[(tour_id, home)] = r_home
            snapshotted_teams.add(home)

        if away not in snapshotted_teams:
            tournament_priors[(tour_id, away)] = r_away
            snapshotted_teams.add(away)

        # Check home advantage (if team name matches host country name)
        adj_home = r_home + (home_advantage if home == host else 0.0)
        adj_away = r_away + (home_advantage if away == host else 0.0)

        # Expected scores
        exp_home = 1.0 / (1.0 + 10.0 ** ((adj_away - adj_home) / 400.0))
        exp_away = 1.0 - exp_home

        # Actual outcomes
        if row["home_team_win"] == 1:
            s_home, s_away = 1.0, 0.0
        elif row["away_team_win"] == 1:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Update ratings
        ratings[home] = r_home + k_factor * (s_home - exp_home)
        ratings[away] = r_away + k_factor * (s_away - exp_away)

    return ratings, tournament_priors


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Loads raw files, cleans names, runs Elo simulation, and saves processed datasets."""
    # Load raw data
    train_df = pd.read_csv(os.path.join(raw_dir, "Train.csv"))
    test_df = pd.read_csv(os.path.join(raw_dir, "Test.csv"))
    matches_df = pd.read_csv(os.path.join(raw_dir, "matches.csv"))

    # Clean country names in Train and Test
    train_cleaned = clean_countries(train_df, ["country"])
    test_cleaned = clean_countries(test_df, ["country"])

    # Run Elo simulation
    final_ratings, tournament_priors = run_elo_simulation(matches_df)

    # Map prior Elo ratings to Train dataset
    train_elos = []
    for _, row in train_cleaned.iterrows():
        tour_id = row["tournament_id"]
        team = row["country"]
        elo = tournament_priors.get((tour_id, team), 1500.0)
        train_elos.append(elo)
    train_cleaned["elo_rating_prior"] = train_elos

    # Map prior Elo ratings to Test dataset (based on final ratings at the end of 2022)
    test_elos = []
    for _, row in test_cleaned.iterrows():
        team = row["country"]
        elo = final_ratings.get(team, 1500.0)
        test_elos.append(elo)
    test_cleaned["elo_rating_prior"] = test_elos

    # Save processed files
    os.makedirs(processed_dir, exist_ok=True)
    train_cleaned.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    test_cleaned.to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)
    print("Preprocessed files written successfully to data/processed/")


if __name__ == "__main__":
    preprocess_data("data/raw", "data/processed")

