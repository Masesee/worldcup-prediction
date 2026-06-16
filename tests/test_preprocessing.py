import os
import tempfile
import pandas as pd
import pytest
from pipelines.preprocessing import (
    clean_countries,
    run_elo_simulation,
    preprocess_data,
)


def test_clean_countries():
    df = pd.DataFrame(
        {
            "country": ["Czechoslovakia", "Zaire", "Turkey", "Ivory Coast", "Brazil"],
            "opponent": ["Germany", "Turkey", "Czechoslovakia", "Brazil", "Zaire"],
        }
    )
    df_clean = clean_countries(df, ["country", "opponent"])

    expected_country = ["Czechia", "DR Congo", "Turkiye", "Cote d'Ivoire", "Brazil"]
    expected_opponent = ["Germany", "Turkiye", "Czechia", "Brazil", "DR Congo"]

    assert df_clean["country"].tolist() == expected_country
    assert df_clean["opponent"].tolist() == expected_opponent


def test_run_elo_simulation():
    # Construct a small mock matches DataFrame
    matches_data = {
        "tournament_id": ["WC-1930", "WC-1930", "WC-1934"],
        "tournament_name": [
            "1930 FIFA Men's World Cup",
            "1930 FIFA Men's World Cup",
            "1934 FIFA Men's World Cup",
        ],
        "match_id": ["M-01", "M-02", "M-03"],
        "match_date": ["1930-07-13", "1930-07-14", "1934-05-27"],
        "home_team_name": ["France", "Argentina", "Italy"],
        "away_team_name": ["Mexico", "France", "United States"],
        "country_name": ["Uruguay", "Uruguay", "Italy"],
        "home_team_win": [1, 1, 1],
        "away_team_win": [0, 0, 0],
        "draw": [0, 0, 0],
    }
    matches_df = pd.DataFrame(matches_data)

    final_ratings, tournament_priors = run_elo_simulation(
        matches_df, k_factor=32.0, home_advantage=100.0
    )

    # Mexico plays 1 match and loses, so rating should decrease (<1500).
    # Argentina plays 1 match and wins, so rating should increase (>1500).
    assert final_ratings["Argentina"] > 1500.0
    assert final_ratings["Mexico"] < 1500.0

    # Italy plays United States in 1934. Host is Italy, so Italy has home advantage.
    # Let's verify tournament priors are recorded correctly.
    assert ("WC-1930", "France") in tournament_priors
    assert tournament_priors[("WC-1930", "France")] == 1500.0


def test_preprocess_data():
    # Create temp files for Train, Test, and Matches
    with tempfile.TemporaryDirectory() as tmpdir:
        train_data = {
            "ID": ["WC-1930_ARG", "WC-1934_ITA"],
            "tournament_id": ["WC-1930", "WC-1934"],
            "country": ["Argentina", "Italy"],
            "team_id": ["T-03", "T-41"],
            "team_code": ["ARG", "ITA"],
            "confederation_name": ["CONMEBOL", "UEFA"],
            "region_name": ["South America", "Europe"],
            "tournament_name": ["1930 FIFA Men's World Cup", "1934 FIFA Men's World Cup"],
            "year": [1930, 1934],
            "matches_played": [5, 5],
            "total_goals": [18, 12],
            "stage_reached": ["final", "final"],
        }
        test_data = {"ID": ["WC-2026_ARG", "WC-2026_ITA"], "country": ["Argentina", "Italy"]}
        matches_data = {
            "tournament_id": ["WC-1930", "WC-1934"],
            "tournament_name": ["1930 FIFA Men's World Cup", "1934 FIFA Men's World Cup"],
            "match_id": ["M-01", "M-02"],
            "match_date": ["1930-07-13", "1934-05-27"],
            "home_team_name": ["Argentina", "Italy"],
            "away_team_name": ["France", "Germany"],
            "country_name": ["Uruguay", "Italy"],
            "home_team_win": [1, 1],
            "away_team_win": [0, 0],
            "draw": [0, 0],
        }

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        matches_df = pd.DataFrame(matches_data)

        train_df.to_csv(os.path.join(tmpdir, "Train.csv"), index=False)
        test_df.to_csv(os.path.join(tmpdir, "Test.csv"), index=False)
        matches_df.to_csv(os.path.join(tmpdir, "matches.csv"), index=False)

        processed_dir = os.path.join(tmpdir, "processed")
        preprocess_data(tmpdir, processed_dir)

        # Verify processed files exist
        assert os.path.exists(os.path.join(processed_dir, "train_processed.csv"))
        assert os.path.exists(os.path.join(processed_dir, "test_processed.csv"))

        train_proc = pd.read_csv(os.path.join(processed_dir, "train_processed.csv"))
        test_proc = pd.read_csv(os.path.join(processed_dir, "test_processed.csv"))

        # Check Elo prior column was added
        assert "elo_rating_prior" in train_proc.columns
        assert "elo_rating_prior" in test_proc.columns
        assert train_proc.loc[train_proc["country"] == "Argentina", "elo_rating_prior"].values[0] == 1500.0
