import os
import tempfile

import pandas as pd

from pipelines.features import (
    extract_features,
    get_team_confederation,
    get_tournament_stages,
)


def test_get_team_confederation():
    train_data = {
        "country": ["Argentina", "France"],
        "confederation_name": ["CONMEBOL", "UEFA"],
    }
    train_df = pd.DataFrame(train_data)

    assert get_team_confederation("Argentina", train_df) == "CONMEBOL"
    assert get_team_confederation("Cabo Verde", train_df) == "Confederation of African Football"
    assert get_team_confederation("Uzbekistan", train_df) == "Asian Football Confederation"


def test_get_tournament_stages():
    train_data = {
        "tournament_id": ["WC-1930", "WC-1930", "WC-1934"],
        "team_id": ["T-01", "T-02", "T-03"],
        "stage_reached": ["group stage", "semi-finals", "final"],
    }
    standings_data = {
        "tournament_id": ["WC-1930", "WC-1930", "WC-1934"],
        "team_id": ["T-01", "T-02", "T-03"],
        "position": [5, 3, 1],
    }

    train_df = pd.DataFrame(train_data)
    standings_df = pd.DataFrame(standings_data)

    train_mapped = get_tournament_stages(train_df, standings_df)

    assert train_mapped.loc[train_mapped["team_id"] == "T-01", "Target"].values[0] == "group"
    assert train_mapped.loc[train_mapped["team_id"] == "T-02", "Target"].values[0] == "sf"
    assert train_mapped.loc[train_mapped["team_id"] == "T-03", "Target"].values[0] == "champion"


def test_extract_features():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal raw files
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
            "home_team_score": [1, 2],
            "away_team_score": [0, 1],
        }
        tournaments_data = {
            "tournament_id": ["WC-1930", "WC-1934"],
            "tournament_name": ["1930 FIFA Men's World Cup", "1934 FIFA Men's World Cup"],
            "start_date": ["1930-07-13", "1934-05-27"],
            "host_country": ["Uruguay", "Italy"],
        }
        standings_data = {
            "tournament_id": ["WC-1930", "WC-1934"],
            "team_id": ["T-03", "T-41"],
            "position": [2, 1],
        }

        # Save to raw
        pd.DataFrame(train_data).to_csv(os.path.join(tmpdir, "Train.csv"), index=False)
        pd.DataFrame(test_data).to_csv(os.path.join(tmpdir, "Test.csv"), index=False)
        pd.DataFrame(matches_data).to_csv(os.path.join(tmpdir, "matches.csv"), index=False)
        pd.DataFrame(tournaments_data).to_csv(os.path.join(tmpdir, "tournaments.csv"), index=False)
        pd.DataFrame(standings_data).to_csv(
            os.path.join(tmpdir, "tournament_standings.csv"), index=False
        )

        # Create processed folder and write processed train/test with Elo prior
        train_proc = pd.DataFrame(train_data)
        train_proc["elo_rating_prior"] = [1500.0, 1500.0]
        test_proc = pd.DataFrame(test_data)
        test_proc["elo_rating_prior"] = [1500.0, 1500.0]

        os.makedirs(os.path.join(tmpdir, "processed"), exist_ok=True)
        train_proc_path = os.path.join(tmpdir, "processed/train_processed.csv")
        test_proc_path = os.path.join(tmpdir, "processed/test_processed.csv")
        train_proc.to_csv(train_proc_path, index=False)
        test_proc.to_csv(test_proc_path, index=False)

        processed_dir = os.path.join(tmpdir, "processed")
        train_feat, test_feat = extract_features(
            train_proc_path, test_proc_path, tmpdir, processed_dir
        )

        assert os.path.exists(os.path.join(processed_dir, "train_features.parquet"))
        assert os.path.exists(os.path.join(processed_dir, "test_features.parquet"))

        # Verify correct features were generated
        assert "historical_goals_scored_per_match" in train_feat.columns
        assert "elo_rating_prior" in test_feat.columns
        assert "qualified_last_tournament" in train_feat.columns
        assert "qualified_two_tournaments_ago" in train_feat.columns
        assert "goals_scored_last_wc" in train_feat.columns
        assert "goals_conceded_last_wc" in train_feat.columns
