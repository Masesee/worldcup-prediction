import os
import tempfile
import json
import pandas as pd
import numpy as np
import pytest
from pipelines.training.tune import (
    run_validation_capacity_mapping,
    evaluate_goals_model,
    evaluate_stage_model,
    tune_pipelines,
)
from pipelines.training.train import (
    run_test_capacity_mapping,
    train_and_predict,
)

FEATURES = [
    "elo_rating_prior",
    "is_host",
    "historical_goals_scored_per_match",
    "historical_goals_conceded_per_match",
    "historical_win_rate",
    "historical_avg_stage_reached",
    "confederation_avg_goals_scored",
    "confederation_avg_stage_reached",
    "decayed_goals_scored",
    "decayed_goals_conceded",
    "decayed_stage_reached",
    "historical_clean_sheet_rate",
    "historical_failed_to_score_rate",
    "historical_appearances",
    "recent_appearances_count",
]


def test_run_validation_capacity_mapping():
    # Construct mock data for 32 teams
    val_data = {
        "ID": [f"T-{i}" for i in range(32)],
        "Target": ["group"] * 32,
    }
    val_df = pd.DataFrame(val_data)
    preds = np.arange(32.0, 0.0, -1.0)  # Descending scores

    mapped_df = run_validation_capacity_mapping(val_df, preds)

    # Verify counts
    counts = mapped_df["Mapped_Target"].value_counts().to_dict()
    assert counts["champion"] == 1
    assert counts["runnerup"] == 1
    assert counts["sf"] == 2
    assert counts["qf"] == 4
    assert counts["roundof16"] == 8
    assert counts["group"] == 16


def test_run_test_capacity_mapping():
    # Construct mock data for 48 teams
    test_data = {
        "ID": [f"T-{i}" for i in range(48)],
    }
    test_df = pd.DataFrame(test_data)
    preds = np.arange(48.0, 0.0, -1.0)  # Descending scores

    mapped_df = run_test_capacity_mapping(test_df, preds)

    # Verify counts
    counts = mapped_df["Target"].value_counts().to_dict()
    assert counts["champion"] == 1
    assert counts["runnerup"] == 1
    assert counts["sf"] == 2
    assert counts["qf"] == 4
    assert counts["roundof16"] == 8
    assert counts["roundof32"] == 16
    assert counts["group"] == 16


def test_evaluate_goals_model():
    # Construct mock data with years < 2018, 2018, and 2022
    data = []
    for year in [2010, 2014, 2018, 2022]:
        for i in range(32):
            row = {f: np.random.randn() for f in FEATURES}
            row["year"] = year
            row["total_goals"] = np.random.randint(0, 10)
            row["Target_ordinal"] = np.random.randint(0, 6)
            row["Target"] = "group"
            data.append(row)

    df = pd.DataFrame(data)
    params = {"n_estimators": 5, "max_depth": 3, "random_state": 42}

    rmse = evaluate_goals_model(params, df)
    assert isinstance(rmse, float)
    assert rmse > 0.0


def test_tune_and_train_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Construct mock feature tables for train and test
        train_data = []
        for year in [2010, 2014, 2018, 2022]:
            # Each tournament has 32 teams
            for i in range(32):
                row = {f: np.random.randn() for f in FEATURES}
                row["ID"] = f"WC-{year}_T-{i}"
                row["year"] = year
                row["matches_played"] = 3
                row["total_goals"] = np.random.randint(0, 10)
                row["Target"] = "group"
                row["Target_ordinal"] = np.random.randint(0, 6)
                row["is_host"] = 0
                train_data.append(row)

        test_data = []
        # 2026 has 48 teams
        for i in range(48):
            row = {f: np.random.randn() for f in FEATURES}
            row["ID"] = f"WC-2026_T-{i}"
            row["is_host"] = 0
            test_data.append(row)

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Save to temporary parquet files
        train_df.to_parquet(os.path.join(tmpdir, "train_features.parquet"), index=False)
        test_df.to_parquet(os.path.join(tmpdir, "test_features.parquet"), index=False)

        # Run Optuna tuning (limit n_trials for speed in tests)
        # We modify optuna.create_study mock or just run it.
        # To make it fast, we can run tune_pipelines.
        # But wait! Optuna tuning is fast with n_trials=50, but in tests, 50 trials might take 2-3 seconds.
        # Let's run it.
        tune_pipelines(tmpdir, tmpdir)

        # Verify best_params.json exists
        best_params_path = os.path.join(tmpdir, "best_params.json")
        assert os.path.exists(best_params_path)

        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        assert "goals" in best_params
        assert "stage" in best_params

        # Run training
        submissions_dir = os.path.join(tmpdir, "submissions")
        train_and_predict(tmpdir, tmpdir, submissions_dir)

        # Verify models and submission files exist
        assert os.path.exists(os.path.join(tmpdir, "goals_model.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "stage_model.pkl"))
        assert os.path.exists(os.path.join(submissions_dir, "submission.csv"))

        sub_df = pd.read_csv(os.path.join(submissions_dir, "submission.csv"))
        assert len(sub_df) == 48
        assert list(sub_df.columns) == ["ID", "total_goals", "Target"]
