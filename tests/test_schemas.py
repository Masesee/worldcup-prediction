import pytest
from pydantic import ValidationError

from contracts.schema import InferenceRowSchema, SubmissionRowSchema, TrainRowSchema


def test_train_row_schema_validation():
    # Valid row
    valid_data = {
        "ID": "WC-1930_ARG",
        "team_id": "T-03",
        "country": "Argentina",
        "team_code": "ARG",
        "confederation_name": "South American Football Confederation",
        "region_name": "South America",
        "tournament_id": "WC-1930",
        "tournament_name": "1930 FIFA Men's World Cup",
        "year": 1930,
        "matches_played": 5,
        "total_goals": 18,
        "stage_reached": "final",
    }
    schema = TrainRowSchema(**valid_data)
    assert schema.ID == "WC-1930_ARG"
    assert schema.year == 1930

    # Invalid year
    invalid_data = valid_data.copy()
    invalid_data["year"] = 1920
    with pytest.raises(ValidationError):
        TrainRowSchema(**invalid_data)

    # Invalid negative goals
    invalid_data = valid_data.copy()
    invalid_data["total_goals"] = -1
    with pytest.raises(ValidationError):
        TrainRowSchema(**invalid_data)


def test_test_row_schema_validation():
    valid_data = {"ID": "WC-2026_AUT", "country": "Austria"}
    schema = InferenceRowSchema(**valid_data)
    assert schema.ID == "WC-2026_AUT"


def test_submission_row_schema_validation():
    # Valid target
    valid_data = {"ID": "WC-2026_AUT", "total_goals": 3.5, "Target": "group"}
    schema = SubmissionRowSchema(**valid_data)
    assert schema.Target == "group"

    # Invalid target stage
    invalid_data = {"ID": "WC-2026_AUT", "total_goals": 3.5, "Target": "invalid_stage"}
    with pytest.raises(ValidationError):
        SubmissionRowSchema(**invalid_data)


def test_processed_data_validation():
    import os

    import pandas as pd

    processed_dir = "worldcup_prediction/data/processed"
    if os.path.exists(processed_dir):
        # Validate train_processed.csv
        train_path = os.path.join(processed_dir, "train_processed.csv")
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            for _, row in train_df.iterrows():
                row_dict = row.to_dict()
                # Remove extra columns that are not part of the base TrainRowSchema
                schema_data = {
                    k: v for k, v in row_dict.items() if k in TrainRowSchema.model_fields
                }
                TrainRowSchema(**schema_data)

        # Validate test_processed.csv
        test_path = os.path.join(processed_dir, "test_processed.csv")
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            for _, row in test_df.iterrows():
                row_dict = row.to_dict()
                schema_data = {
                    k: v for k, v in row_dict.items() if k in InferenceRowSchema.model_fields
                }
                InferenceRowSchema(**schema_data)

    # Validate outputs/submissions/submission.csv
    sub_path = "worldcup_prediction/outputs/submissions/submission.csv"
    if os.path.exists(sub_path):
        sub_df = pd.read_csv(sub_path)
        for _, row in sub_df.iterrows():
            SubmissionRowSchema(**row.to_dict())
