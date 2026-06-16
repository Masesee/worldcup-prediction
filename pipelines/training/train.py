"""Model training and prediction pipeline.

Trains final LightGBM models on historical data and generates predictions
for the 2026 World Cup using 48-team capacity mapping constraints.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb

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
]

DEFAULT_GOALS_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 10,
    "n_estimators": 100,
    "random_state": 42,
}

DEFAULT_STAGE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 10,
    "n_estimators": 100,
    "random_state": 42,
}


def run_test_capacity_mapping(test_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    """Performs tournament capacity mapping on 2026 test predictions.

    Validates against 48-team tournament format.
    """
    test_df = test_df.copy()
    test_df["pred_score"] = preds

    # Sort test teams by predicted score descending
    test_df = test_df.sort_values(by="pred_score", ascending=False).reset_index(drop=True)

    # 48-team capacities
    # Rank 1: champion
    # Rank 2: runnerup
    # Ranks 3-4: sf
    # Ranks 5-8: qf
    # Ranks 9-16: roundof16
    # Ranks 17-32: roundof32
    # Ranks 33-48: group
    mapped_stages = []
    for i in range(len(test_df)):
        rank = i + 1
        if rank == 1:
            mapped_stages.append("champion")
        elif rank == 2:
            mapped_stages.append("runnerup")
        elif rank in [3, 4]:
            mapped_stages.append("sf")
        elif rank in range(5, 9):
            mapped_stages.append("qf")
        elif rank in range(9, 17):
            mapped_stages.append("roundof16")
        elif rank in range(17, 33):
            mapped_stages.append("roundof32")
        else:
            mapped_stages.append("group")

    test_df["Target"] = mapped_stages
    return test_df


def train_and_predict(
    processed_dir: str, models_dir: str, submissions_dir: str
) -> None:
    """Trains final models and runs predictions for the 2026 World Cup."""
    train_feat_path = os.path.join(processed_dir, "train_features.parquet")
    test_feat_path = os.path.join(processed_dir, "test_features.parquet")

    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        print("Error: Parquet feature matrices not found.")
        return

    train_df = pd.read_parquet(train_feat_path)
    test_df = pd.read_parquet(test_feat_path)

    # Load best hyperparameters if they exist, otherwise use defaults
    best_params_path = os.path.join(models_dir, "best_params.json")
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        goals_params = best_params.get("goals", DEFAULT_GOALS_PARAMS)
        stage_params = best_params.get("stage", DEFAULT_STAGE_PARAMS)
        print("Loaded tuned hyperparameters from best_params.json")
    else:
        goals_params = DEFAULT_GOALS_PARAMS
        stage_params = DEFAULT_STAGE_PARAMS
        print("No tuned parameters found. Using default parameters.")

    # 1. Train Goals Predictor
    print("Training goals prediction model...")
    X_train, y_train = train_df[FEATURES], train_df["total_goals"]
    goals_model = lgb.LGBMRegressor(**goals_params, verbose=-1)
    goals_model.fit(X_train, y_train)

    # Save goals model
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "goals_model.pkl"), "wb") as f:
        pickle.dump(goals_model, f)

    # 2. Train Stage Predictor
    print("Training stage prediction model...")
    X_train_stage, y_train_stage = train_df[FEATURES], train_df["Target_ordinal"]
    stage_model = lgb.LGBMRegressor(**stage_params, verbose=-1)
    stage_model.fit(X_train_stage, y_train_stage)

    # Save stage model
    with open(os.path.join(models_dir, "stage_model.pkl"), "wb") as f:
        pickle.dump(stage_model, f)

    # 3. Generate predictions
    print("Generating predictions for World Cup 2026...")
    X_test = test_df[FEATURES]

    # Predict total goals (ensure no negative predictions)
    test_df["total_goals"] = np.clip(goals_model.predict(X_test), 0.0, None)

    # Predict stage score
    test_stage_preds = stage_model.predict(X_test)

    # Map predictions to 2026 capacity constraints
    test_mapped = run_test_capacity_mapping(test_df, test_stage_preds)

    # Create submission file
    submission = test_mapped[["ID", "total_goals", "Target"]]

    # Validate schema using SubmissionRowSchema
    from contracts.schema import SubmissionRowSchema
    for _, row in submission.iterrows():
        SubmissionRowSchema(**row.to_dict())

    # Save submission file
    os.makedirs(submissions_dir, exist_ok=True)
    sub_path = os.path.join(submissions_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved successfully to {sub_path}")


if __name__ == "__main__":
    train_and_predict("data/processed", "outputs/models", "outputs/submissions")

