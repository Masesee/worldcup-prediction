"""Hyperparameter tuning pipeline using Optuna.

Optimizes LightGBM models for goals and stage prediction tasks.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

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
    "qualified_last_tournament",
    "qualified_two_tournaments_ago",
    "goals_scored_last_wc",
    "goals_conceded_last_wc",
]

# Disable Optuna logging output to keep console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_validation_capacity_mapping(val_df: pd.DataFrame, val_preds: np.ndarray) -> pd.DataFrame:
    """Performs tournament capacity mapping on validation predictions.

    Validates against 32-team tournament format (2018/2022).
    """
    val_df = val_df.copy()
    val_df["pred_score"] = val_preds

    # Sort validation teams by predicted score descending
    val_df = val_df.sort_values(by="pred_score", ascending=False).reset_index(drop=True)

    # 32-team capacities
    # Rank 1: champion
    # Rank 2: runnerup
    # Ranks 3-4: sf
    # Ranks 5-8: qf
    # Ranks 9-16: roundof16
    # Ranks 17-32: group
    mapped_stages = []
    for i in range(len(val_df)):
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
        else:
            mapped_stages.append("group")

    val_df["Mapped_Target"] = mapped_stages
    return val_df


def evaluate_goals_model(params: dict, df: pd.DataFrame) -> float:
    """Evaluates goals model using Out-of-Time Cross Validation."""
    rmses = []

    # Fold 1: Train on < 2018, validate on 2018
    train_f1 = df[df["year"] < 2018]
    val_f1 = df[df["year"] == 2018]

    # Fold 2: Train on < 2022, validate on 2022
    train_f2 = df[df["year"] < 2022]
    val_f2 = df[df["year"] == 2022]

    for train, val in [(train_f1, val_f1), (train_f2, val_f2)]:
        if train.empty or val.empty:
            continue
        x_train, y_train = train[FEATURES], train["total_goals"]
        x_val, y_val = val[FEATURES], val["total_goals"]

        model = lgb.LGBMRegressor(**params, verbose=-1)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)

        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else 999.0


def evaluate_stage_model(params: dict, df: pd.DataFrame) -> float:
    """Evaluates stage model using Out-of-Time Cross Validation."""
    f1s = []

    train_f1 = df[df["year"] < 2018]
    val_f1 = df[df["year"] == 2018]

    train_f2 = df[df["year"] < 2022]
    val_f2 = df[df["year"] == 2022]

    for train, val in [(train_f1, val_f1), (train_f2, val_f2)]:
        if train.empty or val.empty:
            continue
        x_train, y_train = train[FEATURES], train["Target_ordinal"]
        x_val = val[FEATURES]

        model = lgb.LGBMRegressor(**params, verbose=-1)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)

        # Map predictions to 32-team structure
        val_mapped = run_validation_capacity_mapping(val, preds)

        # Compute F1 score (macro)
        f1 = f1_score(val_mapped["Target"], val_mapped["Mapped_Target"], average="macro")
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def tune_pipelines(processed_dir: str, outputs_dir: str) -> None:
    """Runs hyperparameter tuning and saves best parameters to JSON."""
    train_features_path = os.path.join(processed_dir, "train_features.parquet")
    if not os.path.exists(train_features_path):
        print(f"Error: Features file not found at {train_features_path}")
        return

    df = pd.read_parquet(train_features_path)

    # 1. Optimize Goals Model (minimize RMSE)
    print("Tuning goals prediction model...")

    def goals_objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "random_state": 42,
        }
        return evaluate_goals_model(params, df)

    goals_study = optuna.create_study(direction="minimize")
    goals_study.optimize(goals_objective, n_trials=50)
    best_goals_params = goals_study.best_params
    print(f"Best Goals RMSE: {goals_study.best_value:.4f}")

    # 2. Optimize Stage Model (maximize macro F1)
    print("Tuning tournament stage prediction model...")

    def stage_objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "random_state": 42,
        }
        return evaluate_stage_model(params, df)

    stage_study = optuna.create_study(direction="maximize")
    stage_study.optimize(stage_objective, n_trials=50)
    best_stage_params = stage_study.best_params
    print(f"Best Stage F1 (Macro): {stage_study.best_value:.4f}")

    # Save to best_params.json
    best_params = {"goals": best_goals_params, "stage": best_stage_params}

    os.makedirs(outputs_dir, exist_ok=True)
    with open(os.path.join(outputs_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    print("Saved best parameters to outputs/models/best_params.json")


if __name__ == "__main__":
    tune_pipelines("data/processed", "outputs/models")
