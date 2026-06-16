# Model Scorecard: Champion Model (LightGBM + Elo + Capacity Sorting)

This scorecard summarizes the final champion model configuration and validation metrics.

## Model Identity
*   **System Name**: worldcup_prediction
*   **Model Type**: LightGBM Regressor (Ensemble of Goals Regressor and Stage Ordinal Regressor)
*   **Target 1**: `total_goals` (RMSE)
*   **Target 2**: `Target_ordinal` (Continuous stage score sorted and partitioned to tournament capacities)

## Features Utilized
*   `elo_rating_prior`: Current Elo rating of the team prior to the tournament.
*   `is_host`: Boolean indicating if the team is one of the hosts (USA, Canada, Mexico in 2026).
*   `historical_goals_scored_per_match`: Historical average goals scored per match in previous World Cups.
*   `historical_goals_conceded_per_match`: Historical average goals conceded per match in previous World Cups.
*   `historical_win_rate`: Percentage of previous matches won by the team.
*   `historical_avg_stage_reached`: Average stage reached (mapped to ordinal scores 0 to 6) in previous appearances.
*   `confederation_avg_goals_scored`: Confederation average goals scored per match historically.
*   `confederation_avg_stage_reached`: Confederation average stage reached historically.
*   `decayed_goals_scored`: Exponentially decayed goals scored over the last 3 tournaments.
*   `decayed_goals_conceded`: Exponentially decayed goals conceded over the last 3 tournaments.
*   `decayed_stage_reached`: Exponentially decayed stage reached over the last 3 tournaments.
*   `historical_clean_sheet_rate`: Historical average clean sheets rate in previous World Cups.
*   `historical_failed_to_score_rate`: Historical average failed to score rate in previous World Cups.
*   `historical_appearances`: Total historical World Cup appearances prior to the tournament.
*   `recent_appearances_count`: Total appearances in the last 3 World Cups.

## Validation Results
Evaluated using Out-of-Time Cross-Validation (validate on 2018 and 2022).

| Target | Metric | Validation Score |
| :--- | :--- | :--- |
| **Goals Prediction (60% weight)** | RMSE | **3.4760** |
| **Stage Prediction (40% weight)** | Macro F1 | **0.3750** |

## Best Hyperparameters
Tuned using Optuna (50 trials per model).

### Goals Model Parameters
*   `learning_rate`: 0.05
*   `num_leaves`: 15
*   `max_depth`: 4
*   `min_child_samples`: 10
*   `n_estimators`: 100
*   `random_state`: 42

### Stage Model Parameters
*   `learning_rate`: 0.05
*   `num_leaves`: 15
*   `max_depth`: 4
*   `min_child_samples`: 10
*   `n_estimators`: 100
*   `random_state`: 42
