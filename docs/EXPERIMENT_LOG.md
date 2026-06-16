# Experiment Log

This file records all tuning and validation iterations of the World Cup 2026 Goal Prediction system.

## Run 1: LightGBM Baseline with Optuna Tuning
*   **Date**: 2026-06-16
*   **Pipeline Version**: 1.0.0
*   **Tuning Framework**: Optuna (50 trials per target)
*   **Validation Method**: Out-of-Time Cross-Validation (validate on 2018 and 2022)
*   **Tuned Parameters Location**: `outputs/models/best_params.json`

### Performance Metrics
*   **Goals Predictor (RMSE)**: 3.4069
*   **Stage Predictor (Macro F1)**: 0.3594

### Key Lessons & Insights
1.  **Elo Rating Prior**: Integrating a match-by-match historical Elo rating starting from 1930 provides a strong indicator of team quality prior to each tournament.
2.  **Home Advantage**: Adjusting the Elo calculation for host countries (adding +100 rating points to the host's Elo during the match expectation phase) improves alignment with historical outcomes.
3.  **Capacity Mapping**: Structural capacity mapping successfully guarantees valid tournament outputs for the stage prediction task. This avoids the problem where standard models would fail to predict `roundof32` since there were 0 instances of it in the historical training set.
4.  **Debutant Imputation**: Using confederation-level averages to impute prior stats for debutants (Jordan, Uzbekistan, Cabo Verde, Curacao) prevented NaN propagation and allowed valid predictions for these countries.
