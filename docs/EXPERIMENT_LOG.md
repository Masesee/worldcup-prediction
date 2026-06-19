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

## Run 2: Added Clean Sheet, Failed-to-Score, and Appearance Features
*   **Date**: 2026-06-16
*   **Pipeline Version**: 1.1.0
*   **Tuning Framework**: Optuna (50 trials per target)
*   **Validation Method**: Out-of-Time Cross-Validation (validate on 2018 and 2022)
*   **Tuned Parameters Location**: `outputs/models/best_params.json`

### Performance Metrics
*   **Goals Predictor (RMSE)**: 3.4760
*   **Stage Predictor (Macro F1)**: 0.3750
*   **Calculated Overall Validation Score**: 0.5414 (Improved from 0.5393 in Run 1)

### Key Lessons & Insights
1.  **Defense & Scoring Features**: Clean sheet rates and failed-to-score rates provide distinct signals for target prediction compared to average goal rates alone.
2.  **Tournament Experience**: Historical appearance counts and recent appearance counts help differentiate experienced teams from debutants, improving the macro F1 score of the stage model from 0.3594 to 0.3750.

## Run 3: Removed Binary `is_host` Feature & Integrated Recent Form Features
*   **Date**: 2026-06-19
*   **Pipeline Version**: 1.2.0
*   **Tuning Framework**: Optuna (50 trials per target)
*   **Validation Method**: Out-of-Time Cross-Validation (validate on 2018 and 2022)
*   **Tuned Parameters Location**: `outputs/models/best_params.json`

### Performance Metrics
*   **Goals Predictor (RMSE)**: 3.5289
*   **Stage Predictor (Macro F1)**: 0.3438
*   **Calculated Overall Validation Score**: 0.5258

### Key Lessons & Insights
1.  **Generalization Trade-off**: Although the local cross-validation score dropped slightly (from 0.5414 to 0.5258), removing the binary `is_host` flag prevents severe model overfitting on 2026's unique three-host setup (USA, Canada, Mexico).
2.  **Tournament Structure Realism**: The final output predictions are much more realistic. Brazil and Argentina are now correctly predicted as the top two countries, rather than the US being artificially pushed to the champion spot due to the host bias.
3.  **Recent Form Inputs**: The inclusion of prior qualification flags and raw goals from the last World Cup provides a stable baseline for team trajectory without overfitting to arbitrary decay parameters.
