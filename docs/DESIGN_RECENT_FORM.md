# Design Spec: Recent Form Feature Integration

This document outlines the design decisions, assumptions, and changes needed to incorporate team-level recent form features into the prediction pipeline.

## 1. Understanding Summary
* **What is being built**: Enhancements to the feature extraction pipeline in `pipelines/features.py` to capture recent team form and qualification gaps.
* **Why it exists**: To improve prediction accuracy for both goals (RMSE) and stages (Macro F1) for the 2026 World Cup.
* **Who it is for**: The World Cup 2026 Goal Prediction Challenge system.
* **Key constraints**:
  * Only historical data from the Fjelstul database is allowed (closed-data challenge).
  * No squad lists or player rosters are available for 2026 teams.
  * The most recent match data in the dataset is from the 2022 World Cup.

## 2. Assumptions
* A team's qualification status in the immediately preceding tournament (2022) is a strong proxy for their current competitive level.
* Un-decayed goals scored/conceded stats from the most recent World Cup, combined with binary qualification status, allow the tree-based model (LightGBM) to learn form changes better than arbitrary hardcoded decay weights.

## 3. Decision Log
* **Decision 1**: Exclude player-level and squad-level features.
  * *Alternatives considered*: Squad age, club appearances, international goal ratios of current squad members.
  * *Reason for choice*: The 2026 test set does not contain squad lists or roster data. Using squad-level features would cause feature mismatch (NaNs) during 2026 inference.
* **Decision 2**: Pass raw goals from the last World Cup and qualification status to LightGBM instead of tuning decay weights with Optuna.
  * *Alternatives considered*: Optimizing decay weights (e.g. $w_1, w_2, w_3$) within the Optuna search loop.
  * *Reason for choice*: Regenerating the feature table on every Optuna trial is computationally slow. By providing the raw values of the last tournament alongside binary qualification indicators, LightGBM can naturally partition and weight these features.

## 4. Final Design Specification

### A. Data Schema Changes
Update `contracts/schema.py` to include:
* `qualified_last_tournament` (int: 0 or 1)
* `qualified_two_tournaments_ago` (int: 0 or 1)
* `goals_scored_last_wc` (float)
* `goals_conceded_last_wc` (float)

### B. Feature Extraction Changes
In `pipelines/features.py`, when calculating stats for team $T$ at tournament year $Y$:
* Look at the immediately preceding World Cup (e.g. $Y-4$) and the one before that ($Y-8$).
* Set `qualified_last_tournament = 1` if the team played in that tournament, else `0`.
* Set `qualified_two_tournaments_ago = 1` if the team played in that tournament, else `0`.
* If `qualified_last_tournament == 1`, retrieve the average goals scored and conceded by the team in that tournament.
* If `qualified_last_tournament == 0` (or for debutant teams), impute goals with the confederation's historical average.

### C. Model & Training Updates
* Add the new columns to the `FEATURES` list in both `tune.py` and `train.py`.
* Ensure that the cross-validation splits (validating on 2018 and 2022) correctly calculate and feature-engineer these columns without temporal leakage.
