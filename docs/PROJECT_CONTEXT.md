# Project Context: World Cup 2026 Goal Prediction Challenge

This document provides all details necessary for any developer, agent, or automated system to understand, execute, and extend this project.

## 1. Project Overview

The goal of this challenge is to predict two distinct outcomes for the 2026 FIFA World Cup:
1.  **Total Goals**: The total number of goals each team will score during the 2026 World Cup (excluding penalty shootouts). Evaluation metric: Root Mean Squared Error (RMSE).
2.  **Tournament Stage**: The exact stage at which each team finishes the tournament. Evaluation metric: F1 Score.
    *   Valid stages: `group`, `roundof32`, `roundof16`, `qf`, `sf`, `runnerup`, `champion`.

The overall leaderboard score is computed as:
$$\text{Overall Score} = 0.60 \times \text{RMSE(Goals)} + 0.40 \times \text{F1(Stage)}$$

This is a **closed-data competition**. Only historical World Cup data from the Fjelstul World Cup Database is permitted.

---

## 2. Directory and Architecture Design

All project code, tests, and configuration reside in the [worldcup_prediction/](worldcup_prediction/) directory.

```
worldcup_prediction/
│
├── .pre-commit-config.yaml       # Pre-commit configuration for linters and formatters
├── PROJECT_CONTEXT.md            # This project context document
│
├── contracts/                    # Data contracts and schema checks
│   ├── __init__.py
│   └── schema.py                 # Pydantic schemas for input, intermediate, and output data
│
├── data/
│   ├── raw/                      # Unmodified input files
│   └── processed/                # Preprocessed feature matrices (cached parquet)
│
├── docs/
│   ├── EXPERIMENT_LOG.md         # Records of all validation runs and learnings
│   └── MODEL_SCORECARD.md        # Summary of the champion model configuration and metrics
│
├── experiments/
│   └── logs/                     # SQLite Optuna database and tuning logs
│
├── outputs/
│   ├── eda/                      # Analysis plots and summaries
│   ├── features/                 # Feature importances and metadata
│   ├── models/                   # Best hyperparameter JSONs and trained model weights
│   ├── evaluation/               # Validation performance summaries
│   └── submissions/              # Final submission files
│
└── pipelines/                    # Processing and modeling pipelines
    ├── __init__.py
    ├── preprocessing.py          # Entity cleaning and Elo computation
    ├── features.py               # Feature generation and aggregation
    ├── training/                 # Model training and tuning scripts
    │   ├── train.py
    │   └── tune.py
    └── evaluation.py             # Validation metrics and CV runner
```

---

## 3. Core Logic & Implementation Walkthrough

### A. Entity Mapping & Data Cleaning
*   **Historical Mapping**: The training set contains historical names that do not match the 2026 test set. The preprocessing code maps:
    *   `Czechoslovakia` to `Czechia`
    *   `Turkey` to `Turkiye`
    *   `Ivory Coast` to `Cote d'Ivoire`
    *   `Zaire` to `DR Congo`
*   **Debutant Baseline Handling**: Teams with no history in the Fjelstul database (Cabo Verde, Curacao, Uzbekistan, Jordan) are assigned their respective confederation's historical average stats to avoid empty feature sets.

### B. Feature Engineering Pipeline
*   **Elo Rating**: A match-by-match Elo simulator runs from 1930 to 2022. It uses only Men's World Cup matches to calculate team ratings before each tournament.
*   **Historical Stats**: We calculate goals scored per match, goals conceded per match, and historical win rates.
*   **Time Decay**: Recent performance (e.g. 2018, 2022) is weighted higher using an exponential decay factor.
*   **Host Advantage**: The host teams (USA, Mexico, Canada in 2026) receive a host flag to account for home advantage.

### C. Model Training & Stage Mapping
*   **Goals Predictor**: A gradient boosting regressor predicts the expected total goals for each team.
*   **Stage Predictor (Structural Capacity Mapping)**:
    *   Instead of predicting stage categories directly (which fails to predict `roundof32`), we train a regression model to predict the expected ordinal stage reached.
    *   We sort the 48 teams in 2026 by their predicted strength score.
    *   We partition the teams according to the exact capacity of the 2026 World Cup tournament:
        *   Rank 1: `champion` (1 team)
        *   Rank 2: `runnerup` (1 team)
        *   Ranks 3-4: `sf` (2 teams)
        *   Ranks 5-8: `qf` (4 teams)
        *   Ranks 9-16: `roundof16` (8 teams)
        *   Ranks 17-32: `roundof32` (16 teams)
        *   Ranks 33-48: `group` (16 teams)

---

## 4. Execution Guide

### Developer Setup
1.  Install dependencies:
    ```bash
    pip install pandas pydantic scikit-learn lightgbm xgboost optuna pytest pre-commit
    ```
2.  Install pre-commit hooks:
    ```bash
    pre-commit install
    ```

### Running Tests
Execute the pytest suite to check logic correctness:
```bash
pytest worldcup_prediction/tests/
```

### Running Pre-commit Hooks manually
Run style and typing checks over all files:
```bash
pre-commit run --all-files
```
