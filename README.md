# FIFA World Cup 2026 Goal & Progression Predictor

This repository contains a production-grade machine learning system designed to predict team goal scoring and tournament progression for the FIFA World Cup 2026. This is developed for the community challenge on Zindi.

## 1. Challenge & Prediction Targets

The prediction objective consists of two tasks:
1. **Total Goals**: Predict the total number of goals each team will score during the 2026 World Cup (excluding penalty shootouts).
   - *Evaluation Metric*: Root Mean Squared Error (RMSE).
   - *Leaderboard Weight*: 60%.
2. **Tournament Stage**: Predict the exact stage at which each team will be eliminated or finish the tournament.
   - *Stages*: `group`, `roundof32`, `roundof16`, `qf`, `sf`, `runnerup`, `champion`.
   - *Evaluation Metric*: Macro-averaged F1 Score.
   - *Leaderboard Weight*: 40%.

The overall evaluation score is computed as:
$$\text{Overall Score} = 0.40 \times \text{Stage\_F1} + 0.60 \times \left(1 - \frac{\text{Goals\_RMSE}}{10}\right)$$

---

## 2. Directory Layout & Architecture

The repository is structured as a modular and parallel-friendly ML pipeline:

```
worldcup_prediction/ (Repository Root)
├── .gitignore                     # Git ignore rules for data, models, and virtual environments
├── .pre-commit-config.yaml        # Pre-commit configuration for style and quality checks
├── pyproject.toml                 # Package configuration, metadata, and tool configurations
├── requirements.txt               # Pinpoint dependencies for the execution environment
├── README.md                      # This main project documentation file
│
├── contracts/                     # Pydantic schemas validating data integrity
│   ├── __init__.py
│   └── schema.py                  # Definitions of input, feature, and output schemas
│
├── data/                          # Shared data directory
│   ├── raw/                       # Unmodified input files (Train.csv, Test.csv, matches.csv, etc.)
│   └── processed/                 # Cached feature tables and preprocessed files
│
├── docs/                          # Detailed design and performance reports
│   ├── EXPERIMENT_LOG.md          # Chronological log of validation scores and learnings
│   ├── MODEL_SCORECARD.md         # Final model identity, parameters, and features list
│   └── PROJECT_CONTEXT.md         # In-depth architectural context and domain rules
│
├── experiments/                   # Logging output directory
│   └── logs/                      # Optuna trials and database files
│
├── outputs/                       # Artifacts generated during execution
│   ├── eda/                       # Exploratory charts and visual analyses
│   ├── models/                    # Serialized model weights (.pkl) and best parameters
│   └── submissions/               # Prepared CSV submissions ready for upload
│
└── pipelines/                     # Source code modules for preprocessing, features, and training
    ├── __init__.py
    ├── preprocessing.py           # Entity normalization and match-by-match Elo simulator
    ├── features.py                # Feature extraction, decay updates, and imputation
    └── training/                  # Model tuning and training scripts
        ├── __init__.py
        ├── train.py               # Main model training and capacity prediction execution
        └── tune.py                # Optuna hyperparameter optimization script
```

---

## 3. Data Validation Contracts

Data integrity is enforced across all pipeline stages using Pydantic schemas defined in [contracts/schema.py](contracts/schema.py):

* [TrainRowSchema](contracts/schema.py#L34): Validates columns and types of the raw training data.
* [InferenceRowSchema](contracts/schema.py#L51): Enforces formatting of the raw testing features.
* [TeamFeatureSchema](contracts/schema.py#L68): Defines the types and constraints for the engineered features.
* [SubmissionRowSchema](contracts/schema.py#L58): Validates that the generated submission matches the exact format and constraints requested by Zindi.

---

## 4. Feature Engineering Pipeline

Features are built dynamically using only historical matches and standings. Key features in the final matrix include:

* **Prior Elo Rating**: Match-by-match Elo rating of the team prior to the tournament starting. Calculated using all historical Men's matches starting from 1930 via [run_elo_simulation](pipelines/preprocessing.py#L28).
* **Host Status Flag**: A binary indicator representing whether the team is hosting the tournament (USA, Canada, Mexico in 2026), granting a +100 point temporary rating boost to capture home field advantage.
* **Historical Team Metrics**: Lifetime average goals scored per match, goals conceded per match, win rate, and average stage reached.
* **Confederation Priors**: Historical average goals scored and stages reached grouped by confederation. These stats are mapped to debutants (Jordan, Uzbekistan, Cabo Verde, Curacao) to prevent empty feature values.
* **Exponentially Decayed Form**: Decayed goals scored, goals conceded, and stage reached over the last three World Cup appearances (using decay weights of 1.0, 0.5, and 0.25).

The feature extraction code is defined in [extract_features](pipelines/features.py#L74).

---

## 5. Machine Learning Models

The solution utilizes two specialized LightGBM Regressors optimized with Optuna:

1. **Goals Model**: Learns the relationship between team quality features and the total goals scored during a tournament.
2. **Stage Model**: Fits a continuous expected tournament stage ranking target (ordinal mapping of stages from 0 to 6).
   - **Tournament Capacity Mapping**: Traditional multiclass classification models fail to predict new stages (like `roundof32` in the 2026 expansion) because they do not exist in historical data. To resolve this, we rank all 48 test teams by their predicted ordinal stage score and partition them to match the exact physical structure of the tournament:
     - Rank 1: `champion` (1 team)
     - Rank 2: `runnerup` (1 team)
     - Ranks 3-4: `sf` (2 teams)
     - Ranks 5-8: `qf` (4 teams)
     - Ranks 9-16: `roundof16` (8 teams)
     - Ranks 17-32: `roundof32` (16 teams)
     - Ranks 33-48: `group` (16 teams)

The final execution pipeline is implemented in [train_and_predict](pipelines/training/train.py#L88).

---

## 6. Validation & Tuning Strategy

* **Out-of-Time Cross-Validation**: To prevent temporal data leakage, models are validated on out-of-time folds.
  - Fold 1: Train on data pre-2018, validate on the 2018 World Cup.
  - Fold 2: Train on data pre-2022, validate on the 2022 World Cup.
* **Hyperparameter Optimization**: Run through [tune_pipelines](pipelines/training/tune.py#L130) using Optuna (50 trials per target) to find the best configuration minimizing goals RMSE and maximizing stage F1 score.

---

## 7. Execution Workflow Walkthrough

All execution commands should be run from the root of the repository (`worldcup_prediction` directory).

### Step 1: Initialize Setup and Dependencies
Install the required packages and configure the local quality verification hooks:
```bash
pip install -r requirements.txt
pre-commit install
```

### Step 2: Clean and Preprocess Raw Data
Run the entity cleaning and match-by-match Elo simulator:
```bash
python -m pipelines.preprocessing
```

### Step 3: Extract Features
Calculate goals scored/conceded rates, win rates, and decayed form stats:
```bash
python -m pipelines.features
```

### Step 4: Hyperparameter Optimization
Run Optuna optimization trials:
```bash
python -m pipelines.training.tune
```

### Step 5: Train and Predict
Train final LightGBM models and generate 2026 predictions under 48-team capacity constraints:
```bash
python -m pipelines.training.train
```

---

## 8. Development Verification

### Running Tests
Verify project logic using pytest:
```bash
pytest
```

### Linting and Styling
Run code quality checks manually across all files:
```bash
pre-commit run --all-files
```
