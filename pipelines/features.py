"""Feature engineering pipeline for the World Cup 2026 Goal Prediction Challenge.

Extracts team-level and confederation-level historical features.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

# Mapping of debutant teams to their confederation names
DEBUTANT_CONFEDERATION = {
    "Cabo Verde": "Confederation of African Football",
    "Curacao": "Confederation of North, Central American and Caribbean Association Football",
    "Uzbekistan": "Asian Football Confederation",
    "Jordan": "Asian Football Confederation",
}

# Mapping of stages to ordinal scores
STAGE_ORDINAL = {
    "group": 0,
    "roundof32": 1,
    "roundof16": 2,
    "qf": 3,
    "sf": 4,
    "runnerup": 5,
    "champion": 6,
}


def get_team_confederation(team: str, train_df: pd.DataFrame) -> str:
    """Gets the confederation name for a team from the training set or debutant mapping."""
    if team in DEBUTANT_CONFEDERATION:
        return DEBUTANT_CONFEDERATION[team]
    match = train_df[train_df["country"] == team]
    if not match.empty:
        return match.iloc[0]["confederation_name"]
    return "Union of European Football Associations"  # Default fallback


def get_tournament_stages(train_df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
    """Maps historical stages to clean target stages."""
    train = train_df.copy()
    # Merge with standings to get exact position
    train = train.merge(
        standings_df[["tournament_id", "team_id", "position"]],
        on=["tournament_id", "team_id"],
        how="left",
    )

    def map_stage(row):
        pos = row["position"]
        stage = row["stage_reached"]
        if pos == 1:
            return "champion"
        elif pos == 2:
            return "runnerup"
        elif pos in [3, 4] or stage in ["semi-finals", "third-place match"]:
            return "sf"
        elif stage in ["quarter-finals", "second group stage"]:
            return "qf"
        elif stage == "round of 16":
            return "roundof16"
        else:
            return "group"

    train["Target"] = train.apply(map_stage, axis=1)
    train["Target_ordinal"] = train["Target"].map(STAGE_ORDINAL)
    return train


def extract_features(
    train_path: str,
    test_path: str,
    raw_dir: str,
    processed_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates features for train and test datasets.

    Args:
        train_path: Path to train_processed.csv.
        test_path: Path to test_processed.csv.
        raw_dir: Directory containing raw data files.
        processed_dir: Directory to save engineered feature data.

    Returns:
        A tuple of (train_features, test_features)
    """
    train_proc = pd.read_csv(train_path)
    test_proc = pd.read_csv(test_path)
    matches_df = pd.read_csv(os.path.join(raw_dir, "matches.csv"))
    tournaments_df = pd.read_csv(os.path.join(raw_dir, "tournaments.csv"))
    standings_df = pd.read_csv(os.path.join(raw_dir, "tournament_standings.csv"))

    # Map target variables in training set
    train_proc = get_tournament_stages(train_proc, standings_df)

    # Clean country names in matches and tournaments
    from pipelines.preprocessing import clean_countries

    matches = clean_countries(matches_df, ["home_team_name", "away_team_name", "country_name"])
    tournaments = clean_countries(tournaments_df, ["host_country"])

    # Ensure dates are datetime objects
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    tournaments["start_date"] = pd.to_datetime(tournaments["start_date"])

    # Merge tournament start dates into Train
    train_proc = train_proc.merge(
        tournaments[["tournament_id", "start_date"]], on="tournament_id", how="left"
    )

    # 2026 World Cup starts in June 2026
    test_proc["tournament_id"] = "WC-2026"
    test_proc["start_date"] = pd.to_datetime("2026-06-11")

    # Add host indicator
    # Hosts in 2026: USA, Canada, Mexico
    test_proc["is_host"] = test_proc["country"].apply(
        lambda c: 1 if c in ["United States", "Canada", "Mexico"] else 0
    )

    # Historical host mapping for training set
    train_proc["is_host"] = train_proc.apply(
        lambda r: 1 if r["country"] == r["tournament_name"].split(" ")[-4] else 0,
        axis=1,
    )
    # Correcting host status from tournaments table
    train_proc = train_proc.drop(columns=["is_host"], errors="ignore")
    train_proc = train_proc.merge(
        tournaments[["tournament_id", "host_country"]], on="tournament_id", how="left"
    )
    train_proc["is_host"] = (train_proc["country"] == train_proc["host_country"]).astype(int)
    train_proc = train_proc.drop(columns=["host_country"])

    # Combine datasets for feature extraction loop
    combined = pd.concat([train_proc, test_proc], ignore_index=True)

    # Build features team-by-team, tournament-by-tournament
    feature_records = []

    for idx, row in combined.iterrows():
        team = row["country"]
        tour_id = row["tournament_id"]
        start_date = row["start_date"]
        confed = get_team_confederation(team, train_proc)

        # 1. Filter all matches played by this team prior to the current tournament
        team_matches = matches[
            ((matches["home_team_name"] == team) | (matches["away_team_name"] == team))
            & (matches["match_date"] < start_date)
        ]

        if not team_matches.empty:
            total_matches = len(team_matches)

            # Goals scored and conceded
            goals_scored = 0
            goals_conceded = 0
            wins = 0
            clean_sheets = 0
            failed_to_score = 0

            for _, m in team_matches.iterrows():
                is_home = m["home_team_name"] == team
                if is_home:
                    s_goals = m["home_team_score"]
                    c_goals = m["away_team_score"]
                    if m["home_team_win"] == 1:
                        wins += 1
                else:
                    s_goals = m["away_team_score"]
                    c_goals = m["home_team_score"]
                    if m["away_team_win"] == 1:
                        wins += 1

                goals_scored += s_goals
                goals_conceded += c_goals
                if c_goals == 0:
                    clean_sheets += 1
                if s_goals == 0:
                    failed_to_score += 1

            goals_scored_per_match = goals_scored / total_matches
            goals_conceded_per_match = goals_conceded / total_matches
            win_rate = wins / total_matches
            clean_sheet_rate = clean_sheets / total_matches
            failed_to_score_rate = failed_to_score / total_matches
        else:
            # Debutant: fill with NaN, will impute with confederation stats later
            goals_scored_per_match = np.nan
            goals_conceded_per_match = np.nan
            win_rate = np.nan
            clean_sheet_rate = np.nan
            failed_to_score_rate = np.nan

        # 2. Get average stage reached in previous tournaments
        past_seasons = train_proc[
            (train_proc["country"] == team) & (train_proc["start_date"] < start_date)
        ]

        if not past_seasons.empty:
            avg_stage_reached = past_seasons["Target_ordinal"].mean()
        else:
            avg_stage_reached = np.nan

        # 3. Decayed features over last 3 appearances
        if not past_seasons.empty:
            past_seasons_sorted = past_seasons.sort_values(by="start_date", ascending=False)
            stages = past_seasons_sorted["Target_ordinal"].tolist()[:3]

            # Calculate goals scored/conceded in those specific tournaments
            past_goals_scored = []
            past_goals_conceded = []

            for _, ps_row in past_seasons_sorted.head(3).iterrows():
                ps_tour_id = ps_row["tournament_id"]
                # Filter matches for this team in this specific past tournament
                ps_matches = matches[
                    ((matches["home_team_name"] == team) | (matches["away_team_name"] == team))
                    & (matches["tournament_id"] == ps_tour_id)
                ]
                ps_goals_s = 0
                ps_goals_c = 0
                for _, m in ps_matches.iterrows():
                    if m["home_team_name"] == team:
                        ps_goals_s += m["home_team_score"]
                        ps_goals_c += m["away_team_score"]
                    else:
                        ps_goals_s += m["away_team_score"]
                        ps_goals_c += m["home_team_score"]
                # Standardize to per-match rate
                match_count = len(ps_matches) if len(ps_matches) > 0 else 3.0
                past_goals_scored.append(ps_goals_s / match_count)
                past_goals_conceded.append(ps_goals_c / match_count)

            # Apply decay weights (1.0, 0.5, 0.25)
            weights = [1.0, 0.5, 0.25][: len(stages)]
            norm_w = sum(weights)

            decayed_stage = sum(s * w for s, w in zip(stages, weights)) / norm_w
            decayed_gs = sum(g * w for g, w in zip(past_goals_scored, weights)) / norm_w
            decayed_gc = sum(g * w for g, w in zip(past_goals_conceded, weights)) / norm_w
        else:
            decayed_stage = np.nan
            decayed_gs = np.nan
            decayed_gc = np.nan

        # 3.5. Recent Form logic
        past_tours = combined[combined["start_date"] < start_date].sort_values(
            by="start_date", ascending=False
        )
        unique_past_dates = past_tours["start_date"].unique()

        last_tour_date = unique_past_dates[0] if len(unique_past_dates) > 0 else None
        two_tours_ago_date = unique_past_dates[1] if len(unique_past_dates) > 1 else None

        qualified_last_tournament = 0
        qualified_two_tournaments_ago = 0
        goals_scored_last_wc = np.nan
        goals_conceded_last_wc = np.nan

        if last_tour_date is not None:
            last_season = train_proc[
                (train_proc["country"] == team) & (train_proc["start_date"] == last_tour_date)
            ]
            if not last_season.empty:
                qualified_last_tournament = 1
                last_tour_id = last_season.iloc[0]["tournament_id"]
                ps_matches = matches[
                    ((matches["home_team_name"] == team) | (matches["away_team_name"] == team))
                    & (matches["tournament_id"] == last_tour_id)
                ]
                ps_goals_s = 0
                ps_goals_c = 0
                for _, m in ps_matches.iterrows():
                    if m["home_team_name"] == team:
                        ps_goals_s += m["home_team_score"]
                        ps_goals_c += m["away_team_score"]
                    else:
                        ps_goals_s += m["away_team_score"]
                        ps_goals_c += m["home_team_score"]
                match_count = len(ps_matches) if len(ps_matches) > 0 else 3.0
                goals_scored_last_wc = ps_goals_s / match_count
                goals_conceded_last_wc = ps_goals_c / match_count

        if two_tours_ago_date is not None:
            two_seasons = train_proc[
                (train_proc["country"] == team) & (train_proc["start_date"] == two_tours_ago_date)
            ]
            if not two_seasons.empty:
                qualified_two_tournaments_ago = 1

        feature_records.append(
            {
                "ID": row["ID"],
                "country": team,
                "tournament_id": tour_id,
                "confederation_name": confed,
                "start_date": start_date,
                "historical_goals_scored_per_match": goals_scored_per_match,
                "historical_goals_conceded_per_match": goals_conceded_per_match,
                "historical_win_rate": win_rate,
                "historical_avg_stage_reached": avg_stage_reached,
                "decayed_goals_scored": decayed_gs,
                "decayed_goals_conceded": decayed_gc,
                "decayed_stage_reached": decayed_stage,
                "historical_clean_sheet_rate": clean_sheet_rate,
                "historical_failed_to_score_rate": failed_to_score_rate,
                "historical_appearances": len(past_seasons) if not past_seasons.empty else 0.0,
                "recent_appearances_count": len(past_seasons.head(3))
                if not past_seasons.empty
                else 0.0,
                "qualified_last_tournament": qualified_last_tournament,
                "qualified_two_tournaments_ago": qualified_two_tournaments_ago,
                "goals_scored_last_wc": goals_scored_last_wc,
                "goals_conceded_last_wc": goals_conceded_last_wc,
            }
        )

    features_df = pd.DataFrame(feature_records)

    # 4. Confederation-level priors per tournament
    confed_stats = []
    for tour_id in combined["tournament_id"].unique():
        tour_start_date = combined[combined["tournament_id"] == tour_id]["start_date"].iloc[0]

        # Calculate confederation averages up to this point in time
        past_features = features_df[features_df["start_date"] < tour_start_date]

        for conf in features_df["confederation_name"].unique():
            conf_past = past_features[past_features["confederation_name"] == conf]

            if not conf_past.empty:
                conf_avg_gs = conf_past["historical_goals_scored_per_match"].mean()
                conf_avg_stage = conf_past["historical_avg_stage_reached"].mean()
                conf_avg_cs = conf_past["historical_clean_sheet_rate"].mean()
                conf_avg_fts = conf_past["historical_failed_to_score_rate"].mean()
            else:
                # Default baseline values for very early years
                conf_avg_gs = 1.2
                conf_avg_stage = 0.5
                conf_avg_cs = 0.25
                conf_avg_fts = 0.25

            confed_stats.append(
                {
                    "tournament_id": tour_id,
                    "confederation_name": conf,
                    "confederation_avg_goals_scored": conf_avg_gs
                    if not np.isnan(conf_avg_gs)
                    else 1.2,
                    "confederation_avg_stage_reached": conf_avg_stage
                    if not np.isnan(conf_avg_stage)
                    else 0.5,
                    "confederation_avg_clean_sheets": conf_avg_cs
                    if not np.isnan(conf_avg_cs)
                    else 0.25,
                    "confederation_avg_failed_to_score": conf_avg_fts
                    if not np.isnan(conf_avg_fts)
                    else 0.25,
                }
            )

    confed_stats_df = pd.DataFrame(confed_stats)

    # Merge features with confederation statistics
    features_df = features_df.merge(
        confed_stats_df, on=["tournament_id", "confederation_name"], how="left"
    )

    # 5. Impute team-level NaNs using confederation-level averages
    for idx, row in features_df.iterrows():
        if np.isnan(row["historical_goals_scored_per_match"]):
            features_df.loc[idx, "historical_goals_scored_per_match"] = row[
                "confederation_avg_goals_scored"
            ]
        if np.isnan(row["historical_goals_conceded_per_match"]):
            features_df.loc[idx, "historical_goals_conceded_per_match"] = (
                1.5 - row["confederation_avg_goals_scored"]
            )
        if np.isnan(row["historical_win_rate"]):
            features_df.loc[idx, "historical_win_rate"] = 0.33
        if np.isnan(row["historical_avg_stage_reached"]):
            features_df.loc[idx, "historical_avg_stage_reached"] = row[
                "confederation_avg_stage_reached"
            ]

        if np.isnan(row["historical_clean_sheet_rate"]):
            features_df.loc[idx, "historical_clean_sheet_rate"] = row[
                "confederation_avg_clean_sheets"
            ]
        if np.isnan(row["historical_failed_to_score_rate"]):
            features_df.loc[idx, "historical_failed_to_score_rate"] = row[
                "confederation_avg_failed_to_score"
            ]

        if np.isnan(row["decayed_goals_scored"]):
            features_df.loc[idx, "decayed_goals_scored"] = row["historical_goals_scored_per_match"]
        if np.isnan(row["decayed_goals_conceded"]):
            features_df.loc[idx, "decayed_goals_conceded"] = row[
                "historical_goals_conceded_per_match"
            ]
        if np.isnan(row["decayed_stage_reached"]):
            features_df.loc[idx, "decayed_stage_reached"] = row["historical_avg_stage_reached"]

        if np.isnan(row["goals_scored_last_wc"]):
            features_df.loc[idx, "goals_scored_last_wc"] = row["confederation_avg_goals_scored"]
        if np.isnan(row["goals_conceded_last_wc"]):
            features_df.loc[idx, "goals_conceded_last_wc"] = (
                1.5 - row["confederation_avg_goals_scored"]
            )

    # Separate train and test features
    train_features = features_df[features_df["tournament_id"] != "WC-2026"].copy()
    test_features = features_df[features_df["tournament_id"] == "WC-2026"].copy()

    # Merge remaining columns from processed train and test datasets
    train_features = train_proc[
        [
            "ID",
            "year",
            "matches_played",
            "total_goals",
            "Target",
            "Target_ordinal",
            "is_host",
            "elo_rating_prior",
        ]
    ].merge(train_features.drop(columns=["start_date", "confederation_name"]), on="ID", how="left")

    test_features = test_proc[["ID", "is_host", "elo_rating_prior"]].merge(
        test_features.drop(columns=["start_date", "confederation_name"]), on="ID", how="left"
    )

    # Save features to parquet
    os.makedirs(processed_dir, exist_ok=True)
    train_features.to_parquet(os.path.join(processed_dir, "train_features.parquet"), index=False)
    test_features.to_parquet(os.path.join(processed_dir, "test_features.parquet"), index=False)
    print("Feature matrices written successfully to data/processed/")

    return train_features, test_features


if __name__ == "__main__":
    extract_features(
        "data/processed/train_processed.csv",
        "data/processed/test_processed.csv",
        "data/raw",
        "data/processed",
    )
