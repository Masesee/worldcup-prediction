"""Data contracts and schemas for the World Cup 2026 Goal Prediction Challenge.

This module defines the schema validation rules, column names, and type checks
for all input, intermediate, and output datasets in the project.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Constants for Stage Names
VALID_STAGES = [
    "group",
    "roundof32",
    "roundof16",
    "qf",
    "sf",
    "runnerup",
    "champion",
]

# Mapping of historical stage_reached to target stages
HISTORICAL_STAGE_MAPPING = {
    "group stage": "group",
    "round of 16": "roundof16",
    "quarter-finals": "qf",
    "second group stage": "qf",  # 2nd group stage is equivalent to quarter-finals
    "semi-finals": "sf",
    "third-place match": "sf",
    "final": None,  # Handled using standings (champion/runnerup)
    "final round": None,  # Handled using standings (1950 group stage final)
}


class TrainRowSchema(BaseModel):
    """Schema for a single row in the Train.csv file."""

    ID: str = Field(..., description="Unique team-tournament identifier")
    team_id: str = Field(..., description="Fjelstul database unique team ID")
    country: str = Field(..., description="Name of the country")
    team_code: str = Field(..., description="Three-letter ISO code or similar")
    confederation_name: str = Field(..., description="Confederation name (e.g. UEFA)")
    region_name: str = Field(..., description="Sub-continental region")
    tournament_id: str = Field(..., description="Unique tournament ID (e.g. WC-1930)")
    tournament_name: str = Field(..., description="Name of the tournament")
    year: int = Field(..., ge=1930, le=2022, description="Year of the tournament")
    matches_played: int = Field(..., ge=1, le=10, description="Matches played by the team")
    total_goals: int = Field(..., ge=0, description="Total goals scored by the team")
    stage_reached: str = Field(..., description="Historical stage reached")


class InferenceRowSchema(BaseModel):
    """Schema for a single row in the Test.csv file."""

    ID: str = Field(..., description="Unique team-tournament identifier (e.g. WC-2026_AUT)")
    country: str = Field(..., description="Name of the country")


class SubmissionRowSchema(BaseModel):
    """Schema for a single row in the submission file."""

    ID: str = Field(..., description="Unique team-tournament identifier (e.g. WC-2026_AUT)")
    total_goals: float = Field(..., ge=0.0, description="Predicted total goals scored")
    Target: Literal["group", "roundof32", "roundof16", "qf", "sf", "runnerup", "champion"] = Field(
        ..., description="Predicted finishing stage of the team"
    )


class TeamFeatureSchema(BaseModel):
    """Schema for intermediate engineered features for a team-tournament row."""

    ID: str
    country: str
    year: int
    is_host: int = Field(..., ge=0, le=1)
    elo_rating_prior: float
    historical_goals_scored_per_match: float
    historical_goals_conceded_per_match: float
    historical_win_rate: float
    historical_avg_stage_reached: float
    confederation_avg_goals_scored: float
    confederation_avg_stage_reached: float
    decayed_goals_scored: float
    decayed_goals_conceded: float
    decayed_stage_reached: float
