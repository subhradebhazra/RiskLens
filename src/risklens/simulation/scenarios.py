"""Validated scenario schemas and deterministic scenario vectors."""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from risklens.risk.stress import DEFAULT_SCENARIOS

class StressConfig(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    market_shock: float = Field(ge=-0.95,le=2.0)
    volatility_multiplier: float = Field(default=1.0,ge=0.1,le=10.0)
    correlation_target: float = Field(default=0.5,ge=-0.99,le=0.99)
    jump_intensity_multiplier: float = Field(default=1.0,ge=0.0,le=20.0)
    @field_validator("name")
    @classmethod
    def allowed_chars(cls,v):
        if not all(ch.isalnum() or ch in " _-/" for ch in v): raise ValueError("Scenario name contains unsupported characters.")
        return v

def deterministic_scenario_vectors():
    return dict(DEFAULT_SCENARIOS)
