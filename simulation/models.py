"""
Pydantic models for the Medication Dosing & Toxicity Control environment.

Implements the full OpenEnv typed model specification:
  - Observation: what the agent sees each step
  - Action: what the agent can do
  - Reward: structured reward breakdown
  - State: full internal state for debugging/replay

The models capture realistic clinical parameters including
patient profiles, multi-compartment pharmacokinetics, and vitals.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class PatientProfile(BaseModel):
    """Patient characteristics that affect drug pharmacokinetics."""
    weight_kg: float = Field(default=70.0, description="Patient weight in kg")
    age: int = Field(default=45, description="Patient age in years")
    renal_function: float = Field(
        default=1.0, ge=0.2, le=1.5,
        description="Renal clearance multiplier (1.0=normal, <1=impaired, >1=enhanced)"
    )
    hepatic_function: float = Field(
        default=1.0, ge=0.3, le=1.5,
        description="Hepatic metabolism multiplier"
    )
    drug_sensitivity: float = Field(
        default=1.0, ge=0.5, le=2.0,
        description="Individual drug sensitivity (1.0=normal)"
    )


class ClinicalEvent(str, Enum):
    """Clinical events that can occur mid-episode."""
    NONE = "none"
    FEVER_SPIKE = "fever_spike"               # increases metabolism
    RENAL_DECLINE = "renal_decline"            # decreases clearance
    DRUG_INTERACTION = "drug_interaction"      # amplifies concentration
    VOMITING = "vomiting"                      # partial dose loss
    FLUID_SHIFT = "fluid_shift"               # dilutes concentration


class Observation(BaseModel):
    """
    Patient observation at a given timestep.

    Provides the agent with clinically relevant information including
    drug levels, metabolism indicators, patient vitals, and alerts.
    """
    step: int = Field(default=0, description="Current timestep")
    concentration: float = Field(default=0.0, description="Primary drug blood concentration")
    secondary_concentration: float = Field(default=0.0, description="Secondary drug blood concentration")
    metabolism_rate: float = Field(default=0.15, description="Current effective metabolism rate")
    last_dose: float = Field(default=0.0, description="Previous dose administered")
    toxicity_flag: bool = Field(default=False, description="True if concentration > risky threshold")
    patient_weight: float = Field(default=70.0, description="Patient weight in kg")
    renal_function: float = Field(default=1.0, description="Current renal clearance multiplier")
    heart_rate: float = Field(default=72.0, description="Heart rate (BPM), affected by drug levels")
    clinical_event: str = Field(default="none", description="Active clinical event, if any")
    time_in_therapeutic: int = Field(default=0, description="Consecutive steps in therapeutic window")
    concentration_trend: float = Field(default=0.0, description="Rate of concentration change vs previous step")


class Action(BaseModel):
    """
    Agent action: how much drug to administer.

    The agent chooses a dose between 0 and 20 units.
    Doses are automatically adjusted for patient weight internally.
    """
    dose: float = Field(ge=0.0, le=20.0, description="Drug dose to administer (0-20 units)")


class Reward(BaseModel):
    """
    Structured reward signal returned at each step.

    The reward combines multiple clinical objectives:
      - efficacy: keeping concentration in the therapeutic window
      - safety: avoiding toxicity
      - stability: maintaining consistent levels (low variance)
      - responsiveness: quickly reaching therapeutic range
    """
    value: float = Field(description="Total scalar reward")
    efficacy: float = Field(default=0.0, description="Reward for being in therapeutic window")
    safety: float = Field(default=0.0, description="Penalty for toxic/dangerous levels")
    stability: float = Field(default=0.0, description="Bonus for consistent concentration")
    shaping: float = Field(default=0.0, description="Continuous distance-from-target signal")
    in_therapeutic: bool = Field(default=False, description="Whether in therapeutic window")
    is_toxic: bool = Field(default=False, description="Whether above toxic threshold")


class State(BaseModel):
    """Full internal environment state for OpenEnv spec compliance."""
    step: int = 0
    concentration: float = 0.0
    drug2_concentration: float = 0.0
    metabolism_rate: float = 0.15
    last_dose: float = 0.0
    toxicity_flag: bool = False
    max_steps: int = 10
    done: bool = False
    heart_rate: float = 72.0
    renal_function: float = 1.0
    clinical_event: str = "none"
    concentration_history: list[float] = []
    time_in_therapeutic: int = 0