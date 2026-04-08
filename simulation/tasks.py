"""
Task configurations for the Medication Dosing environment.

Three difficulty levels that model distinct clinical scenarios:

  Easy:   Stable adult patient, standard ward monitoring.
          Fixed metabolism, low noise. A basic controller succeeds.

  Medium: Post-surgical patient with variable metabolism.
          Requires adaptation to changing clearance rates.
          Moderate noise simulates real lab measurement error.

  Hard:   ICU critical patient with acute complications.
          High metabolism variance, clinical events (fever, renal decline,
          vomiting), metabolism shifts, strong drug interactions, heavier
          patient. Genuinely challenges frontier models.
"""

from .environment import MedicationEnv
from .models import PatientProfile

# ─────────────────────────────────────────────────────────────
# Patient archetypes for each difficulty
# ─────────────────────────────────────────────────────────────

STABLE_PATIENT = PatientProfile(
    weight_kg=70.0,
    age=45,
    renal_function=1.0,
    hepatic_function=1.0,
    drug_sensitivity=1.0,
)

VARIABLE_PATIENT = PatientProfile(
    weight_kg=82.0,
    age=62,
    renal_function=0.85,      # mildly impaired kidneys
    hepatic_function=0.9,     # slightly reduced liver function
    drug_sensitivity=1.1,     # slightly more sensitive
)

CRITICAL_PATIENT = PatientProfile(
    weight_kg=95.0,
    age=71,
    renal_function=0.65,      # significantly impaired kidneys
    hepatic_function=0.75,    # reduced liver function
    drug_sensitivity=1.3,     # hypersensitive to drug
)

# ─────────────────────────────────────────────────────────────
# Task configurations
# ─────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "max_steps": 10,
        "metabolism_base": 0.15,
        "metabolism_variance": 0.0,
        "noise_scale": 0.5,
        "metabolism_shift": False,
        "interaction_strength": 0.3,
        "patient_profile": STABLE_PATIENT,
        "clinical_events": False,
        "event_probability": 0.0,
    },
    "medium": {
        "max_steps": 15,
        "metabolism_base": 0.18,
        "metabolism_variance": 0.05,
        "noise_scale": 1.0,
        "metabolism_shift": False,
        "interaction_strength": 0.35,
        "patient_profile": VARIABLE_PATIENT,
        "clinical_events": False,
        "event_probability": 0.0,
    },
    "hard": {
        "max_steps": 20,
        "metabolism_base": 0.22,
        "metabolism_variance": 0.10,
        "noise_scale": 2.5,
        "metabolism_shift": True,
        "interaction_strength": 0.45,
        "patient_profile": CRITICAL_PATIENT,
        "clinical_events": True,
        "event_probability": 0.2,
    },
}


def get_task(task_name: str = "easy") -> MedicationEnv:
    """
    Create a MedicationEnv configured for the given difficulty.

    Args:
        task_name: one of 'easy', 'medium', 'hard'

    Returns:
        Configured MedicationEnv instance

    Raises:
        ValueError: if task_name is not recognized
    """
    if task_name not in TASK_CONFIGS:
        valid = ", ".join(TASK_CONFIGS.keys())
        raise ValueError(f"Invalid task name '{task_name}'. Choose from: {valid}")

    return MedicationEnv(**TASK_CONFIGS[task_name])