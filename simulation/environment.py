"""
Medication Dosing Environment — Two-Compartment Pharmacokinetic Simulation.

This environment simulates realistic medication dosing with:
  - Two-compartment pharmacokinetic model (central + peripheral)
  - Patient profiles with varying physiology (weight, renal/hepatic function)
  - Clinical events that disrupt steady state (fever, renal decline, vomiting)
  - Heart rate as a secondary vital sign affected by drug levels
  - Multi-objective reward (efficacy + safety + stability + shaping)

The agent must maintain drug concentration within a therapeutic window
(10–50) while avoiding toxicity (>70) and adapting to patient variability.

This models a real clinical workflow: an ICU nurse or automated infusion
pump adjusting drug dosing based on lab values and patient vitals.
"""

import random
import math
from .models import (
    Observation, Action, Reward, State,
    PatientProfile, ClinicalEvent,
)


class MedicationEnv:
    """
    Two-compartment pharmacokinetic medication dosing environment.

    Compartment model:
        Central (blood):    where drug is measured and acts
        Peripheral (tissue): drug reservoir that slowly exchanges with blood

    The agent observes blood concentration and must decide dosing.
    Hidden complexity comes from:
        - Peripheral compartment buffering (delayed effects)
        - Patient-specific metabolism
        - Clinical events that disrupt pharmacokinetics
        - Heart rate coupling to drug concentration
    """

    THERAPEUTIC_LOW = 10.0
    THERAPEUTIC_HIGH = 50.0
    RISKY_HIGH = 70.0
    TOXIC_THRESHOLD = 80.0
    TARGET = 30.0
    NORMAL_HR = 72.0

    def __init__(
        self,
        max_steps: int = 10,
        metabolism_base: float = 0.15,
        metabolism_variance: float = 0.0,
        noise_scale: float = 1.0,
        metabolism_shift: bool = False,
        interaction_strength: float = 0.3,
        patient_profile: PatientProfile | None = None,
        clinical_events: bool = False,
        event_probability: float = 0.0,
    ):
        self.max_steps = max_steps
        self.metabolism_base = metabolism_base
        self.metabolism_variance = metabolism_variance
        self.noise_scale = noise_scale
        self.metabolism_shift = metabolism_shift
        self.interaction_strength = interaction_strength
        self.patient_profile = patient_profile or PatientProfile()
        self.clinical_events_enabled = clinical_events
        self.event_probability = event_probability

        # Internal state — initialized in reset()
        self.concentration = 0.0
        self.peripheral_concentration = 0.0
        self.drug2_concentration = 0.0
        self.step_count = 0
        self.last_dose = 0.0
        self.current_metabolism = metabolism_base
        self.toxicity_flag = False
        self.heart_rate = self.NORMAL_HR
        self.current_event = ClinicalEvent.NONE
        self.concentration_history: list[float] = []
        self.time_in_therapeutic = 0
        self._prev_concentration = 0.0

        # Metabolism shift internals
        self._shift_step = 0
        self._shift_direction = 1.0

    def reset(self) -> Observation:
        """Reset the environment to a clean initial state."""
        self.concentration = 0.0
        self.peripheral_concentration = 0.0
        self.drug2_concentration = 0.0
        self.step_count = 0
        self.last_dose = 0.0
        self.current_metabolism = self.metabolism_base
        self.toxicity_flag = False
        self.heart_rate = self.NORMAL_HR + random.uniform(-3, 3)
        self.current_event = ClinicalEvent.NONE
        self.concentration_history = []
        self.time_in_therapeutic = 0
        self._prev_concentration = 0.0

        # Apply patient profile to metabolism
        self.current_metabolism *= self.patient_profile.renal_function
        self.current_metabolism *= self.patient_profile.hepatic_function

        # Randomize metabolism shift timing
        if self.metabolism_shift:
            self._shift_step = random.randint(
                self.max_steps // 3, 2 * self.max_steps // 3
            )
            self._shift_direction = random.choice([-1.0, 1.0])

        return self._get_obs()

    def step(self, action: Action):
        """
        Execute one timestep of the pharmacokinetic simulation.

        Process:
            1. Apply clinical event (if any)
            2. Absorb dose into central compartment
            3. Two-compartment exchange (central ↔ peripheral)
            4. Secondary drug interaction
            5. Metabolism/clearance
            6. Add patient variability noise
            7. Update vitals (heart rate)
            8. Compute reward
            9. Check termination

        Returns:
            (observation, reward_value, done, info)
        """
        self.step_count += 1
        self._prev_concentration = self.concentration
        dose = max(0.0, min(20.0, action.dose))
        self.last_dose = dose

        # === 1. Clinical event generation ===
        self._generate_clinical_event()
        event_modifier = self._apply_clinical_event(dose)
        dose = event_modifier["effective_dose"]

        # === 2. Variable metabolism ===
        self.current_metabolism = (
            self.metabolism_base
            * self.patient_profile.renal_function
            * self.patient_profile.hepatic_function
            + random.uniform(-self.metabolism_variance, self.metabolism_variance)
        )

        # Metabolism shift mid-episode
        if self.metabolism_shift and self.step_count == self._shift_step:
            shift_amount = 0.12 * self._shift_direction
            self.current_metabolism += shift_amount

        self.current_metabolism = max(0.05, min(0.5, self.current_metabolism))

        # === 3. Drug absorption (weight-adjusted) ===
        weight_factor = 70.0 / self.patient_profile.weight_kg
        sensitivity = self.patient_profile.drug_sensitivity
        self.concentration += dose * weight_factor * sensitivity

        # === 4. Two-compartment exchange ===
        # Transfer rate between central and peripheral compartments
        transfer_rate = 0.08
        transfer = transfer_rate * (self.concentration - self.peripheral_concentration)
        self.concentration -= transfer
        self.peripheral_concentration += transfer

        # === 5. Secondary drug interaction ===
        self.drug2_concentration += 0.2 * dose * sensitivity
        self.concentration += self.interaction_strength * self.drug2_concentration

        # === 6. Metabolism / clearance ===
        decay_factor = 1.0 - self.current_metabolism
        self.concentration *= decay_factor
        self.drug2_concentration *= (decay_factor + 0.05)
        self.peripheral_concentration *= (decay_factor + 0.08)  # slower peripheral clearance

        # === 7. Patient variability noise ===
        self.concentration += random.uniform(-self.noise_scale, self.noise_scale)

        # Clamp to non-negative
        self.concentration = max(0.0, self.concentration)
        self.drug2_concentration = max(0.0, self.drug2_concentration)
        self.peripheral_concentration = max(0.0, self.peripheral_concentration)

        # === 8. Update vitals ===
        self.toxicity_flag = self.concentration > self.RISKY_HIGH
        self._update_heart_rate()
        self.concentration_history.append(round(self.concentration, 2))

        # Track time in therapeutic window
        if self.THERAPEUTIC_LOW <= self.concentration <= self.THERAPEUTIC_HIGH:
            self.time_in_therapeutic += 1
        else:
            self.time_in_therapeutic = 0  # reset streak

        # === 9. Compute reward ===
        reward_obj = self._compute_reward()

        # === 10. Episode termination ===
        done = (
            self.step_count >= self.max_steps
            or self.concentration > self.TOXIC_THRESHOLD
        )

        info = {
            "therapeutic_window": (self.THERAPEUTIC_LOW, self.THERAPEUTIC_HIGH),
            "toxic_threshold": self.TOXIC_THRESHOLD,
            "secondary_drug": round(self.drug2_concentration, 2),
            "peripheral_drug": round(self.peripheral_concentration, 2),
            "metabolism_rate": round(self.current_metabolism, 4),
            "clinical_event": self.current_event.value,
            "heart_rate": round(self.heart_rate, 1),
            "reward_breakdown": reward_obj.model_dump(),
            "patient_weight": self.patient_profile.weight_kg,
        }

        return self._get_obs(), reward_obj.value, done, info

    def _generate_clinical_event(self):
        """Randomly generate clinical events based on probability."""
        if not self.clinical_events_enabled:
            self.current_event = ClinicalEvent.NONE
            return

        if random.random() < self.event_probability:
            self.current_event = random.choice([
                ClinicalEvent.FEVER_SPIKE,
                ClinicalEvent.RENAL_DECLINE,
                ClinicalEvent.DRUG_INTERACTION,
                ClinicalEvent.VOMITING,
                ClinicalEvent.FLUID_SHIFT,
            ])
        else:
            self.current_event = ClinicalEvent.NONE

    def _apply_clinical_event(self, dose: float) -> dict:
        """Apply the effect of the current clinical event."""
        effective_dose = dose

        if self.current_event == ClinicalEvent.FEVER_SPIKE:
            # Fever increases metabolism by 15-25%
            self.current_metabolism *= random.uniform(1.15, 1.25)

        elif self.current_event == ClinicalEvent.RENAL_DECLINE:
            # Acute renal decline reduces clearance
            self.patient_profile.renal_function *= 0.8

        elif self.current_event == ClinicalEvent.DRUG_INTERACTION:
            # Another drug amplifies concentration
            self.concentration *= random.uniform(1.05, 1.15)

        elif self.current_event == ClinicalEvent.VOMITING:
            # Patient vomits, losing 30-60% of oral dose
            effective_dose *= random.uniform(0.4, 0.7)

        elif self.current_event == ClinicalEvent.FLUID_SHIFT:
            # IV fluid dilutes concentration
            self.concentration *= random.uniform(0.85, 0.95)

        return {"effective_dose": effective_dose}

    def _update_heart_rate(self):
        """
        Update heart rate based on drug concentration.
        Drug affects heart rate — high concentration causes tachycardia,
        toxic levels cause dangerous arrhythmia-like spikes.
        """
        # Baseline with small random variation
        base_hr = self.NORMAL_HR + random.uniform(-2, 2)

        # Drug effect on heart rate
        if self.concentration > self.RISKY_HIGH:
            # Dangerous: tachycardia
            base_hr += (self.concentration - self.RISKY_HIGH) * 1.5
        elif self.concentration > self.THERAPEUTIC_HIGH:
            # Elevated
            base_hr += (self.concentration - self.THERAPEUTIC_HIGH) * 0.5
        elif self.concentration < self.THERAPEUTIC_LOW and self.concentration > 0:
            # Sub-therapeutic: slight bradycardia
            base_hr -= (self.THERAPEUTIC_LOW - self.concentration) * 0.3

        # Smooth transition
        self.heart_rate = 0.7 * self.heart_rate + 0.3 * base_hr
        self.heart_rate = max(45, min(160, self.heart_rate))

    def _compute_reward(self) -> Reward:
        """
        Multi-objective reward function.

        Components:
            1. Efficacy (+1.0):   bonus for being in therapeutic window
            2. Safety penalty:    -0.5 underdose, -1.0 risky, -2.0 toxic
            3. Stability (+0.3):  bonus for low concentration variance
            4. Shaping:           continuous signal proportional to distance from target

        This design rewards agents that maintain stable, safe therapeutic levels
        over the full trajectory — not just hitting the target once.
        """
        c = self.concentration

        # --- Efficacy ---
        in_therapeutic = self.THERAPEUTIC_LOW <= c <= self.THERAPEUTIC_HIGH
        is_toxic = c > self.RISKY_HIGH

        if in_therapeutic:
            efficacy = 1.0
        elif c < self.THERAPEUTIC_LOW:
            efficacy = -0.5
        elif c <= self.RISKY_HIGH:
            efficacy = -1.0
        else:
            efficacy = -2.0

        # --- Safety penalty for sustained toxicity ---
        safety = 0.0
        if is_toxic:
            safety = -1.0  # additional penalty on top of efficacy

        # --- Stability bonus ---
        stability = 0.0
        if len(self.concentration_history) >= 2:
            recent = self.concentration_history[-3:]  # last 3 readings
            variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
            if variance < 25:  # low variance = stable dosing
                stability = 0.3 * (1.0 - variance / 25.0)

        # --- Continuous shaping ---
        shaping = -abs(c - self.TARGET) / self.TARGET

        # --- Total ---
        value = round(efficacy + safety + stability + shaping, 4)

        return Reward(
            value=value,
            efficacy=efficacy,
            safety=safety,
            stability=round(stability, 4),
            shaping=round(shaping, 4),
            in_therapeutic=in_therapeutic,
            is_toxic=is_toxic,
        )

    def state(self) -> dict:
        """Return the full internal environment state (OpenEnv spec)."""
        s = State(
            step=self.step_count,
            concentration=round(self.concentration, 2),
            drug2_concentration=round(self.drug2_concentration, 2),
            metabolism_rate=round(self.current_metabolism, 4),
            last_dose=round(self.last_dose, 2),
            toxicity_flag=self.toxicity_flag,
            max_steps=self.max_steps,
            done=self.step_count >= self.max_steps or self.concentration > self.TOXIC_THRESHOLD,
            heart_rate=round(self.heart_rate, 1),
            renal_function=round(self.patient_profile.renal_function, 2),
            clinical_event=self.current_event.value,
            concentration_history=self.concentration_history,
            time_in_therapeutic=self.time_in_therapeutic,
        )
        return s.model_dump()

    def close(self):
        """Clean up environment resources."""
        pass

    def _get_obs(self) -> Observation:
        """Build the current observation for the agent."""
        trend = self.concentration - self._prev_concentration
        return Observation(
            step=self.step_count,
            concentration=round(self.concentration, 2),
            secondary_concentration=round(self.drug2_concentration, 2),
            metabolism_rate=round(self.current_metabolism, 4),
            last_dose=round(self.last_dose, 2),
            toxicity_flag=self.toxicity_flag,
            patient_weight=self.patient_profile.weight_kg,
            renal_function=round(self.patient_profile.renal_function, 2),
            heart_rate=round(self.heart_rate, 1),
            clinical_event=self.current_event.value,
            time_in_therapeutic=self.time_in_therapeutic,
            concentration_trend=round(trend, 2),
        )