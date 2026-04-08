"""
Medication Dosing & Toxicity Control — Inference Script

Runs an AI-powered dosing agent through easy, medium, and hard episodes.
Uses an LLM (via OpenAI-compatible API) to decide doses based on
patient observations, with a heuristic fallback if the model is unavailable.

STDOUT FORMAT (OpenEnv compliant):
  [START] task=<task_name> env=medication-dosing-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
from typing import List, Optional

from openai import OpenAI
from simulation.tasks import get_task
from simulation.models import Action
from simulation.grader import grade

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_NAME = "medication-dosing-env"
USE_LLM = HF_TOKEN is not None

if USE_LLM:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
else:
    client = None


# ──────────────────────────────────────────────────────────────
# Structured logging helpers (OpenEnv compliant)
# ──────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────
# System prompt for the LLM agent
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical pharmacology AI assistant managing drug dosing for a hospitalized patient.

ENVIRONMENT:
- You control an IV drug infusion pump.
- The patient's blood drug concentration must stay in the therapeutic window: 10–50 units.
- Target concentration is 30 units.
- Concentration > 70 is TOXIC (causes tachycardia, organ stress).
- Concentration > 80 causes immediate termination (critical overdose).

PHARMACOKINETICS:
- The drug follows a two-compartment model (central blood + peripheral tissue).
- A secondary drug interacts with the primary — boosting effective concentration.
- Metabolism rate varies with the patient's renal and hepatic function.
- Clinical events (fever, vomiting, renal decline) can disrupt pharmacokinetics.

PATIENT OBSERVATIONS YOU RECEIVE:
- Blood concentration, metabolism rate, heart rate, renal function
- Toxicity flag, clinical events, concentration trend
- Time in therapeutic window (consecutive steps)

STRATEGY GUIDELINES:
1. Start with a moderate dose (8-12 for normal patients, less for sensitive/heavy patients)
2. Watch the concentration_trend — positive means rising, negative means falling
3. If toxicity_flag is True: STOP dosing (dose=0)
4. Account for patient weight: heavier patients need more drug to achieve same concentration
5. Clinical events change the rules — fever speeds metabolism, vomiting reduces absorption
6. Aim for stability: consistent concentrations are better than oscillating

Respond with ONLY a JSON object: {"dose": <number between 0 and 20>}"""


# ──────────────────────────────────────────────────────────────
# Dosing strategies
# ──────────────────────────────────────────────────────────────
def choose_action_heuristic(obs) -> float:
    """
    Adaptive PID-inspired heuristic controller.
    Accounts for patient weight, renal function, clinical events,
    concentration trends, and toxicity flags.
    """
    target = 30.0

    # If toxic, stop dosing immediately
    if obs.toxicity_flag:
        return 0.0

    error = target - obs.concentration
    metabolism_factor = obs.metabolism_rate / 0.15  # normalize around baseline
    weight_factor = obs.patient_weight / 70.0  # heavier patients need more

    # Base proportional control with metabolism and weight compensation
    dose = error * 0.35 * metabolism_factor

    # Weight compensation: heavier patients dilute the drug more
    dose *= weight_factor

    # Derivative control: if concentration is rising fast, reduce dose
    if obs.concentration_trend > 5:
        dose *= 0.5
    elif obs.concentration_trend < -5:
        dose *= 1.3  # compensate for rapid decline

    # Renal function adjustment
    if obs.renal_function < 0.8:
        dose *= obs.renal_function  # reduce dose for impaired kidneys

    # Dampen if we're already close to target
    if abs(error) < 5:
        dose *= 0.5

    # Safety: reduce dose if concentration is getting high
    if obs.concentration > 45:
        dose = max(0, dose * 0.3)

    # Clinical event adjustments
    if obs.clinical_event == "fever_spike":
        dose *= 1.15  # compensate for increased metabolism
    elif obs.clinical_event == "vomiting":
        dose *= 1.4   # compensate for dose loss
    elif obs.clinical_event == "renal_decline":
        dose *= 0.7   # reduce to avoid accumulation
    elif obs.clinical_event == "fluid_shift":
        dose *= 1.1   # compensate for dilution

    # Clamp to valid range
    dose = max(0.0, min(20.0, dose))
    return round(dose, 2)


def choose_action_llm(obs, history: list) -> float:
    """
    Ask the LLM to decide the dose based on patient observations.
    Falls back to heuristic if LLM call fails.
    """
    user_message = (
        f"Current patient state:\n"
        f"- Step: {obs.step}\n"
        f"- Blood concentration: {obs.concentration:.2f}\n"
        f"- Secondary drug concentration: {obs.secondary_concentration:.2f}\n"
        f"- Metabolism rate: {obs.metabolism_rate:.4f}\n"
        f"- Last dose given: {obs.last_dose:.2f}\n"
        f"- Toxicity flag: {obs.toxicity_flag}\n"
        f"- Patient weight: {obs.patient_weight:.1f} kg\n"
        f"- Renal function: {obs.renal_function:.2f}\n"
        f"- Heart rate: {obs.heart_rate:.1f} BPM\n"
        f"- Clinical event: {obs.clinical_event}\n"
        f"- Concentration trend: {obs.concentration_trend:+.2f}\n"
        f"- Consecutive steps in therapeutic: {obs.time_in_therapeutic}\n"
        f"\nRecent history (last 3 steps):\n"
    )

    for h in history[-3:]:
        user_message += (
            f"  Step {h['step']}: dose={h['dose']:.2f} → "
            f"conc={h['concentration']:.2f}, reward={h['reward']:.2f}"
            f"{', event=' + h['event'] if h['event'] != 'none' else ''}\n"
        )

    user_message += "\nWhat dose should be administered? Respond with JSON only."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=50,
        )

        text = response.choices[0].message.content.strip()

        # Parse JSON from response
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            dose = float(data.get("dose", 0))
            return max(0.0, min(20.0, round(dose, 2)))

    except Exception as e:
        pass  # fall through to heuristic

    return choose_action_heuristic(obs)


# ──────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────
def run_episode(task_name: str) -> dict:
    """
    Run a full episode for the given task difficulty.

    Returns:
        dict with episode results and grade
    """
    env = get_task(task_name)
    obs = env.reset()

    model_label = MODEL_NAME if USE_LLM else "heuristic"
    log_start(task=task_name, env=ENV_NAME, model=model_label)

    rewards = []
    concentrations = []
    history = []
    step = 0
    last_error = None

    try:
        done = False
        while not done:
            step += 1
            last_error = None

            # Choose action
            if USE_LLM:
                dose = choose_action_llm(obs, history)
            else:
                dose = choose_action_heuristic(obs)

            action = Action(dose=dose)

            # Step environment
            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                last_error = str(e)
                reward = 0.0
                done = True

            # Record
            rewards.append(reward)
            concentrations.append(obs.concentration)
            history.append({
                "step": step,
                "dose": dose,
                "concentration": obs.concentration,
                "reward": reward,
                "event": obs.clinical_event,
            })

            # Emit structured [STEP] line
            action_str = f"dose({dose:.2f})"
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=last_error,
            )

    except Exception as e:
        last_error = str(e)
        rewards.append(0.0)

    # Grade the episode
    result = grade(concentrations)
    score = result["score"]  # already in [0.0, 1.0]
    success = result["passed"]

    # Always emit [END] (even on exception)
    env.close()
    log_end(success=success, steps=step, score=score, rewards=rewards)

    return {
        "task": task_name,
        "grade": result,
        "score": score,
        "total_reward": sum(rewards),
        "steps": step,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for task in ["easy", "medium", "hard"]:
        result = run_episode(task)
        results.append(result)