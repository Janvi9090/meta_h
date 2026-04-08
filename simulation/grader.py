"""
Multi-criteria grading module for medication dosing episodes.

Evaluates agent performance across four dimensions:

  1. Efficacy (40%):  fraction of steps in therapeutic window [10, 50]
  2. Safety  (30%):   penalizes toxic events (concentration > 70)
  3. Stability (20%): rewards low concentration variance (consistent dosing)
  4. Response (10%):  how quickly the agent reaches therapeutic range

Final score is a weighted combination in [0.0, 1.0].
An episode passes if score >= 0.5 and toxic_steps == 0.
"""

import math

THERAPEUTIC_LOW = 10.0
THERAPEUTIC_HIGH = 50.0
TOXIC_THRESHOLD = 70.0
TARGET = 30.0


def grade(concentrations: list[float]) -> dict:
    """
    Grade an episode based on recorded concentrations.

    Uses multi-criteria evaluation to produce a composite score
    that rewards efficacy, safety, stability, and responsiveness.

    Args:
        concentrations: list of concentration values at each step

    Returns:
        dict with:
            - score: composite score in [0.0, 1.0]
            - efficacy_score: fraction of steps in therapeutic window
            - safety_score: 1.0 if no toxic events, decreases with toxic steps
            - stability_score: based on concentration variance
            - response_score: how quickly therapeutic range was reached
            - therapeutic_steps: count of steps in [10, 50]
            - toxic_steps: count of steps above 70
            - underdose_steps: count of steps below 10
            - overdose_steps: count of steps in (50, 70]
            - max_concentration: peak concentration
            - avg_concentration: mean concentration
            - concentration_std: standard deviation
            - passed: True if score >= 0.5 and no toxic events
    """
    if not concentrations:
        return _empty_result()

    n = len(concentrations)

    # ── Step counts ──
    therapeutic = sum(1 for c in concentrations if THERAPEUTIC_LOW <= c <= THERAPEUTIC_HIGH)
    toxic = sum(1 for c in concentrations if c > TOXIC_THRESHOLD)
    underdose = sum(1 for c in concentrations if c < THERAPEUTIC_LOW)
    overdose = sum(1 for c in concentrations if THERAPEUTIC_HIGH < c <= TOXIC_THRESHOLD)
    max_conc = max(concentrations)
    avg_conc = sum(concentrations) / n

    # ── 1. Efficacy Score (40%) ──
    efficacy_score = therapeutic / n

    # ── 2. Safety Score (30%) ──
    # Perfect if no toxic events, degrades quickly with toxic steps
    if toxic == 0:
        safety_score = 1.0
    else:
        safety_score = max(0.0, 1.0 - (toxic / n) * 3.0)

    # ── 3. Stability Score (20%) ──
    # Lower variance = more stable = higher score
    if n >= 2:
        variance = sum((c - avg_conc) ** 2 for c in concentrations) / n
        std = math.sqrt(variance)
        # Normalize: std < 5 is excellent (score=1.0), std > 20 is poor (score~0)
        stability_score = max(0.0, 1.0 - std / 20.0)
    else:
        stability_score = 0.5

    # ── 4. Response Score (10%) ──
    # How quickly the agent reaches therapeutic range
    first_therapeutic = n  # default: never reached
    for i, c in enumerate(concentrations):
        if THERAPEUTIC_LOW <= c <= THERAPEUTIC_HIGH:
            first_therapeutic = i
            break

    if first_therapeutic == 0:
        response_score = 1.0
    elif first_therapeutic >= n:
        response_score = 0.0
    else:
        # Linear: reaching therapeutic on step 1 = 1.0, step n = 0.0
        response_score = max(0.0, 1.0 - first_therapeutic / (n * 0.5))

    # ── Composite Score ──
    score = (
        0.40 * efficacy_score
        + 0.30 * safety_score
        + 0.20 * stability_score
        + 0.10 * response_score
    )
    score = round(min(1.0, max(0.0, score)), 4)

    # Standard deviation for reporting
    conc_std = math.sqrt(sum((c - avg_conc) ** 2 for c in concentrations) / n) if n > 0 else 0.0

    return {
        "score": score,
        "efficacy_score": round(efficacy_score, 4),
        "safety_score": round(safety_score, 4),
        "stability_score": round(stability_score, 4),
        "response_score": round(response_score, 4),
        "therapeutic_steps": therapeutic,
        "toxic_steps": toxic,
        "underdose_steps": underdose,
        "overdose_steps": overdose,
        "max_concentration": round(max_conc, 2),
        "avg_concentration": round(avg_conc, 2),
        "concentration_std": round(conc_std, 2),
        "passed": score >= 0.5 and toxic == 0,
    }


def _empty_result() -> dict:
    """Return a zero-score result for empty episodes."""
    return {
        "score": 0.0,
        "efficacy_score": 0.0,
        "safety_score": 0.0,
        "stability_score": 0.0,
        "response_score": 0.0,
        "therapeutic_steps": 0,
        "toxic_steps": 0,
        "underdose_steps": 0,
        "overdose_steps": 0,
        "max_concentration": 0.0,
        "avg_concentration": 0.0,
        "concentration_std": 0.0,
        "passed": False,
    }