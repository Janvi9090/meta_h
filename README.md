---
title: Medication Dosing Environment
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
license: mit
pinned: false
---

# 💊 Medication Dosing & Toxicity Control (OpenEnv)

A two-compartment pharmacokinetic simulation for evaluating AI agents on
clinical medication dosing — a task performed daily by ICU nurses, pharmacists,
and automated infusion pumps worldwide.

## 🎯 Motivation

Medication dosing errors are a leading cause of preventable patient harm.
This environment provides a realistic testbed for evaluating whether AI agents
can safely manage drug infusions by:

- Maintaining therapeutic drug levels in the blood
- Avoiding toxic concentrations that cause organ damage
- Adapting to patient-specific physiology and clinical events
- Achieving stable, consistent dosing over time

The simulation models real pharmacokinetic principles including two-compartment
drug distribution, drug-drug interactions, renal/hepatic clearance, and
patient variability — making it directly relevant to clinical decision support.

## 📋 Environment Overview

**Task:** Keep blood drug concentration within the **therapeutic window [10–50]**
while avoiding **toxicity (>70)**. Target concentration is **30**.

**Pharmacokinetic Model:**
- **Two-compartment**: central (blood) + peripheral (tissue) with bidirectional exchange
- **Secondary drug interaction**: hidden pharmacokinetic coupling amplifies concentration
- **Patient-specific metabolism**: affected by renal function, liver function, and body weight
- **Clinical events**: fever, vomiting, renal decline, drug interactions, fluid shifts

**Episode termination:**
- Maximum steps reached (task-dependent)
- Concentration exceeds critical threshold (>80) — emergency overdose

## 🎮 Action & Observation Spaces

### Action Space
| Field | Type  | Range  | Description                         |
|-------|-------|--------|-------------------------------------|
| dose  | float | 0–20   | Drug dose to administer (units)     |

### Observation Space
| Field                    | Type  | Description                                   |
|-------------------------|-------|-----------------------------------------------|
| step                    | int   | Current timestep                              |
| concentration           | float | Blood drug concentration (units)              |
| secondary_concentration | float | Secondary drug blood level                    |
| metabolism_rate          | float | Current effective metabolism rate (0.05–0.5)  |
| last_dose               | float | Previous dose administered                    |
| toxicity_flag           | bool  | True if concentration > 70                    |
| patient_weight          | float | Patient weight in kg                          |
| renal_function          | float | Renal clearance multiplier (1.0=normal)       |
| heart_rate              | float | Heart rate in BPM (affected by drug levels)   |
| clinical_event          | str   | Active clinical event (e.g., "fever_spike")   |
| time_in_therapeutic     | int   | Consecutive steps in therapeutic window       |
| concentration_trend     | float | Rate of change vs. previous step              |

## 📊 Difficulty Levels

Each difficulty models a distinct clinical scenario with a different patient:

| Task   | Patient            | Steps | Metabolism | Variance | Noise | Events | Description                      |
|--------|--------------------|-------|-----------|----------|-------|--------|----------------------------------|
| Easy   | 70kg, healthy      | 10    | 0.15      | 0.00     | 0.5   | None   | Stable ward patient              |
| Medium | 82kg, mild CKD     | 15    | 0.18      | 0.05     | 1.0   | None   | Post-surgical, variable metabolism|
| Hard   | 95kg, ICU critical | 20    | 0.22      | 0.10     | 2.5   | 20%    | Elderly, impaired organs, events |

**Hard task features:**
- 95kg elderly patient (age 71) with significantly impaired renal (0.65) and hepatic (0.75) function
- Drug hypersensitivity (1.3x)
- 20% chance of clinical events each step (fever, vomiting, renal decline, drug interactions, fluid shifts)
- Mid-episode metabolism shift
- High noise and strong secondary drug interactions

## 🏆 Reward Structure

The reward function is **multi-objective**, providing signal across four dimensions:

| Component       | Weight | Condition            | Signal         |
|----------------|--------|---------------------|----------------|
| **Efficacy**   | —      | Therapeutic [10–50]  | +1.0 bonus     |
|                |        | Underdose (<10)      | −0.5 penalty   |
|                |        | Risky high (50–70]   | −1.0 penalty   |
|                |        | Toxic (>70)          | −2.0 penalty   |
| **Safety**     | —      | Toxic (>70)          | −1.0 additional |
| **Stability**  | —      | Low variance (σ<5)   | +0.3 bonus     |
| **Shaping**    | —      | Continuous           | −|c − 30|/30   |

Key design properties:
- **Dense signal**: rewards at every step, not just terminal
- **Partial credit**: continuous shaping guides toward target
- **Safety-weighted**: toxicity is heavily penalized (double penalty)
- **Stability bonus**: rewards consistent dosing over oscillating

## 📏 Grading (Multi-Criteria Score)

Each episode is graded on a **composite score in [0.0, 1.0]**:

| Criterion       | Weight | Measures                                    |
|----------------|--------|---------------------------------------------|
| Efficacy       | 40%    | Fraction of steps in therapeutic window     |
| Safety         | 30%    | Absence of toxic events                     |
| Stability      | 20%    | Low concentration variance (consistent)     |
| Response time  | 10%    | Speed of reaching therapeutic range         |

An episode **passes** if: composite score ≥ 0.5 AND zero toxic events.

## 📈 Baseline Performance (Heuristic Controller)

| Task   | Score | Efficacy | Safety | Stability | Response | Steps | Toxic | Result  |
|--------|-------|----------|--------|-----------|----------|-------|-------|---------|
| Easy   | 0.90  | 0.90     | 1.00   | 0.84      | 0.80     | 10    | 0     | ✅ PASS |
| Medium | 0.92  | 0.93     | 1.00   | 0.89      | 0.80     | 15    | 0     | ✅ PASS |
| Hard   | 0.90  | 0.90     | 1.00   | 0.86      | 0.60     | 20    | 0     | ✅ PASS |

*The heuristic uses PID-inspired control with weight, renal function, and clinical event compensation.*

## 📁 Project Structure
```
meta_h/
├── inference.py              # Baseline agent (LLM + heuristic fallback)
├── app.py                    # FastAPI server (HF Spaces deployment)
├── server/
│   ├── __init__.py           # Package exports
│   ├── app.py                # FastAPI application with OpenEnv endpoints
│   └── medication_environment.py  # OpenEnv Environment wrapper
├── simulation/
│   ├── __init__.py           # Package exports
│   ├── environment.py        # MedicationEnv (two-compartment PK model)
│   ├── models.py             # Pydantic: Observation, Action, Reward, State
│   ├── tasks.py              # Patient profiles + difficulty configs
│   └── grader.py             # Multi-criteria episode grading
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build (port 7860)
├── openenv.yaml              # OpenEnv metadata
└── README.md
```

## 🚀 How to Run

### Local (heuristic mode — no API key needed)
```bash
pip install -r requirements.txt
python inference.py
```

### With LLM agent
```bash
export HF_TOKEN="your-api-key"
export MODEL_NAME="gpt-4.1-mini"                    # optional, default
export API_BASE_URL="https://api.openai.com/v1"      # optional, default
python inference.py
```

### Docker (HF Space server)
```bash
docker build -t medication-dosing .
docker run -p 7860:7860 medication-dosing
```

### API Endpoints

| Method | Path     | Description                              |
|--------|----------|------------------------------------------|
| GET    | /        | Health check                             |
| GET    | /health  | Health check (OpenEnv spec)              |
| GET    | /tasks   | List available tasks with configs        |
| POST   | /reset   | Reset env (body: `{"task": "easy"}`)     |
| POST   | /step    | Take action (body: `{"action": {"dose": 5.0}}`) |
| GET    | /state   | Get current environment state            |

### Example API Usage
```bash
# Reset to hard task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard"}'

# Administer a dose
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"dose": 10.0}}'

# Check state
curl http://localhost:7860/state
```

## 🤖 Agent Modes
1. **LLM Mode** (when `HF_TOKEN` is set): Uses an OpenAI-compatible model to
   decide doses based on the full patient observation vector and recent history.
2. **Heuristic Mode** (fallback): PID-inspired controller that adapts to patient
   weight, renal function, clinical events, and concentration trends.

## 🧪 Clinical Events

On hard difficulty, the environment randomly generates clinical events that
disrupt pharmacokinetics:

| Event             | Effect                                    |
|------------------|-------------------------------------------|
| Fever Spike      | Increases metabolism 15–25%               |
| Renal Decline    | Reduces clearance (accumulation risk)     |
| Drug Interaction | Amplifies blood concentration 5–15%       |
| Vomiting         | Patient loses 30–60% of oral dose         |
| Fluid Shift      | IV fluids dilute concentration 5–15%      |

Agents must detect these events via the observation and adjust dosing strategy.