"""
Medication Dosing Environment — FastAPI Server (OpenEnv HTTP API).

Implements the OpenEnv-compatible HTTP API specification:
  POST /reset  → ResetResponse {observation: {}, reward: null, done: false}
  POST /step   → StepResponse  {observation: {}, reward: float, done: bool}
  GET  /health → HealthResponse {status: "healthy"}
  GET  /state  → Current environment state
  GET  /tasks  → Available task configurations

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from simulation.environment import MedicationEnv
from simulation.models import Action as MedAction
from simulation.tasks import TASK_CONFIGS

app = FastAPI(
    title="Medication Dosing OpenEnv",
    description="Two-compartment pharmacokinetic simulation for AI agent evaluation.",
    version="1.0.0",
)

# ─── Per-session env storage ───
envs: Dict[str, MedicationEnv] = {}
current_task: str = "easy"


# ─── Request/Response models matching OpenEnv spec ───

class ResetRequest(BaseModel):
    """Reset request — accepts optional task name and seed."""
    task: Optional[str] = Field(default=None, description="Task difficulty: easy, medium, hard")
    seed: Optional[int] = Field(default=None, description="Random seed")
    episode_id: Optional[str] = Field(default=None, description="Episode ID")

    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    """Step request — accepts action dict with dose key."""
    action: Dict[str, Any] = Field(..., description="Action dict with 'dose' key")
    timeout_s: Optional[float] = Field(default=None, description="Timeout")
    request_id: Optional[str] = Field(default=None, description="Request ID")

    class Config:
        extra = "allow"


class ResetResponse(BaseModel):
    """OpenEnv reset response."""
    observation: Dict[str, Any] = Field(..., description="Initial observation")
    reward: Optional[float] = Field(default=None, description="Initial reward")
    done: bool = Field(default=False, description="Episode done flag")


class StepResponse(BaseModel):
    """OpenEnv step response."""
    observation: Dict[str, Any] = Field(..., description="Observation")
    reward: Optional[float] = Field(default=None, description="Reward")
    done: bool = Field(default=False, description="Episode done flag")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"


# ─── Endpoints matching OpenEnv HTTP spec ───

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — OpenEnv spec."""
    return HealthResponse(status="healthy")


@app.get("/")
def root():
    """Root health check."""
    return {"status": "healthy", "env": "medication-dosing-env", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = None) -> ResetResponse:
    """
    Reset the environment — OpenEnv spec compliant.

    Accepts optional body with {"task": "easy"|"medium"|"hard"}.
    Empty body or no body defaults to "easy".
    """
    global current_task

    # Handle no body / empty body
    if req is None:
        req = ResetRequest()

    # Get task name from explicit field or extra fields
    task_name = req.task
    if task_name is None:
        extra = req.model_extra or {}
        task_name = extra.get("task", "easy")

    if task_name not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Choose from: {list(TASK_CONFIGS.keys())}",
        )

    env = MedicationEnv(**TASK_CONFIGS[task_name])
    obs = env.reset()
    envs[task_name] = env
    current_task = task_name

    # Build observation dict
    obs_dict = obs.model_dump()
    obs_dict["task"] = task_name
    obs_dict["therapeutic_window"] = [env.THERAPEUTIC_LOW, env.THERAPEUTIC_HIGH]
    obs_dict["toxic_threshold"] = env.TOXIC_THRESHOLD
    obs_dict["target"] = env.TARGET
    obs_dict["max_steps"] = env.max_steps

    return ResetResponse(
        observation=obs_dict,
        reward=None,
        done=False,
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """
    Execute one step — OpenEnv spec compliant.

    Expects action dict with 'dose' key.
    Returns StepResponse with observation, reward, done.
    """
    task_name = current_task

    if task_name not in envs:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )

    env = envs[task_name]

    # Extract dose from action dict
    action_data = req.action
    dose = action_data.get("dose", 0.0)
    dose = max(0.0, min(20.0, float(dose)))

    action = MedAction(dose=dose)

    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {str(e)}")

    # Build observation dict
    obs_dict = obs.model_dump()
    obs_dict.update(info)

    return StepResponse(
        observation=obs_dict,
        reward=round(reward, 4),
        done=done,
    )


@app.get("/state")
def get_state():
    """Return current environment state."""
    task_name = current_task
    if task_name not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return envs[task_name].state()


@app.get("/tasks")
def list_tasks():
    """List available tasks with their configurations."""
    tasks = {}
    for name, config in TASK_CONFIGS.items():
        tasks[name] = {
            "max_steps": config["max_steps"],
            "metabolism_base": config["metabolism_base"],
            "noise_scale": config["noise_scale"],
            "clinical_events": config.get("clinical_events", False),
        }
    return {"tasks": tasks}


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
