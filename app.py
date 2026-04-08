"""
FastAPI server for the Medication Dosing OpenEnv environment.

Exposes the OpenEnv API for Hugging Face Spaces deployment:
  POST /reset  → reset environment, return initial observation
  POST /step   → take an action, return (observation, reward, done, info)
  GET  /state  → return current environment state
  GET  /       → health check
  GET  /tasks  → list available tasks with descriptions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from simulation.environment import MedicationEnv
from simulation.models import Action
from simulation.tasks import TASK_CONFIGS

app = FastAPI(
    title="Medication Dosing OpenEnv",
    description="Two-compartment pharmacokinetic simulation for AI agent evaluation. "
                "Agents must maintain drug concentration in a therapeutic window "
                "while handling patient variability and clinical events.",
    version="1.0",
)

# Active environment instances keyed by task name
envs: dict[str, MedicationEnv] = {}
current_task: str = "easy"


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


class StepRequest(BaseModel):
    task: Optional[str] = None
    dose: float = 0.0


@app.get("/")
def health():
    """Health check — returns 200 with environment metadata."""
    return {
        "status": "ok",
        "environment": "medication-dosing-env",
        "version": "1.0",
        "tasks": list(TASK_CONFIGS.keys()),
        "description": "Two-compartment pharmacokinetic medication dosing simulation",
    }


@app.get("/tasks")
def list_tasks():
    """List available tasks with their configurations."""
    tasks = {}
    for name, config in TASK_CONFIGS.items():
        tasks[name] = {
            "max_steps": config["max_steps"],
            "metabolism_base": config["metabolism_base"],
            "metabolism_variance": config["metabolism_variance"],
            "noise_scale": config["noise_scale"],
            "clinical_events": config.get("clinical_events", False),
            "patient_weight": config.get("patient_profile").weight_kg if config.get("patient_profile") else 70.0,
        }
    return {"tasks": tasks}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial observation."""
    global current_task
    task_name = req.task or "easy"

    if task_name not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Choose from: {list(TASK_CONFIGS.keys())}",
        )

    env = MedicationEnv(**TASK_CONFIGS[task_name])
    obs = env.reset()
    envs[task_name] = env
    current_task = task_name

    return {
        "observation": obs.model_dump(),
        "task": task_name,
        "info": {
            "therapeutic_window": [env.THERAPEUTIC_LOW, env.THERAPEUTIC_HIGH],
            "toxic_threshold": env.TOXIC_THRESHOLD,
            "target": env.TARGET,
            "max_steps": env.max_steps,
        },
    }


@app.post("/step")
def step(req: StepRequest):
    """Execute one step in the environment."""
    task_name = req.task or current_task

    if task_name not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Environment for task '{task_name}' not initialized. Call /reset first.",
        )

    env = envs[task_name]
    action = Action(dose=max(0.0, min(20.0, req.dose)))

    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment step error: {str(e)}")

    return {
        "observation": obs.model_dump(),
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return the current environment state."""
    task_name = current_task

    if task_name not in envs:
        raise HTTPException(
            status_code=400,
            detail="No environment initialized. Call /reset first.",
        )

    return envs[task_name].state()
