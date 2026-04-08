"""
Medication Dosing Environment Server Application.

Creates a FastAPI app using openenv-core's create_app factory,
matching the official OpenEnv pattern for HF Spaces deployment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import Action, Observation

    from .medication_environment import MedicationDosingEnvironment

    app = create_app(
        MedicationDosingEnvironment, Action, Observation,
        env_name="medication_dosing_env"
    )
except ImportError:
    # Fallback: if openenv-core is not installed, use standalone FastAPI
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from simulation.environment import MedicationEnv
    from simulation.models import Action as MedAction
    from simulation.tasks import TASK_CONFIGS

    app = FastAPI(
        title="Medication Dosing OpenEnv",
        description="Two-compartment pharmacokinetic simulation for AI agent evaluation.",
        version="1.0",
    )

    envs: dict[str, MedicationEnv] = {}
    current_task: str = "easy"

    class ResetRequest(BaseModel):
        task: Optional[str] = "easy"

    class StepRequest(BaseModel):
        task: Optional[str] = None
        dose: float = 0.0

    @app.get("/")
    def health():
        return {
            "status": "ok",
            "environment": "medication-dosing-env",
            "version": "1.0",
            "tasks": list(TASK_CONFIGS.keys()),
        }

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    @app.get("/tasks")
    def list_tasks():
        tasks = {}
        for name, config in TASK_CONFIGS.items():
            tasks[name] = {
                "max_steps": config["max_steps"],
                "metabolism_base": config["metabolism_base"],
                "noise_scale": config["noise_scale"],
                "clinical_events": config.get("clinical_events", False),
            }
        return {"tasks": tasks}

    @app.post("/reset")
    def reset(req: ResetRequest = ResetRequest()):
        global current_task
        task_name = req.task or "easy"
        if task_name not in TASK_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")
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
        task_name = req.task or current_task
        if task_name not in envs:
            raise HTTPException(status_code=400, detail=f"Call /reset first.")
        env = envs[task_name]
        action = MedAction(dose=max(0.0, min(20.0, req.dose)))
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": round(reward, 4),
            "done": done,
            "info": info,
        }

    @app.get("/state")
    def state():
        task_name = current_task
        if task_name not in envs:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        return envs[task_name].state()


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
