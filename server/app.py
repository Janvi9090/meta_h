"""
Medication Dosing Environment Server Application.

Creates a FastAPI app that conforms to the OpenEnv HTTP API specification.

Two modes:
  1. If openenv-core is installed → uses create_app() factory (preferred)
  2. Fallback → standalone FastAPI with OpenEnv-compatible endpoints

The OpenEnv spec requires:
  POST /reset  → ResetResponse {observation: {}, reward: null, done: false}
  POST /step   → StepResponse  {observation: {}, reward: float, done: bool}
  GET  /health → HealthResponse {status: "healthy"}

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server import create_app
    from openenv.core.env_server.types import Action as OEAction, Observation as OEObservation
    from .medication_environment import MedicationDosingEnvironment

    # Use OpenEnv's create_app factory — this handles the full spec:
    # /reset, /step, /state, /health, /schema, /ws, /docs
    app = create_app(
        MedicationDosingEnvironment,
        OEAction,
        OEObservation,
        env_name="medication_dosing_env",
    )
    print("[server] Using openenv-core create_app", flush=True)

except Exception as e:
    print(f"[server] openenv-core not available ({e}), using standalone FastAPI", flush=True)
    traceback.print_exc()

    # ─── Standalone fallback that matches OpenEnv HTTP API ───
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from typing import Any, Dict, Optional

    from simulation.environment import MedicationEnv
    from simulation.models import Action as MedAction
    from simulation.tasks import TASK_CONFIGS

    app = FastAPI(
        title="Medication Dosing OpenEnv",
        description="Two-compartment pharmacokinetic simulation for AI agent evaluation.",
        version="1.0",
    )

    # ─── Per-request env storage ───
    envs: dict[str, MedicationEnv] = {}
    current_task: str = "easy"

    # ─── Request/Response models matching OpenEnv spec ───

    class ResetRequest(BaseModel):
        """Matches openenv.core.env_server.types.ResetRequest"""
        seed: Optional[int] = Field(default=None, description="Random seed")
        episode_id: Optional[str] = Field(default=None, description="Episode ID")

        class Config:
            extra = "allow"  # Allow extra fields like 'task'

    class StepRequest(BaseModel):
        """Matches openenv.core.env_server.types.StepRequest"""
        action: Dict[str, Any] = Field(..., description="Action dict with 'dose' key")
        timeout_s: Optional[float] = Field(default=None, description="Timeout")
        request_id: Optional[str] = Field(default=None, description="Request ID")

        class Config:
            extra = "allow"

    class ResetResponse(BaseModel):
        """Matches openenv.core.env_server.types.ResetResponse"""
        observation: Dict[str, Any] = Field(..., description="Initial observation")
        reward: Optional[float] = Field(default=None, description="Initial reward")
        done: bool = Field(default=False, description="Episode done flag")

    class StepResponse(BaseModel):
        """Matches openenv.core.env_server.types.StepResponse"""
        observation: Dict[str, Any] = Field(..., description="Observation")
        reward: Optional[float] = Field(default=None, description="Reward")
        done: bool = Field(default=False, description="Episode done flag")

    class HealthResponse(BaseModel):
        status: str = "healthy"

    # ─── Endpoints matching OpenEnv HTTP spec ───

    @app.get("/health")
    def health() -> HealthResponse:
        """Health check — OpenEnv spec."""
        return HealthResponse(status="healthy")

    @app.get("/")
    def root():
        """Root health check."""
        return {"status": "healthy"}

    @app.post("/reset")
    def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
        """
        Reset the environment — OpenEnv spec compliant.

        Returns ResetResponse with observation dict, reward=None, done=False.
        """
        global current_task

        # Check for task in the extra fields
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

        # Build observation dict — all simulation fields go into observation
        obs_dict = obs.model_dump()

        # Add env info to observation metadata
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

    @app.post("/step")
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
                detail=f"Environment not initialized. Call /reset first.",
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
        # Merge info into observation for richer data
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
        """List available tasks."""
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
