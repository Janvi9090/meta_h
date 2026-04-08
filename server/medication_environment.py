"""
Medication Dosing Environment — OpenEnv Environment Implementation.

Wraps the simulation logic to conform to the openenv-core Environment interface.
This module properly implements the Environment base class with:
  - reset() returning an Observation (with done, reward, metadata)
  - step() accepting an Action and returning an Observation
  - state property returning a State
"""

from typing import Any, Optional
from uuid import uuid4

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MedicationEnv
from simulation.models import Action as MedAction
from simulation.tasks import TASK_CONFIGS

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import (
        Action as OEAction,
        Observation as OEObservation,
        State as OEState,
    )
    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False


if OPENENV_AVAILABLE:
    class MedicationDosingEnvironment(Environment):
        """
        OpenEnv-compatible medication dosing environment.

        Wraps the two-compartment pharmacokinetic simulation to provide
        the standard reset/step/state interface required by openenv-core.

        Key conformance points:
        - reset() returns OEObservation with done=False, reward=None, metadata={}
        - step() returns OEObservation with done, reward, metadata
        - state is a property returning OEState
        """

        def __init__(self):
            super().__init__()
            self._task = "easy"
            self._env = MedicationEnv(**TASK_CONFIGS[self._task])
            self._state = OEState(episode_id=str(uuid4()), step_count=0)

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> OEObservation:
            task = kwargs.get("task", "easy")
            if task in TASK_CONFIGS:
                self._task = task
                self._env = MedicationEnv(**TASK_CONFIGS[task])

            obs = self._env.reset()
            self._state = OEState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
            )

            # Return proper OpenEnv Observation
            obs_data = obs.model_dump()
            return OEObservation(
                done=False,
                reward=None,
                metadata={
                    "task": self._task,
                    "therapeutic_window": [self._env.THERAPEUTIC_LOW, self._env.THERAPEUTIC_HIGH],
                    "toxic_threshold": self._env.TOXIC_THRESHOLD,
                    "target": self._env.TARGET,
                    "max_steps": self._env.max_steps,
                    **obs_data,
                },
            )

        def step(
            self,
            action: OEAction,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> OEObservation:
            self._state.step_count += 1

            # Extract dose from action
            dose = 0.0
            if hasattr(action, 'dose'):
                dose = action.dose
            elif hasattr(action, 'metadata') and isinstance(action.metadata, dict):
                dose = action.metadata.get('dose', 0.0)

            med_action = MedAction(dose=max(0.0, min(20.0, dose)))
            obs, reward, done, info = self._env.step(med_action)

            obs_data = obs.model_dump()
            return OEObservation(
                done=done,
                reward=reward,
                metadata={
                    **obs_data,
                    **info,
                },
            )

        @property
        def state(self) -> OEState:
            return self._state

        def close(self) -> None:
            """Clean up resources."""
            pass

else:
    # Placeholder if openenv-core not available
    class MedicationDosingEnvironment:
        pass
