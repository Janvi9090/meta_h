"""
Medication Dosing Environment — OpenEnv Environment Implementation.

Wraps the simulation logic to conform to the openenv-core Environment interface.
"""

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.env_server.environment import Environment

    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MedicationEnv
from simulation.models import Action as MedAction
from simulation.tasks import TASK_CONFIGS


if OPENENV_AVAILABLE:
    class MedicationDosingEnvironment(Environment):
        """
        OpenEnv-compatible medication dosing environment.

        Wraps the two-compartment pharmacokinetic simulation to provide
        the standard reset/step/state interface required by openenv-core.
        """

        def __init__(self):
            self._task = "easy"
            self._env = MedicationEnv(**TASK_CONFIGS[self._task])
            self._state = State(episode_id=str(uuid4()), step_count=0)

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> Observation:
            task = kwargs.get("task", "easy")
            if task in TASK_CONFIGS:
                self._task = task
                self._env = MedicationEnv(**TASK_CONFIGS[task])

            obs = self._env.reset()
            self._state = State(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
            )

            return Observation(
                done=False,
                reward=0.0,
                metadata={
                    "status": "ready",
                    "task": self._task,
                    "observation": obs.model_dump(),
                    "therapeutic_window": [self._env.THERAPEUTIC_LOW, self._env.THERAPEUTIC_HIGH],
                    "target": self._env.TARGET,
                },
            )

        def step(
            self,
            action: Action,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> Observation:
            self._state.step_count += 1

            # Extract dose from action
            dose = 0.0
            if hasattr(action, 'dose'):
                dose = action.dose
            elif hasattr(action, 'metadata') and isinstance(action.metadata, dict):
                dose = action.metadata.get('dose', 0.0)

            med_action = MedAction(dose=max(0.0, min(20.0, dose)))
            obs, reward, done, info = self._env.step(med_action)

            return Observation(
                done=done,
                reward=reward,
                metadata={
                    "observation": obs.model_dump(),
                    "info": info,
                },
            )

        @property
        def state(self) -> State:
            return self._state

else:
    # Placeholder if openenv-core not available
    class MedicationDosingEnvironment:
        pass
