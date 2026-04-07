import random
from .models import Observation

LOW, HIGH, TOXIC = 10, 50, 70

class MedicationEnv:
    def __init__(self, max_steps=20):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.step_count = 0
        self.concentration = 0.0
        self.metabolism_rate = random.uniform(0.05, 0.2)
        self.last_dose = 0.0
        return self._get_obs()

    def _get_obs(self):
        return Observation(
            step=self.step_count,
            concentration=self.concentration,
            metabolism_rate=self.metabolism_rate,
            last_dose=self.last_dose,
            toxicity_flag=self.concentration > TOXIC
        )

    def update_concentration(self, dose):
        absorbed = dose * 0.8
        cleared = self.concentration * self.metabolism_rate
        self.concentration = max(self.concentration + absorbed - cleared, 0.0)

    def compute_reward(self):
        c = self.concentration
        if c < LOW:
            return -0.5
        elif LOW <= c <= HIGH:
            return 1.0
        elif HIGH < c <= TOXIC:
            return -1.0
        else:
            return -2.0

    def step(self, action):
        self.step_count += 1
        self.last_dose = action.dose

        self.update_concentration(action.dose)
        reward = self.compute_reward()

        done = self.concentration > TOXIC or self.step_count >= self.max_steps

        info = {
            "error": None,
            "concentration": self.concentration
        }

        return self._get_obs(), reward, done, info

    def state(self):
        return {
            "step": self.step_count,
            "concentration": self.concentration
        }