import random
from .models import Observation

class MedicationEnv:
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.concentration = 0.0
        self.drug2_concentration = 0.0  # secondary drug (interaction)
        self.step_count = 0
        return Observation(concentration=self.concentration)

    def step(self, action):
        self.step_count += 1

        # Add primary drug
        self.concentration += action.dose

        # Secondary drug interaction (hidden complexity)
        self.drug2_concentration += 0.2 * action.dose
        self.concentration += 0.3 * self.drug2_concentration

        # Metabolism (decay)
        self.concentration *= 0.85
        self.drug2_concentration *= 0.9

        # Add noise (real-world randomness)
        self.concentration += random.uniform(-1, 1)

        # Clamp concentration
        self.concentration = max(0, self.concentration)

        # Reward (continuous shaping)
        target = 30
        reward = -abs(self.concentration - target) / target

        # Bonus for staying in therapeutic window
        if 10 <= self.concentration <= 50:
            reward += 1.0

        # Done condition
        done = self.step_count >= self.max_steps or self.concentration > 80

        return (
            Observation(concentration=self.concentration),
            reward,
            done,
            {}
        )