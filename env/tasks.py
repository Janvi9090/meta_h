from .environment import MedicationEnv

def get_task(task_name="easy"):
    if task_name == "easy":
        env = MedicationEnv(max_steps=10)
        env.metabolism_rate = 0.1
        return env

    elif task_name == "medium":
        return MedicationEnv(max_steps=15)

    elif task_name == "hard":
        env = MedicationEnv(max_steps=20)
        env.metabolism_rate = 0.15
        return env

    else:
        raise ValueError("Invalid task name")