from .environment import MedicationEnv

def get_task(task_name="easy"):
    if task_name == "easy":
        return MedicationEnv(max_steps=10)

    elif task_name == "medium":
        return MedicationEnv(max_steps=15)

    elif task_name == "hard":
        return MedicationEnv(max_steps=20)

    else:
        raise ValueError("Invalid task name")