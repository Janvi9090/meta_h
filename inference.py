import os
from openai import OpenAI
from env.tasks import get_task
from env.models import Action

# Env variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Smarter adaptive policy
def choose_action(obs):
    target = 30
    error = target - obs.concentration

    # Slightly conservative control for noisy system
    dose = max(0, min(20, error * 0.4))
    return round(dose, 2)


def run_episode(task_name):
    env = get_task(task_name)
    obs = env.reset()

    print(f"[START] task={task_name} env=medication model={MODEL_NAME}")

    rewards = []
    raw_rewards = []
    step = 0

    try:
        done = False
        while not done:
            step += 1

            dose = choose_action(obs)
            action = Action(dose=dose)

            obs, reward, done, info = env.step(action)

            raw_rewards.append(reward)
            rewards.append(f"{reward:.2f}")

            print(
                f"[STEP] step={step} action=dose({dose}) "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

    except Exception as e:
        print(
            f"[STEP] step={step} action=error "
            f"reward=0.00 done=true error={str(e)}"
        )

    finally:
        success = "true" if raw_rewards and all(r >= -0.2 for r in raw_rewards) else "false"
        print(f"[END] success={success} steps={step} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run_episode("easy")
    run_episode("medium")
    run_episode("hard")