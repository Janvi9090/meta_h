import os
from openai import OpenAI
from env.tasks import get_task
from env.models import Action

# Environment variables (as required)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client (required by hackathon rules)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Smarter adaptive dosing policy
def choose_action(obs):
    target = 30  # midpoint of therapeutic window (10–50)
    error = target - obs.concentration

    # proportional control
    dose = max(0, min(20, error * 0.5))
    return round(dose, 2)


def run_episode(task_name="easy"):
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

            # Choose action
            dose = choose_action(obs)
            action = Action(dose=dose)

            # Step environment
            obs, reward, done, info = env.step(action)

            # Store rewards
            raw_rewards.append(reward)
            rewards.append(f"{reward:.2f}")

            # Print STEP (strict format)
            print(
                f"[STEP] step={step} action=dose({dose}) "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

    except Exception as e:
        # Print error step (still required)
        print(
            f"[STEP] step={step} action=error "
            f"reward=0.00 done=true error={str(e)}"
        )

    finally:
        # Success = stable non-negative rewards throughout
        success = "true" if raw_rewards and all(r >= 0 for r in raw_rewards) else "false"

        # Print END (strict format)
        print(f"[END] success={success} steps={step} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run_episode("easy")