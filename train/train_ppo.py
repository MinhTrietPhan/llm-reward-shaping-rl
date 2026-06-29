import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import highway_env
import yaml

from stable_baselines3 import PPO
from envs.reward_wrapper import CustomRewardWrapper


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/manual_reward.yaml")

    env = gym.make(config["env_name"])
    env = CustomRewardWrapper(env, config["reward_weights"])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["train"]["learning_rate"],
        n_steps=config["train"]["n_steps"],
        batch_size=config["train"]["batch_size"],
        gamma=config["train"]["gamma"],
        verbose=1,
    )

    model.learn(total_timesteps=config["train"]["total_timesteps"])

    os.makedirs("results/manual", exist_ok=True)
    model.save("results/manual/ppo_manual_reward")

    env.close()


if __name__ == "__main__":
    main()