import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import highway_env
import yaml
import numpy as np

from stable_baselines3 import PPO
from envs.reward_wrapper import CustomRewardWrapper


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model_path, config_path, episodes=10):
    config = load_config(config_path)

    env = gym.make(config["env_name"])
    env = CustomRewardWrapper(env, config["reward_weights"])

    model = PPO.load(model_path)

    episode_rewards = []
    collision_count = 0
    speeds = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

            speeds.append(info.get("speed", 0))

            if info.get("crashed", False):
                collision_count += 1

        episode_rewards.append(total_reward)

    env.close()

    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "collision_rate": collision_count / episodes,
        "average_speed": float(np.mean(speeds)),
        "episodes": episodes,
    }

    return metrics


if __name__ == "__main__":
    metrics = evaluate(
        model_path="results/manual/ppo_manual_reward",
        config_path="configs/manual_reward.yaml",
        episodes=10,
    )

    print(metrics)