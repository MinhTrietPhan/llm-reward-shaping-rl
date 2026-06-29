import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_weights):
        super().__init__(env)
        self.reward_weights = reward_weights
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        rewards_info = info.get("rewards", {})

        collision = rewards_info.get("collision_reward", 0.0)
        high_speed = rewards_info.get("high_speed_reward", 0.0)
        right_lane = rewards_info.get("right_lane_reward", 0.0)
        on_road = rewards_info.get("on_road_reward", 1.0)

        new_reward = (
            self.reward_weights["collision_penalty"] * abs(collision)
            + self.reward_weights["high_speed_reward"] * high_speed
            + self.reward_weights["right_lane_reward"] * right_lane
            + self.reward_weights["on_road_reward"] * on_road
        )

        info["custom_reward"] = new_reward
        info["original_reward"] = reward

        return obs, new_reward, terminated, truncated, info