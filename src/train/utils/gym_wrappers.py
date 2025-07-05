import numpy as np
import gymnasium as gym
from collections import defaultdict
class NormalizeInput(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

class CustomPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalized_repeat=3e-7, entropy_coeff=1e-500):
        super().__init__(env)
        self.action_counter = defaultdict(int)
        self.penalized_repeat = penalized_repeat
        self.entropy_coeff = entropy_coeff

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_counter.clear()  # Reset action counter
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update action counter
        self.action_counter[action] += 1
        total_actions = sum(self.action_counter.values())
        probs = np.array(
            [count / total_actions for count in self.action_counter.values()]
        )

        # Entropy: -∑p log p
        entropy = -np.sum(probs * np.log(probs + 1e-8))  # avoid log(0)
        entropy = entropy / 6

        # Reward shaping with entropy
        reward += self.entropy_coeff * entropy

        # Calculate entropy-based penalty
        total_actions = sum(self.action_counter.values())
        action_percentage = self.action_counter[action] / total_actions
        if action_percentage >= 0.5 and total_actions > 10:
            reward = 0

        return obs, reward, terminated, truncated, info
