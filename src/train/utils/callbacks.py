import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from tqdm.auto import tqdm
import mlflow
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

class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0, inital=None):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        self.inital = inital

    def _on_training_start(self):
        if self.inital is None:
            self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="step")
        else:
            self.progress_bar = tqdm(total=self.total_timesteps, initial=self.inital, desc="Training Progress", unit="step")

    def _on_step(self):
        self.progress_bar.update(1)
        # Optional: log latest reward if available
        rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
        if rewards:
            self.progress_bar.set_postfix(reward=np.mean(rewards), std=np.std(rewards))
        return True  # Return True to continue training

    def _on_training_end(self):
        self.progress_bar.close()
class MLflowCallback(BaseCallback):
    def __init__(self, best_model_path, experiment_name="SB3_Experiment", run_name=None, log_freq=1000, verbose=0, save_freq=100_000):
        super().__init__(verbose)
        self.experiment_name = experiment_name
        self.log_freq = log_freq
        self.step_count = 0
        self.best_mean_reward = -np.inf
        self.best_model_path = best_model_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        self.step_count = self.num_timesteps
        # print(self.step_count % self.log_freq)
        if self.step_count % self.log_freq == 0:

            step = self.num_timesteps

            rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []

            mean_reward = np.mean(rewards) if rewards else 0.0
            max_reward = np.max(rewards) if rewards else 0.0
            min_reward = np.min(rewards) if rewards else 0.0
            mean_length = np.mean(lengths) if lengths else 0.0
            std_reward = np.std(rewards) if rewards else 0.0
            if hasattr(self.model, "exploration_rate"):
                exploration_mean = self.model.exploration_rate
                mlflow.log_metric("exploration_rate", exploration_mean, step=step)

            loss_mean = self.logger.name_to_value.get("train/loss", 0)
            training_updates = self.logger.name_to_value.get("train/n_updates", 0)

            mlflow.log_metric("timesteps", step, step=step)
            mlflow.log_metric("episode_reward_mean", mean_reward, step=step)
            mlflow.log_metric("episode_reward_max", max_reward, step=step)
            mlflow.log_metric("episode_reward_min", min_reward, step=step)
            mlflow.log_metric("episode_length_mean", mean_length, step=step)
            mlflow.log_metric("episode_reward_std", std_reward, step=step)
            mlflow.log_metric("episode_length_std", std_reward, step=step)
            if loss_mean != 0:
                mlflow.log_metric("loss_mean", loss_mean, step=step)
            if training_updates != 0:
                mlflow.log_metric("training_updates", training_updates, step=step)

            if mean_reward > self.best_mean_reward and self.step_count % self.save_freq == 0:
                self.best_mean_reward = mean_reward
                # Save the best model
                self.model.save(self.best_model_path)
        if self.step_count % 1_000_000 == 0:
            # Log the model as an artifact
            self.model.save(self.best_model_path.replace(".zip", "_lastest.zip"))

        return True

    def _on_training_end(self):
        # Optionally save the model as artifact
        mlflow.log_param("num_episodes", len(self.model.ep_info_buffer))
        mlflow.end_run()
from collections import Counter

class TestCallBack(BaseCallback):
    def __init__(self, env, n_episodes=100, verbose=0, test_timesteps=10000, save_path=None):
        super().__init__(verbose)
        self.env = env
        self.n_episodes = n_episodes
        self.rewards = []
        self.q_values = []
        self.test_timesteps = test_timesteps
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    def _on_step(self) -> bool:
        if self.num_timesteps % self.test_timesteps == 0:  # Test every 1000 steps
            self.rewards = []
            action_counter = Counter()
            for _ in range(self.n_episodes):
                ep_reward = 0
                obs = self.env.reset()
                done = False
                while not done:
                    with torch.no_grad():
                        action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.env.step(action)
                    action_scalar = int(action[0])
                    action_counter[action_scalar] += 1
                    ep_reward += reward
                self.rewards.append(ep_reward)
            mean_reward = np.mean(self.rewards)
            std_reward = np.std(self.rewards)
            mlflow.log_metric("test_reward", mean_reward, step=self.num_timesteps)
            mlflow.log_metric("test_reward_std", std_reward, step=self.num_timesteps)
            mlflow.log_metric("test_reward_max", np.max(self.rewards), step=self.num_timesteps)
            mlflow.log_metric("test_reward_min", np.min(self.rewards), step=self.num_timesteps)
            total_actions = sum(action_counter.values())
            for action, count in action_counter.items():
                mlflow.log_metric(f"action_{action}_count", count, step=self.num_timesteps)
                mlflow.log_metric(f"action_{action}_percentage", count / total_actions, step=self.num_timesteps)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.save_path:
                    self.model.save(self.save_path.replace(".zip", "_best_test.zip"))
        return True
