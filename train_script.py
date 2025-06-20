from PIL import Image
import numpy as np
import mlflow
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback

import gymnasium as gym
import ale_py


from torchvision.models import mobilenet_v2

from tqdm import tqdm
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import AtariWrapper

from sb3_contrib import QRDQN

def make_env():
    gym.register_envs(ale_py)

    env_name = 'SpaceInvaders-v0'
    env = gym.make(env_name)
    #En este caso el AtariWrapper hace lo mismo que Clipreward además de añadir el preprocesado de las imágenes
    env = AtariWrapper(env)

    np.random.seed(123)
    obs, info = env.reset(seed=123)
    nb_actions = env.action_space.n
    print(nb_actions)
    print(obs.shape)
    return env


class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        self.last_timesteps = 0

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="step")

    def _on_step(self):
        steps_since_last = self.num_timesteps - self.last_timesteps
        self.progress_bar.update(steps_since_last)
        self.last_timesteps += 1

        # Optional: log latest reward if available
        infos = self.locals.get("infos", [])
        if infos and isinstance(infos[0], dict) and "episode" in infos[0]:
            self.progress_bar.set_postfix(reward=infos[0]["episode"]["r"])
        return True  # Return True to continue training

    def _on_training_end(self):
        self.progress_bar.close()

class MLflowCallback(BaseCallback):
    def __init__(self, best_model_path, experiment_name="SB3_Experiment", run_name=None, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.experiment_name = experiment_name
        self.log_freq = log_freq
        self.step_count = 0
        self.best_mean_reward = -np.inf
        self.best_model_path = best_model_path

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_freq == 0:
            rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []

            mean_reward = np.mean(rewards) if rewards else 0.0
            max_reward = np.max(rewards) if rewards else 0.0
            min_reward = np.min(rewards) if rewards else 0.0
            mean_length = np.mean(lengths) if lengths else 0.0
            std_reward = np.std(rewards) if rewards else 0.0
            exploration_mean = self.model.exploration_rate
            print(self.model.logger.name_to_value.keys())

            step = self.num_timesteps
            mlflow.log_metric("timesteps", step, step=step)
            mlflow.log_metric("episode_reward_mean", mean_reward, step=step)
            mlflow.log_metric("episode_reward_max", max_reward, step=step)
            mlflow.log_metric("episode_reward_min", min_reward, step=step)
            mlflow.log_metric("episode_length_mean", mean_length, step=step)
            mlflow.log_metric("episode_reward_std", std_reward, step=step)
            mlflow.log_metric("episode_length_std", std_reward, step=step)
            mlflow.log_metric("exploration_rate", exploration_mean, step=step)
            # mlflow.log_metric("loss", loss, step=step)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Save the best model
                self.model.save(self.best_model_path, include=["replay_buffer"])
        return True

    def _on_training_end(self):
        # Optionally save the model as artifact
        mlflow.log_param("num_episodes", len(self.model.ep_info_buffer))
        mlflow.end_run()

class TestCallBack(BaseCallback):
    def __init__(self, env, n_episodes=100, verbose=0, test_timesteps=10000):
        super().__init__(verbose)
        self.env = env
        self.n_episodes = n_episodes
        self.rewards = []
        self.test_timesteps = test_timesteps
    def _on_step(self) -> bool:
        if self.num_timesteps % self.test_timesteps == 0:  # Test every 1000 steps
            for _ in range(self.n_episodes):
                ep_reward = 0
                obs, _ = self.env.reset()
                done = False
                while not done:
                    with torch.no_grad():
                        action, _ = self.model.predict(obs)
                    obs, reward, done, _, _ = self.env.step(action)
                    ep_reward += reward
                self.rewards.append(ep_reward)
            mean_reward = np.mean(self.rewards)
            std_reward = np.std(self.rewards)
            mlflow.log_metric("test_reward", mean_reward, step=self.num_timesteps)
            mlflow.log_metric("test_reward_std", std_reward, step=self.num_timesteps)

def train_script():
    env_name = 'SpaceInvaders-v0'
    env = make_env()
    policy_kwargs = dict(
        net_arch = [256, 256],
    )

    total_timesteps = 20_000_000
    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps)
    ml_callback = MLflowCallback(
        best_model_path="models/deep_mind_finetuning.zip",
        experiment_name="DQN_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=250000
    )

    experiment_name = "DQN_SpaceInvaders"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="DQN_Run_deep_mind_finetuned"):
        mlflow.log_param("env_name", env_name)
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("learning_rate", 25e-5)
        mlflow.log_param("buffer_size", 1_000_000)
        mlflow.log_param("exploration_fraction", 0.1)
        mlflow.log_param("exploration_final_eps", 0.05)
        mlflow.log_param("exploration_initial_eps", 1.0)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("gamma", 0.95)
        mlflow.log_param("learning_starts", 10000)
        mlflow.log_param("target_update_interval", 20000)

        model = DQN("CnnPolicy", env, verbose=0,
                    policy_kwargs=policy_kwargs,
                    learning_rate=25e-5, buffer_size=1_000_000,
                    exploration_fraction=0.1, exploration_initial_eps=1.0,
                    exploration_final_eps=0.05, target_update_interval=1000,
                    batch_size=32, learning_starts=100_000, gamma=0.95,seed=23)

        t_model = model.learn(total_timesteps=total_timesteps, callback=[progress_bar_callback, ml_callback, TestCallBack(env, test_timesteps=200_000)])

if __name__ == "__main__":
    train_script()