from PIL import Image
import numpy as np
import mlflow
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_atari_env
from gymnasium.wrappers import ClipReward

import gymnasium as gym
import ale_py
from torchvision.models import mobilenet_v2
from tqdm.auto import tqdm
import mlflow
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import AtariWrapper

from collections import defaultdict


def make_env_original():
    gym.register_envs(ale_py)
    class CustomPenaltyWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.last_action = None
            self.same_action_count = 0
            self.last_lives = None
            self.no_shoot_count = 0
            self.no_score_count = 0
            self.killing_streak = 0
            self.tolerance_killing_streak = 5  # Tolerance for killing streak
            self.survival_reward = 0

            self.entropy_scale = 0.00001
            self.action_counts = defaultdict(int)
            self.total_actions = 0

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.last_action = None
            self.same_action_count = 0
            self.no_shoot_count = 0
            self.no_score_count = 0
            self.killing_streak = 0
            self.tolerance_killing_streak = 5  # Reset tolerance
            self.survival_reward = 0

            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # print("initial_reward:", reward)

            self.action_counts[action] += 1
            self.total_actions += 1

            p = self.action_counts[action] ** 2 / self.total_actions

            reward -= self.entropy_scale * p

            if reward == 0:
                self.no_score_count += 1
                self.tolerance_killing_streak -= 1
                if self.tolerance_killing_streak == 0:
                    self.killing_streak = 0
                    self.tolerance_killing_streak = 5  # Reset tolerance
            else:
                self.no_score_count = 0
                self.killing_streak += 1

            # reward = reward * self.killing_streak

            self.survival_reward += 1
            if self.survival_reward > 50:
                reward += 0.05
                self.survival_reward = 0

            if self.no_score_count >= 5:
                reward -= 0.00005

            # --- Penalty for repeated no-op
            if action == 0:
                if self.last_action == 0:
                    self.same_action_count += 1
                else:
                    self.same_action_count = 1
            else:
                self.same_action_count = 0
            self.last_action = action

            if self.same_action_count >= 3:
                reward -= 0.00005

            # --- Penalty for not shooting for too long
            if int(action) in [1, 4, 5]:  # shooting actions
                self.no_shoot_count = 0
            else:
                self.no_shoot_count += 1
                if self.no_shoot_count >= 10:
                    reward -= 0.00007

            reward = np.clip(reward, -1.0, 1.0)
            # reward = np.tanh(reward)

            return obs, reward, terminated, truncated, info



    # Se usa la version sin skip de frames ya que el AtariWrapper lo hace
    env_name = 'SpaceInvadersNoFrameskip-v4'
    env = gym.make(env_name)
    # En este caso el AtariWrapper hace lo mismo que Clipreward además de añadir el preprocesado de las imágenes
    # Normalizamos las imágenes a 0-1
    # env = NormalizeInput(env)
    env = AtariWrapper(env, frame_skip=4)
    normal_env_real = NormalizeInput(env)
    pen_env = CustomPenaltyWrapper(normal_env_real)  # Añadimos el wrapper de penalización

    # Train env con penalizaciones
    env = Monitor(pen_env)
    env = DummyVecEnv([lambda: env])  # Convertimos a un entorno vectorizado
    # # # Se crea el entorno de vectores y se apilan los frames
    env = VecFrameStack(env, 4)

    # Test env sin penalizaciones
    normal_env = Monitor(normal_env_real)
    normal_env = DummyVecEnv([lambda: normal_env])  # Convertimos a un entorno vectorizado
    # # # Se crea el entorno de vectores y se apilan los frames
    normal_env = VecFrameStack(normal_env, 4)

    return normal_env


class NormalizeInput(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

class CustomPenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_counter = defaultdict(int)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_counter.clear()  # Reset action counter
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update action counter
        self.action_counter[action] += 1

        # Calculate entropy-based penalty
        total_actions = sum(self.action_counter.values())
        action_percentage = self.action_counter[action] / total_actions
        if action_percentage >= 0.5 and total_actions > 10:
            reward -= 0.025

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
            rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer] if self.model.ep_info_buffer else []

            mean_reward = np.mean(rewards) if rewards else 0.0
            max_reward = np.max(rewards) if rewards else 0.0
            min_reward = np.min(rewards) if rewards else 0.0
            mean_length = np.mean(lengths) if lengths else 0.0
            std_reward = np.std(rewards) if rewards else 0.0
            exploration_mean = self.model.exploration_rate
            loss_mean = self.logger.name_to_value.get("train/loss", 0)
            training_updates = self.logger.name_to_value.get("train/n_updates", 0)

            step = self.num_timesteps
            mlflow.log_metric("timesteps", step, step=step)
            mlflow.log_metric("episode_reward_mean", mean_reward, step=step)
            mlflow.log_metric("episode_reward_max", max_reward, step=step)
            mlflow.log_metric("episode_reward_min", min_reward, step=step)
            mlflow.log_metric("episode_length_mean", mean_length, step=step)
            mlflow.log_metric("episode_reward_std", std_reward, step=step)
            mlflow.log_metric("episode_length_std", std_reward, step=step)
            mlflow.log_metric("exploration_rate", exploration_mean, step=step)
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

class DeepMindCNN(BaseFeaturesExtractor):
    """
    DeepMind-style CNN used in the original DQN paper (Mnih et al., 2015).
    Input shape: (n_stack, 84, 84) → (4, 84, 84)
    """

    def __init__(self, observation_space, features_dim=512):
        # features_dim is the output of the last linear layer (fc1)
        super().__init__(observation_space, features_dim)

        # Check input shape
        n_input_channels = observation_space.shape[0]  # e.g., 4 stacked grayscale frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                 # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),                 # (64, 7, 7)
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # Modification
        self.linear = nn.Sequential(
            nn.Linear(n_flatten , n_flatten // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_flatten // 2, n_flatten // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_flatten // 4, features_dim),
            nn.ReLU()
        )
        # Original
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)

def train_script():
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 10_000_000
    def make_env():
        def _init() -> gym.Env:
            normal_env = gym.make(env_name)
            normal_env.action_space.seed(23)
            normal_env = NormalizeInput(normal_env)
            normal_env = CustomPenaltyWrapper(normal_env)
            normal_env = AtariWrapper(normal_env, frame_skip=4)
            normal_env = Monitor(normal_env)
            return normal_env
        return _init

    def make_env_no_pen():
        def _init() -> gym.Env:
            normal_env = gym.make(env_name)
            normal_env.action_space.seed(23)
            normal_env = NormalizeInput(normal_env)
            normal_env = AtariWrapper(normal_env, frame_skip=4)
            normal_env = Monitor(normal_env)
            return normal_env
        return _init

    dummy_env = DummyVecEnv([make_env()])
    dummy_env.seed(23)
    normal_env_vec = VecFrameStack(dummy_env, n_stack=4)

    dummy_env = DummyVecEnv([make_env_no_pen()])
    dummy_env.seed(23)
    normal_env_vec_no_pen = VecFrameStack(dummy_env, n_stack=4)

    obs = normal_env_vec.reset()
    print(obs.shape)

    policy_kwargs = dict(
        features_extractor_class=DeepMindCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 216],
        optimizer_class=torch.optim.RMSprop,
        optimizer_kwargs=dict(eps=1e-6, alpha=0.99, momentum=0),
    )

    # env = make_env_original()

    # model = DQN.load("models/custom_DeepMind_v2_8_RMS_lastest.zip", env=env)

    params = {
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "learning_rate": 1e-6,
        "buffer_size": 1_000_000,
        "gamma": 0.99,
        "file_name": "models/custom_DeepMind_v2_4_RMS_hsuanchuu_pen.zip",
        "exp_name": "custom_DeepMind_v2_4_RMS_hsuanchuu_pen",
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "batch_size": 32,
        "learning_starts": 100_000,
        "target_update_interval": 5_000,
        "optimizer_class": "RMSprop",
        "penalized_repeat": 0.025
    }

    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps)
    ml_callback = MLflowCallback(
        best_model_path=params["file_name"],
        experiment_name="DQN_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=50_000)
    test_callback = TestCallBack(normal_env_vec_no_pen, test_timesteps=100_000, save_path=params["file_name"])

    experiment_name = "DQN_SpaceInvaders"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"DQN_Run_{params['exp_name']}"):
        mlflow.log_params(params)
        model = DQN(
            "CnnPolicy",
            normal_env_vec,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            target_update_interval=params["target_update_interval"],
            exploration_fraction=params["exploration_fraction"],
            exploration_initial_eps=params["exploration_initial_eps"],
            exploration_final_eps=params["exploration_final_eps"],
            policy_kwargs=policy_kwargs,
            seed=23,
        )

        t_model = model.learn(total_timesteps=total_timesteps, callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000, reset_num_timesteps=False)

        del model
        del t_model


if __name__ == "__main__":
    train_script()