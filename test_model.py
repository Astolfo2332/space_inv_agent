from PIL import Image
import numpy as np
import mlflow
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import ale_py


from torchvision.models import mobilenet_v2

from tqdm.notebook import tqdm
import mlflow
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import AtariPreprocessing

from sb3_contrib import QRDQN
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
gym.register_envs(ale_py)
class CustomPenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None
        self.same_action_count = 0
        self.last_lives = None
        self.no_shoot_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        self.same_action_count = 0
        self.no_shoot_count = 0

        if "lives" in info:
            self.last_lives = info["lives"]
        else:
            self.last_lives = 3

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

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
            reward -= 1.0

        # --- Penalty for getting hit
        if "lives" in info and self.last_lives is not None:
            if info["lives"] < self.last_lives:
                reward -= 0.5
            self.last_lives = info["lives"]

        # --- Penalty for not shooting for too long
        if int(action) in [1, 4, 5]:  # shooting actions
            self.no_shoot_count = 0
        else:
            self.no_shoot_count += 1
            if self.no_shoot_count >= 3:
                reward -= 0.7

        reward = np.clip(reward, -1.0, 1.0)

        return obs, reward, terminated, truncated, info


class NormalizeInput(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

env_name = 'SpaceInvadersNoFrameskip-v4'
env = gym.make(env_name)
#En este caso el AtariWrapper hace lo mismo que Clipreward además de añadir el preprocesado de las imágenes
#Normalizamos las imágenes a 0-1
# env = NormalizeInput(env)
env = AtariWrapper(env)
obs, _ = env.reset()
print("Observación inicial:", obs.shape)

normal_env = NormalizeInput(env)
pen_env = CustomPenaltyWrapper(normal_env)  # Añadimos el wrapper de penalización

#Train env con penalizaciones
env = Monitor(pen_env)
env = DummyVecEnv([lambda: env])  # Convertimos a un entorno vectorizado
# # # Se crea el entorno de vectores y se apilan los frames
env = VecFrameStack(env, 4)

#Test env sin penalizaciones
normal_env = Monitor(normal_env)
normal_env = DummyVecEnv([lambda: normal_env])  # Convertimos a un entorno vectorizado
# # # Se crea el entorno de vectores y se apilan los frames
normal_env = VecFrameStack(normal_env, 4)

np.random.seed(123)
obs = env.reset()
nb_actions = env.action_space.n
# print(env.shape)
print(obs.shape)
print(nb_actions)
print("maximo de altura", max(obs[0, :, 0, :].flatten()))
print("maximo de ancho", max(obs[0, 0 :, :].flatten()))


def main():
    test_env_name = "SpaceInvaders-v4"
    test_env = gym.make(test_env_name, render_mode="human")
    obs,_ = test_env.reset()

    model = PPO.load("models/deep_mind_PPO.zip", env=env)

    done = False
    obs = env.reset()
    rewards = []

    try:
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            _, test_reward, _, _, _ = test_env.step(action[0])
            rewards.append(np.clip(test_reward, -1.0, 1.0))
            test_env.render()
    finally:
        print("Total reward:", sum(rewards))
        test_env.close()

if __name__ == "__main__":
    main()