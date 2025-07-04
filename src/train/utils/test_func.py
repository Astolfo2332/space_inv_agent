import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import QRDQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

def test_env():
    env_name = 'SpaceInvadersNoFrameskip-v4'
    params = {
        "frame_skip": 4,
        "n_stack": 1,
    }
    def make_env_no_pen():
        def _init() -> gym.Env:
            normal_env = gym.make(env_name)
            normal_env.action_space.seed(23)
            # normal_env = NormalizeInput(normal_env)
            normal_env = AtariWrapper(normal_env, frame_skip=params["frame_skip"])
            return normal_env
        return _init

    dummy_env_pen = DummyVecEnv([make_env_no_pen()])
    dummy_env_pen.seed(23)
    dummy_env_pen = VecFrameStack(dummy_env_pen, n_stack=params["n_stack"])
    dummy_env_pen = VecTransposeImage(dummy_env_pen)

    model = A2C.load("models/deepmind_zoo_imp_a2c_continue_v2.zip", env=dummy_env_pen)

    total_rewards = []
    total_actions = {}
    for i in range(100):
        episode_rewards = []
        obs = dummy_env_pen.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = dummy_env_pen.step(action)
            episode_rewards.append(reward)
            total_actions[action[0]] = action[0]
        total_rewards.append(np.sum(episode_rewards))

    print("Mean reward:", np.mean(total_rewards))
    print("Std reward:", np.std(total_rewards))
    print("Max reward:", np.max(total_rewards))
    print("Min reward:", np.min(total_rewards))
    total_actions_num = sum(total_actions.values())
    for action, count in total_actions.items():
        print(f"Action {action} count: {count}, percentage: {count / total_actions_num:.2%}")
