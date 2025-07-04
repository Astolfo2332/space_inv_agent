import gymnasium as gym

from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3 import DQN, A2C, PPO

def save_a_video():
    env_name = 'SpaceInvadersNoFrameskip-v4'
    params = {
        "frame_skip": 4,
        "n_stack": 4,
    }
    def make_env_no_pen():
        def _init() -> gym.Env:
            normal_env = gym.make(env_name, render_mode="rgb_array")
            normal_env.action_space.seed(23)
            # normal_env = NormalizeInput(normal_env)
            normal_env = AtariWrapper(normal_env, frame_skip=params["frame_skip"])
            return normal_env
        return _init

    dummy_env_pen = DummyVecEnv([make_env_no_pen()])
    dummy_env_pen.seed(23)
    dummy_env_pen = VecFrameStack(dummy_env_pen, n_stack=params["n_stack"])
    dummy_env_pen = VecTransposeImage(dummy_env_pen)

    model = PPO.load("models/deepmind_zoo_imp_ppo_continue_v2.zip", env=dummy_env_pen)

    video_path = "videos"
    eval_env = VecVideoRecorder(
        dummy_env_pen,
        video_folder=video_path,
        record_video_trigger=lambda step: step == 0,
        video_length=10000,
        name_prefix="test-ppo_30M"
    )

    obs = eval_env.reset()
    done = False
    rewards = 0
    lives = 3
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if "lives" in info[0]:
            if info[0]["lives"] < lives:
                lives = info[0]["lives"]
                if done:
                    lives = 3
                done = True
                if lives == 0:
                    lives = 3
        rewards += reward[0]
        if done:
            print(f"Reward: {rewards}")
            done = False
            rewards = 0
        eval_env.step(action)

    print("Rewards:", rewards)

    eval_env.close()

    print(f"Video saved to {video_path}")
