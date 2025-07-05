import mlflow
import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from src.train.utils.callbacks import TQDMProgressCallback, MLflowCallback, TestCallBack
from stable_baselines3.common.utils import get_linear_fn
import ale_py

def train_ppo():
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 12_000_000

    params = {
        "exp_name": "deepmind_zoo_imp_ppo_continue_v2",
        "file_name": "models/deepmind_zoo_imp_ppo_continue_v2.zip",
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "vf_coef": 0.5,
        "env_var": 8,
        "n_stack": 4,
        "frame_skip": 4,
        "n_steps": 128,
        "batch_size": 256,
        "ent_coef": 0.01,
        "n_epochs": 4,
        "learning_rate": get_linear_fn(2.5e-4, 1e-5, total_timesteps),
        "clip_range": get_linear_fn(0.1, 0.01, total_timesteps)
    }

    def make_env():
        def _init() -> gym.Env:
            normal_env = gym.make(env_name)
            normal_env.action_space.seed(23)
            # normal_env = NormalizeInput(normal_env)
            # normal_env = CustomPenaltyWrapper(normal_env, params["penalized_repeat"])
            normal_env = AtariWrapper(normal_env, frame_skip=params["frame_skip"])
            normal_env = Monitor(normal_env)
            return normal_env
        return _init

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

    dummy_env = make_vec_env(make_env(), n_envs=params["env_var"], vec_env_cls=SubprocVecEnv)
    dummy_env.seed(23)
    test_env_vec_stack = VecFrameStack(dummy_env, n_stack=params["n_stack"])
    test_env_vec = VecTransposeImage(test_env_vec_stack)

    obs = dummy_env_pen.reset()
    print("train_env", obs.shape)

    obs = test_env_vec.reset()
    print("test_env", obs.shape)

    model = PPO.load("models/deepmind_zoo_imp_ppo_continue.zip", env=test_env_vec)


    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps // params["env_var"])
    ml_callback = MLflowCallback(
        best_model_path=params["file_name"],
        experiment_name="A2C_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=50_000)
    test_callback = TestCallBack(dummy_env_pen, test_timesteps=500_000, save_path=params["file_name"])

    experiment_name = "PPO_SpaceInvaders"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"PPO_Run_{params['exp_name']}"):
        mlflow.log_params(params)

        # model = PPO(
        #     "CnnPolicy",
        #     test_env_vec,
        #     clip_range=params["clip_range"],
        #     n_steps=params["n_steps"],
        #     batch_size=params["batch_size"],
        #     n_epochs=params["n_epochs"],
        #     ent_coef=params["ent_coef"],
        #     vf_coef=params["vf_coef"],
        #     learning_rate=params["learning_rate"],
        #     seed=23,
        #     verbose=1,
        # )

        t_model = model.learn(total_timesteps=total_timesteps,
                              callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000)

        del model
        del t_model
