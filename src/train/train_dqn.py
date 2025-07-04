import mlflow
import gymnasium as gym
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from src.train.utils.callbacks import TQDMProgressCallback, MLflowCallback, TestCallBack
from src.train.utils.models import DeepMindCNN

def train_script_dqn():
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 10_000_000

    params = {
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "learning_rate": 1e-4,
        "buffer_size": 10_000,
        "gamma": 0.99,
        "file_name": "models/custom_DeepMind_v3_4.zip",
        "exp_name": "custom_DeepMind_v3_4",
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "batch_size": 32,
        "learning_starts": 100_000,
        "target_update_interval": 1_000,
        "optimizer_class": "Adam",
        "penalized_repeat": 0.25,
        "n_stack": 4,
        "frame_skip": 4,
        "env_var": 1,
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


    base_env = gym.make(env_name)
    base_env.action_space.seed(23)

    dummy_env_pen = DummyVecEnv([make_env_no_pen()])
    dummy_env_pen.seed(23)
    dummy_env_pen = VecFrameStack(dummy_env_pen, n_stack=params["n_stack"])
    dummy_env_pen = VecTransposeImage(dummy_env_pen)

    test_env = AtariWrapper(base_env, frame_skip=params["frame_skip"])
    test_env_n = Monitor(test_env)
    dummy_env = make_vec_env(make_env(), n_envs=params["env_var"], vec_env_cls=SubprocVecEnv)
    dummy_env.seed(23)
    test_env_vec_stack = VecFrameStack(dummy_env, n_stack=params["n_stack"])
    test_env_vec = VecTransposeImage(test_env_vec_stack)

    obs = dummy_env_pen.reset()
    print("train_env", obs.shape)

    obs = test_env_vec.reset()
    print("test_env", obs.shape)

    policy_kwargs = dict(
        features_extractor_class=DeepMindCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 512],
        # optimizer_class=RMSpropTFLike,
        # optimizer_kwargs=dict(eps=0.01),
    )

    # env = make_env_original()

    # model = DQN.load("models/custom_DeepMind_v2_8_RMS_lastest.zip", env=env)

    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps)
    ml_callback = MLflowCallback(
        best_model_path=params["file_name"],
        experiment_name="DQN_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=50_000)
    test_callback = TestCallBack(dummy_env_pen, test_timesteps=500_000, save_path=params["file_name"])

    experiment_name = "DQN_SpaceInvaders"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"DQN_Run_{params['exp_name']}"):
        mlflow.log_params(params)
        model = DQN(
            "CnnPolicy",
            test_env_vec,
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
            verbose=1
        )

        t_model = model.learn(total_timesteps=total_timesteps,
                              callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000)

        del model
        del t_model
