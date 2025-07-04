import mlflow
import gymnasium as gym
from stable_baselines3 import A2C

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from src.train.utils.callbacks import TQDMProgressCallback, MLflowCallback, TestCallBack
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def train_script_a2c():
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 20_000_000

    params = {
        "learning_rate": 1e-3,
        "exp_name": "deepmind_zoo_imp_a2c_continue_v2",
        "file_name": "models/deepmind_zoo_imp_a2c_continue_v2.zip",
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "ent_coef": 0.01,
        "vf_coef": 0.25,
        "normalize": False,
        "env_var": 32,
        "n_stack": 1,
        "frame_skip": 4
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

    dummy_env = make_vec_env(make_env(), n_envs=params["env_var"], vec_env_cls=SubprocVecEnv)
    dummy_env.seed(23)
    test_env_vec_stack = VecFrameStack(dummy_env, n_stack=params["n_stack"])
    test_env_vec = VecTransposeImage(test_env_vec_stack)

    obs = dummy_env_pen.reset()
    print("test_env", obs.shape)

    obs = test_env_vec.reset()
    print("train_env", obs.shape)
    # env = make_env_original()

    model = A2C.load("models/deepmind_zoo_imp_a2c_continue.zip", env=test_env_vec)

    a2c_kwargs = {
        "optimizer_class": RMSpropTFLike,
        "optimizer_kwargs": {
            "eps": 1e-5,
        }
    }


    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps // params["env_var"])
    ml_callback = MLflowCallback(
        best_model_path=params["file_name"],
        experiment_name="A2C_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=50_000)
    test_callback = TestCallBack(dummy_env_pen, test_timesteps=500_000, save_path=params["file_name"])

    experiment_name = "A2C_SpaceInvaders"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"A2C_Run_{params['exp_name']}"):
        mlflow.log_params(params)

        # model = A2C(
        #     "CnnPolicy",
        #     test_env_vec,
        #     # learning_rate=params["learning_rate"],
        #     ent_coef=params["ent_coef"],
        #     vf_coef=params["vf_coef"],
        #     policy_kwargs=a2c_kwargs,
        #     seed=23,
        #     verbose=1,
        # )

        t_model = model.learn(total_timesteps=total_timesteps,
                              callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000)

        del model
        del t_model
