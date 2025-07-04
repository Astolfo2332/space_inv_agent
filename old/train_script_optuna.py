import optuna

import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecEnvWrapper, VecMonitor, \
    VecTransposeImage, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import ale_py

from tqdm.auto import tqdm
import mlflow

from stable_baselines3.common.atari_wrappers import AtariWrapper

from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


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
            # sample_input = self._preprocess(sample_input)  # Preprocess the input
            n_flatten = self.cnn(sample_input).shape[1]

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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)

class NormalizeInput(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

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

            step = self.num_timesteps
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
    def __init__(self, env, n_episodes=100, verbose=0, test_timesteps=10000, save_path=None, trial=None):
        super().__init__(verbose)
        self.env = env
        self.n_episodes = n_episodes
        self.rewards = []
        self.q_values = []
        self.test_timesteps = test_timesteps
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.trial = trial
        self.acumulative_rewards = []

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
            self.acumulative_rewards.append(mean_reward)
            total_actions = sum(action_counter.values())
            for action, count in action_counter.items():
                mlflow.log_metric(f"action_{action}_count", count, step=self.num_timesteps)
                mlflow.log_metric(f"action_{action}_percentage", count / total_actions, step=self.num_timesteps)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.save_path:
                    self.model.save(self.save_path.replace(".zip", "_best_test.zip"))

            if self.trial:
                self.trial.report(mean_reward, step=self.num_timesteps)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        return True

def objective(trial: optuna.Trial):
    gym.register_envs(ale_py)
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 20_000

    frame_skips = trial.suggest_categorical("frame_skips", [4, 8, 12])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical("buffer_size", [350_000, 450_000, 1_000_000])
    batch_size = trial.suggest_categorical("batch_size", [32])
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    net_arch = trial.suggest_categorical("net_arch", ["[512, 216]", "[512]"])
    optimizers = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])


    def make_env(frame_skip=frame_skips):
        def _init(frame_skip=frame_skips) -> gym.Env:
            normal_env = gym.make(env_name)
            normal_env.action_space.seed(23)
            normal_env = NormalizeInput(normal_env)
            normal_env = AtariWrapper(normal_env, frame_skip=frame_skips)
            normal_env = Monitor(normal_env)
            return normal_env
        return _init

    dummy_env = DummyVecEnv([make_env(frame_skip=frame_skips)])
    dummy_env.seed(23)
    normal_env_vec = VecFrameStack(dummy_env, n_stack=4)

    obs = normal_env_vec.reset()
    print(obs.shape)
    policy_kwargs = dict(
        features_extractor_class=DeepMindCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512] if net_arch == "[512]" else [512, 216],
        optimizer_class=torch.optim.Adam if optimizers == "Adam" else torch.optim.RMSprop
    )

    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps)

    ml_callback = MLflowCallback(
        best_model_path=f"models/optuna_opti/custom_DeepMind_{trial.number}.zip",
        experiment_name="DQN_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=25_000)
    test_callback = TestCallBack(normal_env_vec, test_timesteps=10_000,
                                 save_path=f"models/optuna_opti/custom_DeepMind_{trial.number}.zip", trial=trial)

    experiment_name = "DQN_SpaceInvaders_Optuna"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"DQN_Run_custom_DeepMind_Optuna_{trial.number}"):
        mlflow.log_params({
            "frame_skips": frame_skips,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "exploration_final_eps": exploration_final_eps,
            "net_arch": net_arch,
            "total_timesteps": total_timesteps,
            "optimizer": optimizers
        })

        model = DQN("CnnPolicy", normal_env_vec, policy_kwargs=policy_kwargs, learning_rate=learning_rate,
                    buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, exploration_fraction=0.01,
                    exploration_final_eps=exploration_final_eps, learning_starts=10_000, target_update_interval=10_000)

        t_model = model.learn(total_timesteps=total_timesteps, callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000, reset_num_timesteps=False)

        del model
        del t_model

        top_10 = sorted(test_callback.acumulative_rewards, reverse=True)[:10]
        top_10_mean = np.mean(top_10)
        top_10_std = np.std(top_10)
        
        return top_10_mean - top_10_std

def objective_a2c(trial: optuna.Trial):
    gym.register_envs(ale_py)
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 10_000_000


    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.01, 0.5, log=True)
    lr_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    eps = trial.suggest_float("eps", 1e-6, 1e-4, log=True)

    params = {
        "learning_rate": lr_rate,
        "eps": eps,
        "exp_name": f"deepmind_zoo_imp_a2c_optuna_{trial.number}",
        "file_name": f"models/optuna_opti/a2c/deepmind_zoo_imp_a2c_optuna_{trial.number}.zip",
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
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
    normalize_env = NormalizeInput(base_env)
    # pen_env = CustomPenaltyWrapper(normalize_env, params["penalized_repeat"])
    pen_env = AtariWrapper(base_env, frame_skip=params["frame_skip"])
    pen_env = Monitor(pen_env)

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

    a2c_kwargs = {
        "optimizer_class": RMSpropTFLike,
        "optimizer_kwargs": {
            "eps": eps,
        }
    }


    progress_bar_callback = TQDMProgressCallback(total_timesteps=total_timesteps // params["env_var"])
    ml_callback = MLflowCallback(
        best_model_path=params["file_name"],
        experiment_name="A2C_SpaceInvaders",
        run_name="DQN_Run",
        log_freq=50_000)
    test_callback = TestCallBack(dummy_env_pen, test_timesteps=500_000, save_path=params["file_name"], trial=trial)

    experiment_name = "A2C_SpaceInvaders_optuna"
    exist_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not exist_experiment:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"A2C_Run_{params['exp_name']}"):
        mlflow.log_params(params)

        model = A2C(
            "CnnPolicy",
            test_env_vec,
            learning_rate=lr_rate,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=a2c_kwargs,
            seed=23,
            verbose=1,
        )

        t_model = model.learn(total_timesteps=total_timesteps,
                              callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000)

        del model
        del t_model
        top_10 = sorted(test_callback.acumulative_rewards, reverse=True)[:10]
        top_10_mean = np.mean(top_10)
        top_10_std = np.std(top_10)

        return top_10_mean - top_10_std


if __name__ == "__main__":
    from optuna.trial import TrialState
    from optuna.storages import JournalStorage, JournalFileStorage

    os.makedirs("optuna_storage", exist_ok=True)


    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name="a2c_optimization",
        direction="maximize",
        storage="sqlite:///optuna_storage/a2c_spaceinvaders.db",
        load_if_exists=True,
        pruner=pruner
    )
    # finished_trials = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    # failed_trials = [t for t in study.trials if t.state in (TrialState.FAIL, TrialState.RUNNING)]
    # running_trials = [t for t in study.trials if t.state == TrialState.RUNNING]
    #
    #
    # for trial in running_trials:
    #     study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
    #
    # for trial in failed_trials:
    #     study.enqueue_trial(trial.params)

    n_trials_target = 20
    # n_trials_remaining = n_trials_target - len(finished_trials)
    study.optimize(objective_a2c, n_trials=n_trials_target, n_jobs=3)