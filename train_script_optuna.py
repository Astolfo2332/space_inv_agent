import optuna
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from tqdm.auto import tqdm
import mlflow
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py


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
        infos = self.locals.get("infos", [])
        if infos and isinstance(infos[0], dict) and "episode" in infos[0]:
            self.progress_bar.set_postfix(reward=infos[0]["episode"]["r"])
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
    def _on_step(self) -> bool:
        if self.num_timesteps % self.test_timesteps == 0:  # Test every 1000 steps
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

            if self.trial:
                self.trial.report(mean_reward, step=self.num_timesteps)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        return True

def objective(trial: optuna.Trial):
    gym.register_envs(ale_py)
    env_name = 'SpaceInvadersNoFrameskip-v4'
    total_timesteps = 1_000_000

    frame_skips = trial.suggest_categorical("frame_skips", [4, 8, 12])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_categorical("buffer_size", [350_000, 400_000, 450_000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
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
    test_callback = TestCallBack(normal_env_vec, test_timesteps=50_000,
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
            "total_timesteps": total_timesteps
        })

        model = DQN("CnnPolicy", normal_env_vec, policy_kwargs=policy_kwargs, learning_rate=learning_rate,
                    buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, exploration_fraction=0.01,
                    exploration_final_eps=exploration_final_eps, learning_starts=10_000, target_update_interval=10_000)

        t_model = model.learn(total_timesteps=total_timesteps, callback=[progress_bar_callback, ml_callback, test_callback], log_interval=2_000, reset_num_timesteps=False)


if __name__ == "__main__":
    os.makedirs("optuna_storage", exist_ok=True)

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name="dqn_optimization",
        direction="maximize",
        storage="sqlite:///optuna_storage/dqn_spaceinvaders.db",
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(objective, n_trials=20, n_jobs=1)