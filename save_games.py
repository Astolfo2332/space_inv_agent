import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from stable_baselines3.common.monitor import Monitor
import numpy as np
import pygame
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper


gym.register_envs(ale_py)
# --- Custom Wrappers ---
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
            self.last_lives = None

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
            print("⚠️  -1.0 for repeated NOOP")
            reward -= 1.0

        # --- Penalty for getting hit
        if "lives" in info and self.last_lives is not None:
            if info["lives"] < self.last_lives:
                print("🟥 -0.5 for losing a life")
                reward -= 0.5
            self.last_lives = info["lives"]

        # --- Penalty for not shooting for too long
        if int(action) in [1, 4, 5]:  # shooting actions
            self.no_shoot_count = 0
        else:
            self.no_shoot_count += 1
            if self.no_shoot_count >= 3:
                print("🚫 -0.7 for not shooting")
                reward -= 0.7

        reward = np.clip(reward, -1.0, 1.0)

        return obs, reward, terminated, truncated, info


def play_game():
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    env_name = 'SpaceInvadersNoFrameskip-v4'
    normal_base_env = gym.make(env_name, render_mode="rgb_array")  # no render_mode here
    base_env = AtariWrapper(normal_base_env)
    wrapped_env = NormalizeInput(CustomPenaltyWrapper(base_env))

    wrapped_env = Monitor(wrapped_env)

    vec_env = DummyVecEnv([lambda: wrapped_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # Pygame Setup
    pygame.init()
    screen_width, screen_height = 210, 160  # Atari native res
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    # Key to action mapping
    KEY_TO_ACTION = {
        pygame.K_SPACE: 1,   # FIRE
        pygame.K_RIGHT: 2,
        pygame.K_LEFT: 3,
        pygame.K_d: 4,       # RIGHTFIRE
        pygame.K_a: 5,       # LEFTFIRE
        pygame.K_RETURN: 0   # NOOP
    }

    current_action = 0
    obs = vec_env.reset()

    buffer = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "next_obs": []
    }

    print("🎮 Controls: SPACE = fire | ←/→ = move | D/A = move+fire | ENTER = NOOP | ESC = quit")

    n_actions = 6

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                print("🛑 Quitting game...")
                running = False

            if event.type == pygame.KEYDOWN:
                current_action = KEY_TO_ACTION.get(event.key, 0)

        # Step env
        one_hot = np.zeros(n_actions)
        one_hot[current_action] = 1
        next_obs, reward, done, info = vec_env.step(np.array([current_action]))
        _, _, _, _, _ = normal_base_env.step(current_action)


        # Render RGB frame from base env (unwrap vec_env)
        frame = normal_base_env.render()  # (H, W, C)
        # Pygame needs (W, H, C)
        frame = np.transpose(frame, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame)

        # Display
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Record transitions
        buffer["obs"].append(obs.copy())
        buffer["actions"].append(one_hot)
        buffer["rewards"].append(reward[0])
        buffer["dones"].append(done[0])
        buffer["next_obs"].append(next_obs.copy())

        obs = next_obs
        current_action = 0

        if done[0]:
            print("🔁 Episode done. Restarting...")
            obs = vec_env.reset()

        clock.tick(30)  # FPS limit

    # Save data on exit
    pygame.quit()
    vec_env.close()
    np.savez_compressed("data/human_penalized_buffer.npz", **buffer)
    print("✅ Saved buffer to data/human_penalized_buffer.npz")

def test_game():
    import gymnasium as gym
    import time

    env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")
    obs, info = env.reset()

    done = False
    while not done:
        print("Acciones posibles: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHT+FIRE, 5=LEFT+FIRE")
        action = int(input("Ingresa acción (0-5): "))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Recompensa: {reward}, Terminado: {terminated}, Truncado: {truncated}, Info: {info}")

    env.close()

if __name__ == "__main__":
    play_game()
