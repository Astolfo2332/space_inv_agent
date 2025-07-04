from src.train.train_ppo import train_ppo
from src.train.train_a2c import train_script_a2c
from src.train.train_dqn import train_script_dqn
from src.train.train_qrdqn import train_script_qrdqn

if __name__ == "__main__":
    # Aquí se cambia para entrenar el modelo que se quiera
    train_ppo()