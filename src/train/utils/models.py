import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
            n_flatten = self.cnn(sample_input).shape[1]

        # Modification
        self.linear = nn.Sequential(
            nn.Linear(n_flatten , n_flatten // 2),
            nn.ReLU(),
            nn.Linear(n_flatten // 2, n_flatten // 4),
            nn.ReLU(),
            nn.Linear(n_flatten // 4, features_dim),
            nn.ReLU()
        )
        # Original
        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, features_dim)
        # )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)
