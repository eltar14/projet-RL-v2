import torch.nn as nn
import torch

class DQN_1(nn.Module):
    """
    Tres simple peut etre trop. Simple MLP
    """
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(DQN_1, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        c, h, w = input_shape  # ex: (4, 10, 10)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),  # (4,10,10) → (16,10,10)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (16,10,10) → (32,10,10)
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # output_dim = nb d'actions
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x