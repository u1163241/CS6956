import torch.nn as nn


class DeterministicNN(nn.Module):
    def __init__(self):
        super(DeterministicNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)
