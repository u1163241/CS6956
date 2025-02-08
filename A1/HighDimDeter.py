import torch.nn as nn


class HighDeterministicNN(nn.Module):
    def __init__(self):
        super(HighDeterministicNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
