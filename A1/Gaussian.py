import torch
import torch.nn as nn
from torch.distributions import Normal


class SingleGaussianNN(nn.Module):
    def __init__(self):
        super(SingleGaussianNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        self.mu_layer = nn.Linear(32, 1)
        self.sigma_layer = nn.Linear(32, 1)

    def forward(self, x):
        h = self.hidden(x)
        mu = self.mu_layer(h)
        sigma = torch.exp(self.sigma_layer(h))
        return mu, sigma


def single_gaussian_loss(mu, sigma, y):
    normal_dists = Normal(mu, sigma)
    loss = -normal_dists.log_prob(y).mean()
    return loss
