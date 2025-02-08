import torch
import torch.nn as nn
from torch.distributions import Normal


class MDN(nn.Module):
    def __init__(self, input_dim=13, output_dim=1, n_components=5, hidden_dim=64):
        super(MDN, self).__init__()
        self.n_components = n_components
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.pi_layer = nn.Linear(hidden_dim, n_components)
        self.mu_layer = nn.Linear(hidden_dim, n_components * output_dim)
        self.sigma_layer = nn.Linear(hidden_dim, n_components * output_dim)

    def forward(self, x):
        h = self.hidden(x)
        pi = torch.softmax(self.pi_layer(h), dim=1)
        mu = self.mu_layer(h).view(-1, self.n_components, 1)
        sigma = torch.exp(self.sigma_layer(h)).view(-1, self.n_components, 1)
        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, y):
    y = y.unsqueeze(1).expand_as(mu)
    normal_dists = Normal(mu, sigma)
    probs = torch.exp(normal_dists.log_prob(y))
    weighted_probs = pi * probs.squeeze()
    loss = -torch.log(torch.sum(weighted_probs, dim=1) + 1e-8)
    return loss.mean()
