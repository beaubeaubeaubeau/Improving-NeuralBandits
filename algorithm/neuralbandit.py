import numpy as np
import torch
import torch.nn as nn

from exploration.epsilon_greedy import epsilon_greedy_probabilities
from exploration.thompson_sampling import ThompsonSampling

class NeuralBandit1:
    def __init__(self, n_features, n_arms, n_hidden=25, lr=0.01, gamma=0.005, seed=None, exploration="epsilon_greedy"):
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.K = n_arms
        self.gamma = gamma
        self.exploration = exploration
        self.thompson = None
        if exploration == "thompson":
            self.thompson = ThompsonSampling(
                n_features=n_features,
                n_arms=n_arms,
                gamma=gamma,
            )
        self.nets = [
            nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.Sigmoid(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_arms)
        ]
        self.opts = [torch.optim.SGD(net.parameters(), lr=lr) for net in self.nets]
        self.loss_fn = nn.MSELoss()

    def _probs(self, x):
        with torch.no_grad():
            preds = np.array([net(x).item() for net in self.nets])

        if self.thompson is None:
            probabilities = epsilon_greedy_probabilities(preds, self.gamma)
        else:
            probabilities = self.thompson.probabilities(preds, x.detach().cpu().numpy(), self.rng)
        return preds, probabilities

    def select(self, context):
        x = torch.as_tensor(context, dtype=torch.float32)
        _, p = self._probs(x)
        k = self.rng.choice(self.K, p=p)
        return k, p[k]

    def update(self, context, arm, reward, prob):
        """Gradient step on arm's network only, scaled by 1 / P_t(arm)."""
        x = torch.as_tensor(context, dtype=torch.float32)
        y = torch.tensor([float(reward)])
        net, opt = self.nets[arm], self.opts[arm]
        opt.zero_grad()
        pred = net(x)
        loss = self.loss_fn(pred, y) / prob 
        loss.backward()
        opt.step()

        if self.thompson is not None:
            self.thompson.update(context, arm, reward, pred.detach().item())

