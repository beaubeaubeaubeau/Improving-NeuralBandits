from exploration.epsilon_greedy import epsilon_greedy_probabilities
import numpy as np
import torch
import torch.nn as nn

class NeuralBandit1:
    def __init__(self, n_features, n_arms, n_hidden=25, lr=0.01, gamma=0.005, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.K = n_arms
        self.gamma = gamma
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
        """Predicted rewards for each arm and the exploration distribution P_t."""
        with torch.no_grad():
            preds = np.array([net(x).item() for net in self.nets])

        p = epsilon_greedy_probabilities(preds, self.gamma)
        return preds, p

    def select(self, context):
        """Return (chosen arm, probability with which it was chosen)."""
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
        loss = self.loss_fn(net(x), y) / prob 
        loss.backward()
        opt.step()

