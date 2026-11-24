from algorithm.neuralbandit import NeuralBandit1
from exploration.epsilon_greedy import epsilon_greedy_probabilities
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

if __name__ == "__main__":
    seed = 0
    n = 10000
    report_every = 1000

    covertype = fetch_ucirepo(id=31)

    X = covertype.data.features.to_numpy(dtype=np.float32)
    y = covertype.data.targets.to_numpy().reshape(-1).astype(np.int64) - 1
    
    n_arms = 7

    bandit = NeuralBandit1(
        n_features=X.shape[1],
        n_arms=n_arms,
        n_hidden=25,
        lr=0.01,
        gamma=0.05,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))[:n]

    mean = np.zeros(10)
    sum_squared_deviations = np.zeros(10)
    correct_predictions = 0

    for t, index in enumerate(order, start=1):
        context = X[index].copy()

        values = context[:10].copy()
        delta = values - mean
        mean += delta / t
        sum_squared_deviations += delta * (values - mean)
        std = np.sqrt(sum_squared_deviations / t)
        context[:10] = (values - mean) / np.maximum(std, 1.0)

        arm, probability = bandit.select(context)
        correct = int(arm == y[index])
        bandit.update(context, arm, correct, probability)

        correct_predictions += correct
        if t % report_every == 0 or t == len(order):
            print(
                f"round={t:6d}  "
                f"classification accuracy={correct_predictions / t:.3f}  "
                f"cumulative regret={t - correct_predictions}"
            )

    print(f"Uniform random baseline: {1 / n_arms:.3f}")
    
