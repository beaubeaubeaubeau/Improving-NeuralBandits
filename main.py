from algorithm.neuralbandit import NeuralBandit1
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed = 0
    n = 10000
    report_every = 1000

    covertype = fetch_ucirepo(id=31)

    X = covertype.data.features.to_numpy(dtype=np.float32)
    y = covertype.data.targets.to_numpy().reshape(-1).astype(np.int64) - 1
    
    n_arms = 7

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))[:n]

    bandits = {
        'Epsilon greedy': NeuralBandit1(
        n_features=X.shape[1],
        n_arms=n_arms,
        n_hidden=25,
        lr=0.01,
        gamma=0.05,
        seed=seed,
        exploration='epsilon'
        ),
        'Thompson sampling': NeuralBandit1(
        n_features=X.shape[1],
        n_arms=n_arms,
        n_hidden=25,
        lr=0.01,
        gamma=0.05,
        seed=seed,
        exploration='thompson'
        ),
    }

    regrets = {
        name: np.zeros(len(order), dtype=np.int64)
        for name in ("Random", *bandits)
    }
    totals = dict.fromkeys(regrets, 0)
    mean = np.zeros(10)
    sum_squared_deviations = np.zeros(10)

    for t, index in enumerate(order, start=1):
        context = X[index].copy()

        values = context[:10].copy()
        delta = values - mean
        mean += delta / t
        sum_squared_deviations += delta * (values - mean)
        std = np.sqrt(sum_squared_deviations / t)
        context[:10] = (values - mean) / np.maximum(std, 1.0)

        random_arm = rng.integers(n_arms)
        totals["Random"] += int(random_arm != y[index])
        regrets["Random"][t - 1] = totals["Random"]

        for name, bandit in bandits.items():
            arm, probability = bandit.select(context)
            reward = int(arm == y[index])
            bandit.update(context, arm, reward, probability)

            totals[name] += 1 - reward
            regrets[name][t - 1] = totals[name]

        if t % 1000 == 0 or t == len(order):
            print(f"Round {t}: " + ", ".join(
                f"{name} regret={total}" for name, total in totals.items()
            ))

    rounds = np.arange(1, len(order) + 1)
    plt.plot(rounds, regrets["Random"], ":", label="Random")
    plt.plot(rounds, regrets["Epsilon greedy"], "--", label="Epsilon greedy")
    plt.plot(rounds, regrets["Thompson sampling"], "-", label="Thompson sampling")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regret")
    plt.grid(axis="y", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cumulative_regret.png", dpi=150)
    plt.show()
    
