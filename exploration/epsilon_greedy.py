import numpy as np


def epsilon_greedy_probabilities(predictions, epsilon):
    """Return the probability of choosing each arm."""
    n_arms = len(predictions)
    probabilities = np.full(n_arms, epsilon / n_arms)
    probabilities[np.argmax(predictions)] += 1.0 - epsilon
    return probabilities