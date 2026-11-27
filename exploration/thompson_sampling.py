import numpy as np


class ThompsonSampling:
    def __init__(self, n_features, n_arms, gamma=0.05, n_samples=128, prior_variance=0.1, observation_variance=1.0,
    ):
        self.n_features = n_features
        self.n_arms = n_arms
        self.gamma = gamma
        self.n_samples = n_samples
        self.observation_variance = observation_variance
        self.mean = np.zeros((n_arms, n_features + 1))
        self.variance = np.full(
            (n_arms, n_features + 1),
            prior_variance,
        )

    def _features(self, context):
        context = np.asarray(context, dtype=np.float64)
        if context.shape != (self.n_features,):
            raise ValueError(f"Expected context shape ({self.n_features},)")
        return np.append(context, 1.0)

    def probabilities(self, predictions, context, rng):
        features = self._features(context)
        residual_mean = self.mean @ features
        residual_variance = self.variance @ (features ** 2)

        sampled_scores = rng.normal(
            loc=predictions + residual_mean,
            scale=np.sqrt(residual_variance),
            size=(self.n_samples, self.n_arms),
        )

        winners = np.argmax(sampled_scores, axis=1)
        votes = np.bincount(winners, minlength=self.n_arms)

        probabilities = (
            (1 - self.gamma) * votes / self.n_samples
            + self.gamma / self.n_arms
        )
        return probabilities

    def update(self, context, arm, reward, prediction):
        features = self._features(context)
        residual = float(reward) - float(prediction)
        variance = self.variance[arm]
        predicted_residual = self.mean[arm] @ features
        denominator = self.observation_variance + np.dot(
            variance, features ** 2
        )

        gain = variance * features / denominator
        self.mean[arm] += gain * (residual - predicted_residual)
        self.variance[arm] = variance - gain * features * variance