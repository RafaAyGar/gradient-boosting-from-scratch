import numpy as np

from gradient_boosting_from_scratch._utils import logistic, softmax

# Regression Losses
# -----------------


class LossFunctionMSE:
    def __init__(self):
        self.name = "mse"

    def negative_gradient(self, y, y_pred):
        return y - y_pred

    def update_terminal_region(self, tree, leaf, y, residuals, base_predictions):
        pass


# Classification Losses
# -----------------


class LossFunctionBinomialDeviance:
    def __init__(self) -> None:
        self.name = "binomial_deviance"

    def negative_gradient(self, y, y_pred):
        p = logistic(y_pred)
        return y - p

    def update_terminal_region(self, tree, leaf, y, residuals, base_predictions):
        p = logistic(base_predictions)
        updated_lambda = np.sum(residuals) / np.sum((p * (1 - p)))
        tree.value[leaf, 0, 0] = updated_lambda


class LossFunctionMultinomialDeviance:
    def __init__(self) -> None:
        self.name = "multinomial_deviance"

    def negative_gradient(self, y_k, y_pred, k):
        probs = softmax(y_pred, axis=1)
        return y_k - probs[:, k]

    def update_terminal_region(self, tree, leaf, y, residuals, base_predictions, K):
        numerator = np.sum(residuals) * ((K - 1) / K)
        denominator = np.sum((y - residuals) * (1 - y + residuals))

        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator
