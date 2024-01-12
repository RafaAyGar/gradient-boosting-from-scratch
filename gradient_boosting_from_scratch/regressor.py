import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from gradient_boosting_from_scratch.base import BaseGradientBoosting


class GradientBoostingRegressor(BaseGradientBoosting):
    def fit(self, X, y):
        self.base_estimator = DummyRegressor(strategy="mean")
        self.base_estimator.fit(X, y)
        base_predictions = self.base_estimator.predict(X)

        self.estimators_ = np.empty(self.n_stages, dtype=DecisionTreeRegressor)
        for m in range(self.n_stages):
            residuals_m = self.loss.negative_gradient(y, base_predictions)

            estimator = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state,
                ccp_alpha=self.ccp_alpha,
            )
            estimator.fit(X, residuals_m)

            terminal_regions = estimator.apply(X)
            self._update_terminal_regions(
                terminal_regions, estimator, y, residuals_m, base_predictions
            )

            self.estimators_[m] = estimator
            base_predictions += self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X):
        predictions = self.base_estimator.predict(X)

        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)

        return predictions

    def _update_terminal_regions(
        self, terminal_regions, estimator, y, residuals, base_predictions
    ):
        for leaf in np.unique(terminal_regions):
            mask = terminal_regions == leaf
            self.loss.update_terminal_region(
                estimator.tree_, leaf, y[mask], residuals[mask], base_predictions[mask]
            )

        return estimator
