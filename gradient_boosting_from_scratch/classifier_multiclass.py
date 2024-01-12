import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from gradient_boosting_from_scratch._utils import softmax
from gradient_boosting_from_scratch.base import BaseGradientBoosting


class GradientBoostingMulticlassClassifier(BaseGradientBoosting):
    def fit(self, X, y):
        self._rng = check_random_state(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_one_hot = self._one_hot_encode(y)

        self.base_estimator = DummyClassifier(strategy="prior")
        self.base_estimator.fit(X, y)
        base_predictions = self.base_estimator.predict_proba(X)
        # See why we use log-probabilities instead of log-odds in [...]
        base_predictions = np.log(base_predictions)

        self.estimators_ = np.empty(
            (self.n_stages, self.n_classes_), dtype=DecisionTreeRegressor
        )
        for m in range(self.n_stages):
            for k in range(self.n_classes_):
                residuals_m_k = self.loss.negative_gradient(
                    y_one_hot[:, k], base_predictions, k
                )

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
                    random_state=self._rng,
                    ccp_alpha=self.ccp_alpha,
                )
                estimator.fit(X, residuals_m_k)

                terminal_regions = estimator.apply(X)
                self._update_terminal_regions(
                    terminal_regions,
                    estimator,
                    y_one_hot[:, k],
                    residuals_m_k,
                    base_predictions[:, k],
                )

                self.estimators_[m, k] = estimator
                base_predictions[:, k] += self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X):
        predictions = self.predict_proba(X)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def predict_proba(self, X):
        predictions = self.base_estimator.predict_proba(X)
        predictions = np.log(predictions)

        for m in range(self.n_stages):
            for k in range(self.n_classes_):
                predictions[:, k] += self.learning_rate * self.estimators_[
                    m, k
                ].predict(X)

        predictions = softmax(predictions, axis=1)
        return predictions

    def _update_terminal_regions(
        self, terminal_regions, estimator, y, residuals, base_predictions
    ):
        for leaf in np.unique(terminal_regions):
            mask = terminal_regions == leaf
            self.loss.update_terminal_region(
                estimator.tree_,
                leaf,
                y[mask],
                residuals[mask],
                base_predictions[mask],
                self.n_classes_,
            )

        return estimator

    def _one_hot_encode(self, y):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        one_hot_encoded = np.zeros((n_samples, n_classes))

        for i, label in enumerate(y):
            one_hot_encoded[i, label] = 1

        return one_hot_encoded
