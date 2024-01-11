import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from gradient_boosting_from_scratch._utils import logistic


class BaseGradientBoosting:
    def __init__(
        self,
        loss,
        task,
        learning_rate=0.1,
        n_stages=100,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        max_features=None,
        ccp_alpha=0.0,
        max_leaf_nodes=None,
        random_state=None,
    ):
        self.loss = loss
        self.task = task
        self.learning_rate = learning_rate
        self.n_stages = n_stages
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, X, y):
        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y_one_hot = self._one_hot_encode(y)

        self.base_estimator = self.init_base_estimator()
        self.base_estimator.fit(X, y)
        base_predictions = self._base_predict(X)

        self.estimators_ = []

        for _ in range(self.n_stages):
            residuals_m = self.loss.negative_gradient(y_one_hot, base_predictions)

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
            self._update_terminal_regions(
                estimator, X, y_one_hot, residuals_m, base_predictions
            )
            self.estimators_.append(estimator)

            base_predictions += self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X):
        predictions = self._base_predict(X)

        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)

        if self.task == "classification":
            predictions = self.predict_proba(X)
            if self.n_classes_ == 2:
                predictions = np.round(predictions)
            else:
                predictions = np.argmax(predictions, axis=1)

        return predictions

    def _base_predict(self, X):
        if self.task == "classification":
            base_predictions = self.base_estimator.predict_proba(X)
            if self.n_classes_ == 2:
                base_predictions = base_predictions[:, 1]
            else:
                base_predictions = np.array(base_predictions)
            base_predictions = np.log(base_predictions / (1 - base_predictions))
        else:
            base_predictions = self.base_estimator.predict(X)

        return base_predictions

    def _update_terminal_regions(self, estimator, X, y, residuals, base_predictions):
        terminal_regions = estimator.apply(X)

        for leaf in np.unique(terminal_regions):
            mask = terminal_regions == leaf
            self.loss.update_terminal_region(
                estimator.tree_, leaf, y[mask], residuals[mask], base_predictions[mask]
            )

        return estimator

    def _one_hot_encode(self, y):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        one_hot_encoded = np.zeros((n_samples, n_classes))

        for i, label in enumerate(y):
            one_hot_encoded[i, label] = 1

        return one_hot_encoded


class GradientBoostingClassifier(BaseGradientBoosting):
    def init_base_estimator(self):
        return DummyClassifier(strategy="prior")

    def predict_proba(self, X):
        predictions = self._base_predict(X)

        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)

        predictions = logistic(predictions)
        return predictions


class GradientBoostingRegressor(BaseGradientBoosting):
    def init_base_estimator(self):
        return DummyRegressor(strategy="mean")
