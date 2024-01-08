import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaseGradientBoosting:
    def __init__(
        self,
        loss,
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
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.base_estimator = self.init_base_estimator()
        self.base_estimator.fit(X, y)
        base_predictions = self.base_estimator.predict(X)

        self.estimators_ = []

        for _ in range(self.n_stages):
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
            self._update_terminal_regions(estimator, X, y, residuals_m)
            self.estimators_.append(estimator)

            base_predictions += self.learning_rate * estimator.predict(X)

        return self

    def predict(self, X):
        predictions = self.base_estimator.predict(X)

        for estimator in self.estimators_:
            predictions += self.learning_rate * estimator.predict(X)

        return predictions

    def _update_terminal_regions(self, estimator, X, y, residuals):
        terminal_regions = estimator.apply(X)

        for leaf in np.unique(terminal_regions):
            mask = terminal_regions == leaf
            self.loss.update_terminal_region(estimator.tree_, leaf, y[mask], residuals[mask])

        return estimator


class GradientBoostingClassifier(BaseGradientBoosting):
    def init_base_estimator(self):
        return DummyClassifier(strategy="prior")


class GradientBoostingRegressor(BaseGradientBoosting):
    def init_base_estimator(self):
        return DummyRegressor(strategy="mean")


class LossFunctionMSE:
    def __init__(self):
        self.name = "mse"

    def negative_gradient(self, y, y_pred):
        return y - y_pred

    def update_terminal_region(self, tree, leaf, y, residuals):
        pass
