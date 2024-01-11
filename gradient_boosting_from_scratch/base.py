import numpy as np


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
