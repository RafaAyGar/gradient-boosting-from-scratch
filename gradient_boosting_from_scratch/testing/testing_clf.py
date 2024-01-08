import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier

from gradient_boosting_from_scratch.base import BaseGradientBoosting

data = pd.read_csv(
    "gradient_boosting_from_scratch/testing/testing_data_clf.csv", sep=" ", header=None
)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

gb = BaseGradientBoosting(
    base_estimator=DummyClassifier(),
    estimator=DecisionTreeClassifier(),
    n_stages=10,
    loss_function=None,
    learning_rate=0.1,
)

gb.fit(X, y)
