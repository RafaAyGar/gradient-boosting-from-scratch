import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from gradient_boosting_from_scratch.base import GradientBoostingRegressor as GBRegressor
from gradient_boosting_from_scratch.base import LossFunctionMSE

data = pd.read_csv(
    "gradient_boosting_from_scratch/testing/data_synthetic_regression.csv"
)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

gb = GBRegressor(
    loss=LossFunctionMSE(),
    random_state=22,
    max_depth=3,
)
gb.fit(X, y)

predictions = gb.predict(X)
print("From scratch Gradient Boosting:", mean_absolute_error(y, predictions))


### Compare with sklearn's GradientBoostingRegressor
##
#

from sklearn.ensemble import GradientBoostingRegressor

gb_sklearn = GradientBoostingRegressor(random_state=22, max_depth=3)
gb_sklearn.fit(X, y)

predictions_sklearn = gb_sklearn.predict(X)
print("From sklearn Gradient Boosting:", mean_absolute_error(y, predictions_sklearn))
