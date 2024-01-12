import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC_from_sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from gradient_boosting_from_scratch._losses import \
    LossFunctionMultinomialDeviance
from gradient_boosting_from_scratch.classifier_multiclass import \
    GradientBoostingMulticlassClassifier as GBC_from_scratch

N_SEEDS = 25

### Prepare the data
##
#
data = pd.read_csv(
    "gradient_boosting_from_scratch/testing/data_synthetic_classification.csv"
)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

skl_y_train_preds = []
skl_y_test_preds = []
scr_y_train_preds = []
scr_y_test_preds = []

seeds = np.random.randint(0, high=1000, size=N_SEEDS)
for seed in seeds:
    ### Build and fit from scratch gradient boosting classifier
    ##
    #
    gb_scratch = GBC_from_scratch(
        loss=LossFunctionMultinomialDeviance(),
        n_stages=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
    )
    gb_scratch.fit(X_train, y_train)

    ### Compare with sklearn's GradientBoostingRegressor
    ##
    #
    gb_sklearn = GBC_from_sklearn(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
    )
    gb_sklearn.fit(X_train, y_train)

    skl_y_train_preds.append(mean_absolute_error(y_train, gb_sklearn.predict(X_train)))
    skl_y_test_preds.append(mean_absolute_error(y_test, gb_sklearn.predict(X_test)))
    scr_y_train_preds.append(mean_absolute_error(y_train, gb_scratch.predict(X_train)))
    scr_y_test_preds.append(mean_absolute_error(y_test, gb_scratch.predict(X_test)))

print("")
print("Average sklearn train MAE:", np.mean(skl_y_train_preds))
print("Average scratch train MAE:", np.mean(scr_y_train_preds))
print("··················")
print("Average sklearn test MAE:", np.mean(skl_y_test_preds))
print("Average scratch test MAE:", np.mean(scr_y_test_preds))
print("")
