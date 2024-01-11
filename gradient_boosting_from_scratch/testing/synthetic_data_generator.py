import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Global variables
N_SAMPLES = 500
N_FEATURES = 4
DATASET_TYPE = "regression"  # can be 'classification' or 'regression'
NUM_CLASSES = 5  # Only used for classification. For binary, set it to 2


def create_synthetic_data(dataset_type, num_classes=2, n_samples=150, n_features=4):
    """
    Generate synthetic data based on the specified dataset type.

    Args:
    - dataset_type (str): Type of dataset to generate ('classification' or 'regression').
    - num_classes (int): Number of classes for classification. Default is 2 for binary classification.

    Returns:
    - X (np.ndarray): Features of the synthetic dataset.
    - y (np.ndarray): Labels or target values for the synthetic dataset.
    """
    if dataset_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features - 2,
            n_classes=num_classes,
            n_clusters_per_class=1,
            random_state=22,
        )
    elif dataset_type == "regression":
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, random_state=22
        )
    else:
        raise ValueError("dataset_type must be either 'classification' or 'regression'")

    return X, y


# Generate the data
X, y = create_synthetic_data(DATASET_TYPE, NUM_CLASSES, N_SAMPLES, N_FEATURES)

# Display the shape of the datasets
print("Features shape:", X.shape)
print("Labels shape:", y.shape)

dataset = pd.DataFrame(X)
dataset["target"] = y

# Save the dataset
dataset.to_csv(f"synthetic_data_{DATASET_TYPE}.csv", index=False)
