import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 150

# Generate synthetic feature variables
feature1 = np.random.rand(num_samples) * 10
feature2 = np.random.rand(num_samples) * 5
feature3 = np.random.rand(num_samples) * 8

# Generate synthetic target variable (response)
# You can customize the coefficients based on your regression equation
# In this example, the true coefficients are [2, 3, 1.5]
true_coefficients = [2, 3, 1.5]
noise = np.random.normal(0, 2, num_samples)
target = (
    true_coefficients[0] * feature1
    + true_coefficients[1] * feature2
    + true_coefficients[2] * feature3
    + noise
)

# Create a DataFrame
data = pd.DataFrame(
    {"Feature1": feature1, "Feature2": feature2, "Feature3": feature3, "Target": target}
)

# Display the first few rows of the synthetic dataset
print(data.head())

# Save the synthetic dataset to a CSV file
data.to_csv(
    "gradient_boosting_from_scratch/testing/testing_data_reg.csv",
    sep=" ",
    index=False,
    header=False,
)
