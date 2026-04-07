"""Sparse logistic and FDR regression examples using the iQ-ML SDK.

This example creates a synthetic binary classification dataset where only 2
out of 15 features carry the true signal, then compares sparse logistic
regression and FDR-based feature selection.
"""

import numpy as np
import iq.api.iqrestapi
import iq.ml.logistic_regression
import iq.ml.fdr_regression

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(1)
n_samples = 400
n_features = 15
k = 2
true_features = [3, 9]

# Build the feature matrix
X = rng.standard_normal((n_samples, n_features))

# True decision boundary: sign of a linear combination of true features
signal = X[:, true_features[0]] - X[:, true_features[1]]
y = np.sign(signal + 0.3 * rng.standard_normal(n_samples))
y[y == 0] = 1  # resolve ties

print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"True informative features: {true_features}")
print(f"Class balance: {(y == 1).sum()} positive, {(y == -1).sum()} negative")
print()

# --- Sparse logistic regression ---
selected_lr, weights_lr = iq.ml.logistic_regression.solve_logreg_pa(
    X, y,
    k=k,
    lambda_l2=0.01,
    options={"copies": 100},
    random_number_generator_seed=42,
    description="Sparse logreg example",
)
print("Sparse Logistic Regression:")
print(f"  Selected features: {list(selected_lr)}")
print(f"  Model weights (intercept + k features): {np.round(weights_lr, 4)}")
print()

# --- Sparse FDR regression ---
selected_fdr, weights_fdr = iq.ml.fdr_regression.solve_fdr_pa(
    X, y,
    k=k,
    lambda_l2=0.01,
    options={"copies": 100},
    random_number_generator_seed=42,
    description="Sparse FDR example",
)
print("Sparse FDR Regression:")
print(f"  Selected features: {list(selected_fdr)}")
print(f"  Model weights (intercept + k features): {np.round(weights_fdr, 4)}")
print()
print(f"True features were: {true_features}")
