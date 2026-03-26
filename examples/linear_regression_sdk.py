"""Sparse linear regression example using the iQ-ML SDK.

This example creates a synthetic regression dataset where only 4 out of 20
features are truly informative, then uses the solver to recover those features.
"""

import numpy as np
import iq.api.iqrestapi
import iq.ml.linear_regression

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(0)
n_samples = 300
n_features = 20
k = 4
true_features = [1, 7, 12, 17]

# Build the feature matrix
X = rng.standard_normal((n_samples, n_features))

# True coefficients: nonzero only on true_features
true_w = np.zeros(n_features)
true_w[true_features] = [2.0, -1.5, 0.8, -1.2]

# Observations with small noise
y = X @ true_w + 0.1 * rng.standard_normal(n_samples)

print(f"Dataset: {n_samples} samples, {n_features} features, {k} true nonzero features")
print(f"True nonzero features: {true_features}")
print()

w, mse = iq.ml.linear_regression.solve_sparse_linreg(
    X, y,
    k=k,
    lambda_l2=0.0,
    options={"copies": 100},
    random_number_generator_seed=42,
    description="Sparse linreg example",
)

w = np.array(w)
selected = np.where(np.abs(w) > 1e-10)[0].tolist()
print(f"Selected features: {selected}")
print(f"True features:     {true_features}")
print(f"MSE: {mse:.6f}")
print()
print("Estimated coefficients on selected features:")
for i in selected:
    print(f"  Feature {i:2d}: estimated={w[i]:.4f}  true={true_w[i]:.4f}")
