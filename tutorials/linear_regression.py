# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: Sparse Linear Regression for House Price Prediction
#
# This tutorial demonstrates how to use the **iQ sparse linear regression
# API** to perform feature selection and price prediction on the
# [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.
#
# ## What you will learn
#
# 1. How to load and explore the training and test datasets.
# 2. How to clean and scale the data.
# 3. How to call `iq.ml.linear_regression.solve_sparse_linreg` to find
#    the best sparse coefficient vector with exactly `k` nonzero entries.
# 4. How to evaluate the sparse model using standard regression metrics.
# 5. How to sweep over different values of `k` to find the best sparsity
#    level.
# 6. How to generate predictions on the held-out test set.
#
# ## Prerequisites
#
# * Python >= 3.11
# * The `iq` package installed and configured with valid API credentials.
# * The House Prices training and test datasets stored as CSV files.

# %% [markdown]
# ## Step 1 - Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import iq.api.iqrestapi
import iq.ml.linear_regression

# %% [markdown]
# ## Step 1.2 - Initialize the API credentials

# %%
iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# %% [markdown]
# ## Step 2 - Load and Explore the Datasets
#
# The House Prices dataset from Kaggle contains 79 explanatory variables
# describing aspects of residential homes sold in Ames, Iowa. The data is
# split into two files:
#
# - **home_prices_train.csv**: contains both the features and the target
#   variable (`SalePrice`).
# - **home_prices_test.csv**: contains only the features. The target is
#   withheld for competition scoring.
#
# Since the test set has no target, we will split the training set into a
# train and test partition to evaluate the model. The Kaggle test set
# will be used at the end to generate final predictions.

# %%
# -- Configuration -------------------------------------------------------------
TRAIN_DATASET_PATH = Path("./data/home_prices_train.csv")
TEST_DATASET_PATH = Path("./data/home_prices_test.csv")

# -- Load data -----------------------------------------------------------------
df_train_full = pd.read_csv(TRAIN_DATASET_PATH, index_col=0)
df_test = pd.read_csv(TEST_DATASET_PATH, index_col=0)

print(f"Training set shape : {df_train_full.shape}")
print(f"Test set shape     : {df_test.shape}")
print()
print(df_train_full.head())

# %% [markdown]
# ## Step 3 - Keep Only Numerical Features
#
# For simplicity, we keep only the numerical columns. Rows with any missing
# values are dropped from the training set. For the test set, missing values
# are filled with the column median computed on the training set to avoid
# losing samples.

# %%
TARGET_COLUMN = "SalePrice"

# Separate target before filtering columns.
y_full = df_train_full[TARGET_COLUMN].copy()

# Keep only numerical feature columns (exclude the target).
numerical_columns = (
    df_train_full
    .drop(columns=[TARGET_COLUMN])
    .select_dtypes(include=[np.number])
    .columns
)

X_train_full_df = df_train_full[numerical_columns].copy()
X_test_df = df_test[numerical_columns].copy()

# Drop training rows with missing values.
b_row_has_no_nans = X_train_full_df.notna().all(axis=1)
X_train_full_df = X_train_full_df.loc[b_row_has_no_nans]
y_full = y_full.loc[b_row_has_no_nans]

# Fill test set missing values with the training median.
train_medians = X_train_full_df.median()
X_test_df = X_test_df.fillna(train_medians)

feature_names = np.array(numerical_columns)
num_features = len(feature_names)

print(f"Number of numerical features : {num_features}")
print(f"Training samples (clean)     : {X_train_full_df.shape[0]}")
print(f"Test samples                 : {X_test_df.shape[0]}")

# %% [markdown]
# ## Step 4 - Prepare Arrays and Split into Train / Test
#
# We hold out 20 % of the training data as a test set so we can evaluate
# the model. Setting a fixed `random_state` guarantees reproducibility.

# %%
DATASET_DTYPE = np.float64
TEST_SET_RATIO = 0.2
SPLIT_RANDOM_STATE = 114168850

X_full = X_train_full_df.to_numpy().astype(DATASET_DTYPE)
y_full_array = y_full.to_numpy().astype(DATASET_DTYPE)
X_kaggle_test = X_test_df.to_numpy().astype(DATASET_DTYPE)

X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full_array,
    test_size=TEST_SET_RATIO,
    random_state=SPLIT_RANDOM_STATE,
)

print(f"Training samples   : {X_train.shape[0]}")
print(f"Test samples       : {X_test.shape[0]}")
print(f"Kaggle test samples: {X_kaggle_test.shape[0]}")

# %% [markdown]
# ## Step 5 - Feature Scaling
#
# The sparse linear regression solver benefits from features being on a
# comparable scale. We fit a `StandardScaler` on the **training set only**
# and then apply the same transformation to the test set and the Kaggle
# test set to avoid data leakage.

# %%
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_kaggle_test = scaler.transform(X_kaggle_test)

# %% [markdown]
# ## Step 6 - Sparse Linear Regression with `solve_sparse_linreg`
#
# Here we call the iQ API to find a coefficient vector with exactly `k`
# nonzero entries that minimizes the mean squared error. The solver uses a
# quantum-inspired algorithm that explores feature subsets more efficiently
# than exhaustive search.
#
# ### Key parameters
#
# | Parameter | Description |
# |-----------|-------------|
# | `X` | Scaled feature matrix (training set). |
# | `y` | Observation vector (sale prices). |
# | `k` | Number of nonzero coefficients. |
# | `lambda_l2` | L2 regularization strength. |
# | `options` | Dict with `copies` (trajectories) and `tol` (convergence). |
# | `random_number_generator_seed` | Seed for reproducibility. |

# %%
# -- Hyper-parameters ----------------------------------------------------------
NUM_NONZERO_COEFFICIENTS = 10
LAMBDA_L2 = 1e-5 * X_train.shape[0]
RNG_SEED = 123321

options = {
    "copies": 60,
    "tol": 1e-3,
}

# -- Call the API --------------------------------------------------------------
sparse_coefficients, training_mse = iq.ml.linear_regression.solve_sparse_linreg(
    X=X_train,
    y=y_train,
    k=NUM_NONZERO_COEFFICIENTS,
    lambda_l2=LAMBDA_L2,
    options=options,
    random_number_generator_seed=RNG_SEED,
    description=f"iQ-ML Tutorial: House Prices sparse regression, k={NUM_NONZERO_COEFFICIENTS}",
)

selected_feature_indices = np.sort(np.asarray(sparse_coefficients, dtype=int))

# Fit an OLS model on the selected features to obtain the regression weights.
ols_model = LinearRegression()
ols_model.fit(X_train[:, selected_feature_indices], y_train)

print(f"Training MSE (API)       : {training_mse:,.2f}")
print(f"Selected feature indices : {selected_feature_indices.tolist()}")
print(f"Selected feature names   : {feature_names[selected_feature_indices].tolist()}")
print()
print("Regression weights:")
for ix, coef in zip(selected_feature_indices, ols_model.coef_):
    print(f"  {feature_names[ix]:30s} : {coef:+.6f}")

# %% [markdown]
# ## Step 7 - Evaluate the Model
#
# We compute standard regression metrics on both the training and test
# sets: mean squared error (MSE), root mean squared error (RMSE), mean
# absolute error (MAE), and the coefficient of determination (R-squared).

# %%
def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True observation values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    metrics : dict
        Dictionary with MSE, RMSE, MAE and R-squared.

    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mse": round(mse, 2),
        "rmse": round(np.sqrt(mse), 2),
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "r_squared": round(r2_score(y_true, y_pred), 4),
    }
    return metrics


# -- Training set --------------------------------------------------------------
y_pred_train = ols_model.predict(X_train[:, selected_feature_indices])
metrics_train = compute_regression_metrics(y_train, y_pred_train)

# -- Test set ------------------------------------------------------------------
y_pred_test = ols_model.predict(X_test[:, selected_feature_indices])
metrics_test = compute_regression_metrics(y_test, y_pred_test)

# -- Display results -----------------------------------------------------------
df_metrics = pd.DataFrame({"train": metrics_train, "test": metrics_test})
print(df_metrics.to_string())

# %% [markdown]
# ## Step 8 - Sweep Over Different Values of `k`
#
# In practice, you don't know the ideal number of features in advance. To
# select the best `k` without biasing our test-set estimate, we split the
# training data into a smaller training subset and a validation set. The loop
# below runs the API for several values of `k`, fits on the training subset,
# and measures R-squared on the validation set. We then pick the `k` with the
# highest validation R-squared and report generalisation performance on the
# held-out test set.

# %%
VALIDATION_RATIO = 0.2
VALIDATION_RANDOM_STATE = 738291054

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_RATIO,
    random_state=VALIDATION_RANDOM_STATE,
)

print(f"Training subset samples : {X_tr.shape[0]}")
print(f"Validation samples      : {X_val.shape[0]}")
print(f"Test samples            : {X_test.shape[0]}")

# %%
MAX_K = min(30, num_features - 1)
k_values = np.arange(1, MAX_K + 1)
sweep_results = []

for k in k_values:
    # Sparse regression via iQ API (fitted on training subset only).
    coefficients_k, mse_k = iq.ml.linear_regression.solve_sparse_linreg(
        X=X_tr,
        y=y_tr,
        k=int(k),
        lambda_l2=LAMBDA_L2,
        options=options,
        random_number_generator_seed=RNG_SEED,
        description=f"iQ-ML Tutorial: House Prices sparse regression, k={k}",
    )
    indices_k = np.sort(np.asarray(coefficients_k, dtype=int))

    # Fit OLS on the selected features and evaluate on the validation set.
    ols_k = LinearRegression()
    ols_k.fit(X_tr[:, indices_k], y_tr)
    y_pred_k = ols_k.predict(X_val[:, indices_k])
    r_squared_k = r2_score(y_val, y_pred_k)
    rmse_k = np.sqrt(mean_squared_error(y_val, y_pred_k))

    sweep_results.append({
        "k": int(k),
        "rmse_val": round(rmse_k, 2),
        "r_squared_val": round(r_squared_k, 4),
    })
    print(f"k = {k:3d}  |  RMSE (val) = {rmse_k:12,.2f}  |  R2 (val) = {r_squared_k:.4f}")

df_sweep = pd.DataFrame(sweep_results)
print()
print(df_sweep.to_string(index=False))

# %% [markdown]
# ## Step 9 - Evaluate the Best `k` on the Test Set
#
# We select the `k` that achieved the highest validation R-squared, retrain
# on the full training set, and measure generalisation on the untouched test
# set.

# %%
best_row = df_sweep.loc[df_sweep["r_squared_val"].idxmax()]
best_k = int(best_row["k"])
print(f"Best k = {best_k}  (validation R2 = {best_row['r_squared_val']:.4f})")

# Sparse regression on the full training set with the chosen k.
best_coefficients, best_mse = iq.ml.linear_regression.solve_sparse_linreg(
    X=X_train,
    y=y_train,
    k=best_k,
    lambda_l2=LAMBDA_L2,
    options=options,
    random_number_generator_seed=RNG_SEED,
    description=f"iQ-ML Tutorial: House Prices final model, k={best_k}",
)
best_feature_indices = np.sort(np.asarray(best_coefficients, dtype=int))
print(f"Selected features: {feature_names[best_feature_indices].tolist()}")

# Fit OLS on the selected features and evaluate on the test set.
best_ols = LinearRegression()
best_ols.fit(X_train[:, best_feature_indices], y_train)
y_pred_test_best = best_ols.predict(X_test[:, best_feature_indices])
r_squared_test_best = r2_score(y_test, y_pred_test_best)
rmse_test_best = np.sqrt(mean_squared_error(y_test, y_pred_test_best))
print(f"Test RMSE with k={best_k}: {rmse_test_best:,.2f}")
print(f"Test R2 with k={best_k}  : {r_squared_test_best:.4f}")

weights = dict(
    sorted(
        zip(
            feature_names[best_feature_indices].tolist(),
            best_ols.coef_.tolist(),
        ),
        key=lambda x: x[1],
    )
)
print("\nModel weights (sorted by value):")
for feat, w in weights.items():
    print(f"  {feat:>40s}: {w:+.4f}")

# %% [markdown]
# ## Step 10 - Generate Predictions on the Kaggle Test Set
#
# Finally, we use the best sparse coefficients obtained in Step 9 to
# generate predictions on the Kaggle test set. These predictions can be
# exported to a CSV file and submitted to the competition.

# %%
y_pred_kaggle_test = best_ols.predict(X_kaggle_test[:, best_feature_indices])

# Build a submission dataframe with the original test set index.
df_submission = pd.DataFrame({
    "Id": df_test.index,
    "SalePrice": y_pred_kaggle_test,
})

SUBMISSION_PATH = Path("./data/submission.csv")
df_submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Predictions generated for {len(y_pred_kaggle_test)} test samples.")
print(f"Submission saved to: {SUBMISSION_PATH}")
print()
print(df_submission.head(10))

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Loaded the House Prices training and test datasets from Kaggle.
# 2. Kept only numerical features, handled missing values, and standardized
#    the features.
# 3. Split the data into training and test sets.
# 4. Called `iq.ml.linear_regression.solve_sparse_linreg` to find the best
#    sparse coefficient vector using a quantum-inspired algorithm.
# 5. Evaluated the sparse model using MSE, RMSE, MAE and R-squared.
# 6. Used a train/validation split to sweep over values of `k` and selected
#    the best sparsity level without leaking test-set information.
# 7. Retrained on the full training set with the optimal `k` and measured
#    generalisation on the held-out test set.
# 8. Generated predictions on the Kaggle test set for competition submission.
#
# You can extend this workflow by adding cross-validation folds (see
# `KFold` in scikit-learn) or by trying different values of `lambda_l2`.