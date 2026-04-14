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
# # Tutorial: Sparse Logistic Regression for Credit Risk Classification
#
# This tutorial demonstrates how to use the **iQ sparse logistic regression
# API** to perform feature selection and binary classification on the
# [Give Me Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.
#
# ## What you will learn
#
# 1. How to load and explore the dataset.
# 2. How to split and scale the data.
# 3. How to call `iq.ml.logistic_regression.solve_sparse_logreg` to select
#    the best `k` features.
# 4. How to fit a logistic regression model on the selected features and
#    evaluate its performance.
#
# ## Prerequisites
#
# * Python >= 3.11
# * The `iq` package installed and configured with valid API credentials.
# * The Give Me Credit dataset stored as a CSV file.

# %% [markdown]
# ## Step 1 - Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import iq.api.iqrestapi
import iq.ml.logistic_regression

# %% [markdown]
# ## Step 1.2 - Initialize the API credentials

# %%
iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

# %% [markdown]
# ## Step 2 - Load and Explore the Dataset
#
# The Give Me Credit dataset contains financial attributes of borrowers and a
# binary target variable (`SeriousDlqin2yrs`) that indicates whether a borrower
# experienced 90-day-or-more delinquency.
#
# We assume the data has already been cleaned and saved as a single CSV where
# the **first column** (after the index) is the target and the remaining
# columns are features.

# %%
# -- Configuration -------------------------------------------------------------
DATASET_PATH = Path("./data/give_me_credit.csv")

# -- Load data -----------------------------------------------------------------
df = pd.read_csv(DATASET_PATH, index_col=0)

num_samples = df.shape[0]
num_features = df.shape[1] - 1  # Subtract the target column.

print(f"Dataset shape      : {df.shape}")
print(f"Number of samples  : {num_samples}")
print(f"Number of features : {num_features}")
print()
print(df.head())

# %% [markdown]
# ## Step 3 - Prepare the Feature Matrix and Label Vector
#
# We separate the target vector `y` from the feature matrix `X` and cast them
# to `float32` to keep memory usage low.

# %%
DATASET_DTYPE = np.float32

y = df.iloc[:, 0].to_numpy().astype(DATASET_DTYPE)
X = df.iloc[:, 1:].to_numpy().astype(DATASET_DTYPE)

feature_names = np.array(df.columns[1:])

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Class balance - positive: {y.mean():.2%}, negative: {1 - y.mean():.2%}")

# %% [markdown]
# ## Step 4 - Train / Test Split
#
# We hold out 20 % of the data as a test set. Setting a fixed
# `random_state` guarantees reproducibility.

# %%
TEST_SET_RATIO = 0.2
SPLIT_RANDOM_STATE = 114168850

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SET_RATIO,
    random_state=SPLIT_RANDOM_STATE,
    stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# %% [markdown]
# ## Step 5 - Feature Scaling
#
# Logistic regression is sensitive to the scale of the features. We fit a
# `StandardScaler` on the **training set only** and then apply the same
# transformation to the test set to avoid data leakage.

# %%
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Step 6 - Feature Selection with `solve_sparse_logreg`
#
# Here we call the iQ API to jointly select the best `k`
# features and fit a sparse logistic regression model. The solver uses a
# quantum-inspired algorithm that explores feature subsets more efficiently
# than exhaustive search.
#
# ### Key parameters
#
# | Parameter | Description |
# |-----------|-------------|
# | `X` | Scaled feature matrix (training set). |
# | `y` | Binary label vector. |
# | `k` | Number of features to select. |
# | `lambda_l2` | L2 regularisation strength. |
# | `accelerate` | Use Fisher Discriminant Ratio for faster exploration. |
# | `options` | Dict with `copies` (trajectories) and `tol` (convergence). |
# | `random_number_generator_seed` | Seed for reproducibility. |

# %%
# -- Hyper-parameters ----------------------------------------------------------
NUM_FEATURES_TO_SELECT = 5
LAMBDA_L2 = 1e-5 * X_train.shape[0]
RNG_SEED = 123321

options = {
    "copies": 60,
    "tol": 1e-3,
}

# -- Call the API --------------------------------------------------------------
selected_feature_indices, api_weights = iq.ml.logistic_regression.solve_sparse_logreg(
    X=X_train,
    y=y_train,
    k=NUM_FEATURES_TO_SELECT,
    lambda_l2=LAMBDA_L2,
    accelerate=True,
    options=options,
    random_number_generator_seed=RNG_SEED,
    description=f"iQ-ML Tutorial: Give Me Credit feature selection, k={NUM_FEATURES_TO_SELECT}",
)

selected_feature_indices = np.asarray(selected_feature_indices)
api_weights = np.asarray(api_weights)

print(f"Selected feature indices : {selected_feature_indices}")
print(f"Selected feature names   : {feature_names[selected_feature_indices].tolist()}")
print()
print("API weights (intercept + coefficients):")
print(f"  Intercept : {api_weights[0]:.6f}")
for i, ix in enumerate(selected_feature_indices):
    print(f"  {feature_names[ix]:30s} : {api_weights[i + 1]:.6f}")

# %% [markdown]
# ## Step 7 - Fit a Logistic Regression on the Selected Features
#
# The API returns the optimal feature subset. We now fit a standard
# `sklearn.linear_model.LogisticRegression` on only the selected columns
# so that we can easily obtain predictions and probability estimates.

# %%
# Restrict the datasets to the selected features.
X_train_selected = np.ascontiguousarray(X_train[:, selected_feature_indices])
X_test_selected = np.ascontiguousarray(X_test[:, selected_feature_indices])

# The regularisation strength in sklearn is C = 1 / lambda.
MIN_LAMBDA = 1e-10
C_PARAM = 1.0 / max(LAMBDA_L2, MIN_LAMBDA)

model = LogisticRegression(
    C=C_PARAM,
    max_iter=1000,
    solver="lbfgs",
    tol=1e-5,
    l1_ratio=0  # Set only L2 penalty.
)
model.fit(X_train_selected, y_train)

# %% [markdown]
# ## Step 8 - Evaluate the Model
#
# We compute standard classification metrics on both the training and test
# sets: accuracy, AUC-ROC, precision, and recall.

# %%
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_pred: np.ndarray,
) -> dict:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    p_pred : np.ndarray
        Predicted probability matrix of shape (n_samples, 2). The second
        column is the probability of the positive class.

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, AUC-ROC, precision and recall.

    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, p_pred[:, 1]), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    return metrics


# -- Training set --------------------------------------------------------------
y_pred_train = model.predict(X_train_selected)
p_pred_train = model.predict_proba(X_train_selected)
metrics_train = compute_classification_metrics(y_train, y_pred_train, p_pred_train)

# -- Test set ------------------------------------------------------------------
y_pred_test = model.predict(X_test_selected)
p_pred_test = model.predict_proba(X_test_selected)
metrics_test = compute_classification_metrics(y_test, y_pred_test, p_pred_test)

# -- Display results -----------------------------------------------------------
df_metrics = pd.DataFrame({"train": metrics_train, "test": metrics_test})
print(df_metrics.to_string())

# %% [markdown]
# ## Step 9 - Sweep Over Different Values of `k`
#
# In practice, you don't know the ideal number of features in advance. To
# select the best `k` without biasing our test-set estimate, we split the
# training data into a smaller training subset and a validation set. The loop
# below runs the API for several values of `k`, trains on the training subset,
# and measures ROC-AUC on the validation set. We then pick the `k` with the
# highest validation AUC and report generalisation performance on the held-out
# test set.

# %%
VALIDATION_RATIO = 0.2
VALIDATION_RANDOM_STATE = 738291054

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_RATIO,
    random_state=VALIDATION_RANDOM_STATE,
    stratify=y_train
)

print(f"Training subset samples : {X_tr.shape[0]}")
print(f"Validation samples      : {X_val.shape[0]}")
print(f"Test samples            : {X_test.shape[0]}")

# %%
MAX_K = min(20, num_features - 1)
k_values = np.arange(1, MAX_K + 1)
auc_results = []

for k in k_values:
    # Feature selection via iQ API (fitted on training subset only).
    selected_indices, _ = iq.ml.logistic_regression.solve_sparse_logreg(
        X=X_tr,
        y=y_tr,
        k=int(k),
        lambda_l2=LAMBDA_L2,
        accelerate=True,
        options=options,
        random_number_generator_seed=RNG_SEED,
        description=f"iQ-ML Tutorial: Give Me Credit feature selection, k={k}",
    )
    selected_indices = np.asarray(selected_indices)

    # Fit sklearn model on the training subset with selected features.
    X_tr_k = np.ascontiguousarray(X_tr[:, selected_indices])
    X_val_k = np.ascontiguousarray(X_val[:, selected_indices])

    model_k = LogisticRegression(
        C=C_PARAM, max_iter=1000, solver="lbfgs", tol=1e-5, l1_ratio=0
    )
    model_k.fit(X_tr_k, y_tr)

    p_pred_k = model_k.predict_proba(X_val_k)
    auc_k = roc_auc_score(y_val, p_pred_k[:, 1])
    auc_results.append({"k": int(k), "auc_roc_val": round(auc_k, 4)})
    print(f"k = {k:3d}  |  AUC-ROC (val) = {auc_k:.4f}")

df_auc = pd.DataFrame(auc_results)
print()
print(df_auc.to_string(index=False))

# %% [markdown]
# ## Step 10 - Evaluate the Best `k` on the Test Set
#
# We select the `k` that achieved the highest validation AUC, retrain on the
# full training set, and measure generalisation on the untouched test set.

# %%
best_row = df_auc.loc[df_auc["auc_roc_val"].idxmax()]
best_k = int(best_row["k"])
print(f"Best k = {best_k}  (validation AUC-ROC = {best_row['auc_roc_val']:.4f})")

# Feature selection on the full training set with the chosen k.
best_indices, _ = iq.ml.logistic_regression.solve_sparse_logreg(
    X=X_train,
    y=y_train,
    k=best_k,
    lambda_l2=LAMBDA_L2,
    accelerate=True,
    options=options,
    random_number_generator_seed=RNG_SEED,
    description=f"iQ-ML Tutorial: Give Me Credit final model, k={best_k}",
)
best_indices = np.asarray(best_indices)

print(f"Selected features: {feature_names[best_indices].tolist()}")

# Retrain on the full training set with the selected features.
X_train_best = np.ascontiguousarray(X_train[:, best_indices])
X_test_best = np.ascontiguousarray(X_test[:, best_indices])

model_best = LogisticRegression(
    C=C_PARAM, max_iter=1000, solver="lbfgs", tol=1e-5, l1_ratio=0
)
model_best.fit(X_train_best, y_train)

p_pred_test_best = model_best.predict_proba(X_test_best)
auc_test_best = roc_auc_score(y_test, p_pred_test_best[:, 1])
print(f"Test AUC-ROC with k={best_k}: {auc_test_best:.4f}")

weights = dict(
    sorted(
        zip(
            feature_names[best_indices].tolist(),
            model_best.coef_[0].tolist(),
        ),
        key=lambda x: x[1],
    )
)
print("\nModel weights (sorted by value):")
for feat, w in weights.items():
    print(f"  {feat:>40s}: {w:+.4f}")

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Loaded the Give Me Credit dataset.
# 2. Split the data into training and test sets and standardised the features.
# 3. Called `iq.ml.logistic_regression.solve_sparse_logreg` to select the
#    best `k` features using a quantum-inspired algorithm.
# 4. Evaluated a logistic regression model on the selected features.
# 5. Used a train/validation split to sweep over values of `k` and selected
#    the best sparsity level without leaking test-set information.
# 6. Retrained on the full training set with the optimal `k` and measured
#    generalisation on the held-out test set.
#
# You can extend this workflow by adding cross-validation folds (see
# `StratifiedKFold` in scikit-learn) or by trying different values of
# `lambda_l2`.