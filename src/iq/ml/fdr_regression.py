"""Sparse classification solver using the Fisher Discriminant Ratio (FDR)."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_feature_matrix(X: NDArray, max_n_samples: int, max_n_features: int):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] > max_n_samples or X.shape[1] > max_n_features:
        raise Exception("Not a valid feature matrix")
    num_digits_for_single_precision = 8
    return np.round(X, num_digits_for_single_precision).tolist()


def _validate_observation_vector(y: NDArray, X: NDArray):
    if y is not None:
        y = np.asarray(y)
        X = np.asarray(X)
        if y.ndim != 1:
            raise Exception("Not a valid observations vector")
        if y.shape[0] != X.shape[0]:
            raise Exception(
                f"Number of observations in y: {y.shape[0]} and X: {X.shape[0]} does not match"
            )
        y = y.tolist()
    return y


def solve_fdr_pa(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int,
    lambda_l2: float = 0.0,
    options: dict | None = None,
    random_number_generator_seed: int = 123321,
    description: str = "",
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Solve a sparse classification problem using the Fisher Discriminant Ratio.

    Given a feature matrix X of shape (n_samples, n_features) and a binary
    label vector y, this solver selects k features that maximize the Fisher
    Discriminant Ratio (FDR) - the ratio of between-class variance to
    within-class variance. It then fits a linear classifier on those features.

    The FDR criterion for a projection direction w is:

        FDR(w) = (w^T (mu_1 - mu_0))^2 / (w^T (Sigma_1 + Sigma_0) w)

    where mu_c and Sigma_c are the class-conditional mean and covariance.

    Maximizing the FDR finds the k features that best separate the two classes
    in a linear sense. A logit model is then fit on the selected features
    to produce a probabilistic classifier.

    This criterion is often preferable to logistic regression when the number
    of samples is small relative to the number of features (high-dimensional
    settings), because it has a closed-form solution for fixed feature
    support and is less sensitive to class imbalance.

    Parameters
    ----------
    X : NDArray[np.float64]
        Real feature matrix of shape (n_samples, n_features).
        Maximum supported size: 100000 samples x 2048 features.
    y : NDArray[np.float64]
        Binary label vector of length n_samples. Labels should be +1 or -1
        (or 1 and 0).
    k : int
        Number of nonzero features to select (between 1 and n_features - 1).
    lambda_l2 : float, default=0.0
        L2 regularization parameter added to the within-class covariance
        matrices to improve numerical stability.
    options : dict, default=None
        Optimization hyperparameters. Accepted keys:

        - copies : int, default=100
            Number of stochastic trajectories (between 1 and 500).
        - tol : float, default=1e-6
            Relative error criterion to stop the solver.
    random_number_generator_seed : int, default=123321
        Seed for the random number generator.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    solution : NDArray[np.int32]
        Integer vector of length k containing the indices of the selected
        features (0-indexed).
    weights : NDArray[np.float64]
        Coefficient vector for the logit model fit on the selected features.
        The first element is the intercept (bias), followed by k weights.

    """
    options = dict(options) if options is not None else {}
    # Validate arguments inside 'options'.
    options["copies"] = int(validate.integer(options.get("copies", 100), 1, 500))
    options["tol"] = float(validate.real(options.get("tol", 1e-6)))

    validate.dictionary(options)

    r = iqrestapi.post(
        "v1/iq-ml/sparse-fdr-regression",
        json={
            "X": _validate_feature_matrix(X, 100_000, 2048),
            "y": _validate_observation_vector(y, X),
            "k": validate.integer(k, 1, np.asarray(X).shape[1] - 1),
            "lambda_l2": validate.real(lambda_l2),
            "options": options,
            "random_number_generator_seed": validate.integer(
                random_number_generator_seed, 0, 0xFFFFFFF
            ),
            "description": validate.string(description),
        },
    )
    return r["solution"], r["weights"]
