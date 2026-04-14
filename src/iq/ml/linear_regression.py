"""Sparse linear regression solver."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_feature_matrix(
    X: NDArray[np.float64], max_n_samples: int, max_n_features: int
) -> list[list[float]]:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] > max_n_samples or X.shape[1] > max_n_features:
        raise Exception("Not a valid feature matrix")
    return X.tolist()


def _validate_observation_vector(
    y: NDArray[np.float64], X: NDArray[np.float64]
) -> list[float]:
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


def solve_sparse_linreg(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int,
    lambda_l2: float = 0.0,
    options: dict | None = None,
    random_number_generator_seed: int = 123321,
    description: str = "",
) -> tuple[NDArray[np.float64], float]:
    """Solve a sparse linear regression problem with exactly k nonzero coefficients.

    Given a feature matrix X of shape (n_samples, n_features) and an
    observation vector y of length n_samples, this solver finds a coefficient
    vector w with exactly k nonzero entries that minimizes the mean squared
    error (MSE) with an optional L2 regularization term:

        MSE(w) = (1/n) ||y - X w||^2 + lambda_l2 * ||w||^2

    subject to: count(w_i != 0) = k

    This problem is NP-hard in general due to the cardinality constraint.
    The solver uses a quantum inspired algorithm to find high-quality approximate
    solutions efficiently.

    Parameters
    ----------
    X : NDArray[np.float64]
        Real feature matrix of shape (n_samples, n_features).
        Maximum supported size: 100000 samples x 2048 features.
    y : NDArray[np.float64]
        Real observation vector of length n_samples.
    k : int
        Number of nonzero coefficients in the solution (between 1 and
        n_features - 1).
    lambda_l2 : float, default=0.0
        L2 (Ridge) regularization parameter. Larger values shrink the
        nonzero coefficients toward zero.
    options : dict, default=None
        Optimization hyperparameters. Accepted keys:

        - copies : int, default=100
            Number of stochastic trajectories (between 1 and 500).
        - tol : float, default=1e-6
            Relative error criterion to stop the inner linear solver.
    random_number_generator_seed : int, default=123321
        Seed for the random number generator.
    description : str, default=""
        Descriptive name of the computation.

    Returns
    -------
    s : NDArray[np.float64]
        Array of length k containing the indices of the selected features.
    MSE : float
        Mean squared error of the solution on the training data.

    """
    options = dict(options) if options is not None else {}
    # Validate arguments inside 'options'.
    options["copies"] = int(validate.integer(options.get("copies", 100), 1, 500))
    options["tol"] = float(validate.real(options.get("tol", 1e-6)))

    validate.dictionary(options)

    r = iqrestapi.post(
        "v1/iq-ml/linear-regression",
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
    return r["solution"], r["cost"]
