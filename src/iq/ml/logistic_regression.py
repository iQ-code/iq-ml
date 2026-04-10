"""Sparse classification solver using logistic regression."""

import numpy as np
from numpy.typing import NDArray

from iq.api import iqrestapi, validate


def _validate_feature_matrix(X, max_n_samples, max_n_features):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] > max_n_samples or X.shape[1] > max_n_features:
        raise Exception("Not a valid feature matrix")
    num_digits_for_single_precision = 8
    return np.round(X, num_digits_for_single_precision).tolist()


def _validate_observation_vector(y, X):
    if y is not None:
        y = np.asarray(y)
        X = np.asarray(X)
        if y.ndim != 1:
            raise Exception("Not a valid observations vector")
        if y.shape[0] != X.shape[0]:
            raise Exception(
                f"Number of observations in y: {y.shape[0]} and X: {X.shape[0]} does not match"
            )
        if np.any(y == -1):
            y = np.where(y == -1, 0, y)
        y = y.tolist()
    return y


def solve_sparse_logreg(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int,
    lambda_l2: float = 0.0,
    accelerate: bool = True,
    options: dict | None = None,
    random_number_generator_seed: int = 123321,
    description: str = "",
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Solve a sparse logistic regression problem with exactly k nonzero features.

    Given a feature matrix X of shape (n_samples, n_features) and a binary label vector y, this
    solver selects k features and fits a logistic regression model using only those features. The
    solver minimizes the logistic cross-entropy with an optional L2 regularization term:

        L(w, b) = (1/n) sum_i y_i * log(p_i) + (1 - y_i) * log(1 - p_i) + lambda_l2 * ||w||^2

    with p_i = 1 / (1 + exp(X_i @ w + b)) and subject to: count(w_j != 0) = k

    The feature selection and model fitting are performed jointly using a quantum-inspired
    algorithm, finding the k features that together give the best classification performance.

    If the `accelerate` algorithm is set to True, then the quantum-inspired algorithm does the 
    solution exploration method using the Fisher Discriminant Ratio as a simpler classification
    algorithm that is much faster to compute.

    Parameters
    ----------
    X : NDArray[np.float64]
        Real feature matrix of shape (n_samples, n_features).
        Maximum supported size: 100000 samples x 2048 features.
    y : NDArray[np.float64]
        Binary label vector of length n_samples. Labels should be either (-1,+1) or (0,1).
    k : int
        Number of nonzero features to select (between 1 and n_features - 1).
    lambda_l2 : float, default=0.0
        L2 (Ridge) regularization parameter for the logistic regression weights.
    accelerate : bool, default=True
        If True, then the solver uses a simpler classifier, FDR [see above], to do the
        exploration of the optimal solutions. This method is much faster than using 
        only logistic regression, but might find less optimal results.
    options : dict, default=None
        Optimization hyperparameters. Accepted keys:

        - copies : int, default=100
            Number of stochastic trajectories (between 1 and 500).
        - tol : float, default=1e-6
            Relative error criterion to stop the inner logistic regression solver.
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
        Coefficient vector for the logistic model. The first element is the
        intercept (bias), followed by k weights corresponding to the selected
        features.

    """
    options = dict(options) if options is not None else {}
    # Validate arguments inside 'options'.
    options["copies"] = int(validate.integer(options.get("copies", 100), 1, 500))
    options["tol"] = float(validate.real(options.get("tol", 1e-6)))

    validate.dictionary(options)

    url = (
        "v1/iq-ml/sparse-fdr-regression" 
        if accelerate 
        else "v1/iq-ml/logistic-regression"
    )
    r = iqrestapi.post(
        url,
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
