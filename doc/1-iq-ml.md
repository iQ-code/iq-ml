# 1. iQ-ML

## 1.1. Overview

**iQ-ML** is Inspiration-Q's suite of sparse machine-learning solvers. All three solvers tackle the same fundamental challenge: **feature selection** — given a dataset with many features, find the small subset of $k$ features that best explains the target variable, while fitting a model on those features alone.

Classical solvers (e.g. LASSO) impose a soft sparsity penalty that can leave many small nonzero coefficients. iQ-ML solvers enforce an **exact cardinality constraint** ($\|w\|_0 = k$), producing solutions that are exactly $k$-sparse. This hard constraint is combinatorially hard in general; the solvers use a quantum-inspired algorithm to find high-quality solutions efficiently at scale.

| Solver | Task | API Endpoint |
|--------|------|-------------|
| **Sparse Linear Regression** | Regression with exactly k nonzero coefficients | `v1/iq-ml/linear-regression` |
| **Sparse Logistic Regression** | Binary classification selecting exactly k features | `v1/iq-ml/logistic-regression` |
| **Sparse FDR Regression** | Binary classification via Fisher Discriminant Ratio | `v1/iq-ml/sparse-fdr-regression` |

---

## 1.2. Sparse Linear Regression

### Problem Definition

Given a feature matrix $X \in \mathbb{R}^{n \times p}$ and an observation vector $y \in \mathbb{R}^n$, the sparse linear regression problem minimizes the mean squared error over all coefficient vectors with exactly $k$ nonzero entries:

$$
\min_{w \in \mathbb{R}^p} \quad \frac{1}{n} \|y - X w\|^2 + \lambda_{\ell_2} \|w\|^2
$$
$$
\text{subject to} \quad \|w\|_0 = k
$$

where $\lambda_{\ell_2} \geq 0$ is an optional L2 (Ridge) regularization parameter.

The inner minimization over the continuous weights $w$ for a fixed support (the set of $k$ nonzero indices) has a closed-form solution via ridge regression. The outer minimization over the support is combinatorial and is solved using a quantum-inspired algorithm.

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-ml/linear-regression
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `X` | array | Feature matrix of shape (n_samples, n_features); max 100000 × 2048 |
| `y` | array | Observation vector of length n_samples |
| `k` | integer | Number of nonzero coefficients (1 ≤ k ≤ n_features − 1) |
| `lambda_l2` | float | L2 regularization parameter (default: `0.0`) |
| `options.copies` | integer | Number of trajectories (default: `100`) |
| `options.tol` | float | Inner solver tolerance (default: `1e-6`) |
| `random_number_generator_seed` | integer | RNG seed (default: `123321`) |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Coefficient vector of length n_features; exactly k elements are nonzero |
| `cost` | float | Mean squared error: (1/n) ‖y − X w‖² |

### Example API Response

```json
{
  "computationId": "f03efc3c-cd8f-4341-a84e-000caa380f89",
  "status": "Ok",
  "computationTimeInSeconds": 3.21,
  "solution": [0.0, 2.03, 0.0, 0.0, 0.0, -1.47, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "cost": 0.011
}
```

---

## 1.3. Sparse Logistic Regression

### Problem Definition

Given a feature matrix $X \in \mathbb{R}^{n \times p}$ and a binary label vector $y \in \{-1, +1\}^n$, the sparse logistic regression problem minimizes the logistic loss with L2 regularization, selecting exactly $k$ features:

$$
\min_{w \in \mathbb{R}^p,\, b \in \mathbb{R}} \quad \frac{1}{n} \sum_{i=1}^{n} \log\!\left(1 + e^{-y_i (X_i w + b)}\right) + \lambda_{\ell_2} \|w\|^2
$$
$$
\text{subject to} \quad \|w\|_0 = k
$$

For a fixed support, the logistic model weights and intercept $b$ are fit by standard gradient-based optimization. The support search is performed by our quantum-inspired algorithm approach.

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-ml/logistic-regression
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `X` | array | Feature matrix of shape (n_samples, n_features); max 100000 × 2048 |
| `y` | array | Binary label vector; labels should be +1 / −1 or 1 / 0 |
| `k` | integer | Number of features to select (1 ≤ k ≤ n_features − 1) |
| `lambda_l2` | float | L2 regularization parameter (default: `0.0`) |
| `options.copies` | integer | Number of trajectories (default: `100`) |
| `options.tol` | float | Inner solver tolerance (default: `1e-6`) |
| `options.n_jobs` | integer | Parallel jobs for inner solver (default: `1`) |
| `random_number_generator_seed` | integer | RNG seed (default: `123321`) |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Integer vector of length k; indices of the selected features (0-indexed) |
| `weights` | array | Logit model coefficients; first element is the intercept, followed by k feature weights |

### Example API Response

```json
{
  "computationId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "Ok",
  "computationTimeInSeconds": 5.84,
  "solution": [1, 9],
  "weights": [-0.12, 1.87, -1.53]
}
```

---

## 1.4. Sparse FDR Regression

### Problem Definition

The Fisher Discriminant Ratio (FDR) criterion selects the $k$ features that best **linearly separate** two classes. For a projection direction $w$, the FDR is defined as:

$$
\text{FDR}(w) = \frac{\left(w^T (\mu_1 - \mu_0)\right)^2}{w^T (\Sigma_1 + \Sigma_0)\, w}
$$

where $\mu_c$ and $\Sigma_c$ are the class-conditional mean and covariance matrix of class $c \in \{0, 1\}$.

The sparse FDR problem finds the $k$-element support that maximizes the FDR:

$$
\max_{S \subseteq \{1,\ldots,p\},\, |S|=k} \quad \text{FDR}(w_S^*)
$$

where $w_S^*$ is the optimal FDR direction restricted to support $S$ (closed-form within $S$). A logit model is then fit on the selected features to produce a probabilistic classifier.

**When to prefer FDR over logistic regression:**
- When the number of samples is small relative to the number of features (high-dimensional regime), the FDR criterion has a closed-form inner solution and is numerically more stable.
- When class imbalance is present, FDR is less sensitive than logistic regression.

### API Endpoint

```
POST https://www.inspiration-q.com/api/v1/iq-ml/sparse-fdr-regression
```

### Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `X` | array | Feature matrix of shape (n_samples, n_features); max 100000 × 2048 |
| `y` | array | Binary label vector; labels should be +1 / −1 or 1 / 0 |
| `k` | integer | Number of features to select (1 ≤ k ≤ n_features − 1) |
| `lambda_l2` | float | L2 regularization added to within-class covariances (default: `0.0`) |
| `options.copies` | integer | Number of trajectories (default: `100`) |
| `options.tol` | float | Inner solver tolerance (default: `1e-6`) |
| `options.n_jobs` | integer | Parallel jobs (default: `1`) |
| `random_number_generator_seed` | integer | RNG seed (default: `123321`) |
| `description` | string | Optional label |

### Output Parameters

| Field | Type | Description |
|-------|------|-------------|
| `solution` | array | Integer vector of length k; indices of the selected features (0-indexed) |
| `weights` | array | Logit model coefficients; first element is the intercept, followed by k feature weights |
