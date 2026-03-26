# 3. iQ-ML SDK

The **Inspiration-Q SDK** simplifies interactions with the iQ-ML API by handling HTTP calls and result polling inside a single Python function call.

## 3.1. Installing a Python Environment

Create an isolated Python environment and install the required dependencies:

### 1. Using `conda`
```bash
conda create --name iq-ml python=3.11 numpy requests
conda activate iq-ml
```

### 2. Using `venv` (Standard Library)
```bash
python -m venv iq-ml
source iq-ml/bin/activate   # macOS / Linux
# On Windows: iq-ml\Scripts\activate
pip install numpy requests
```

### 3. Using `virtualenv`
```bash
pip install virtualenv
virtualenv iq-ml
source iq-ml/bin/activate
pip install numpy requests
```

### 4. Using `pipenv`
```bash
pip install pipenv
pipenv --python 3.11
pipenv install numpy requests
pipenv shell
```

### 5. Using `poetry`
```bash
pip install poetry
poetry new iq-ml && cd iq-ml
poetry add numpy requests
poetry shell
```

---

## 3.2. SDK Setup

### 1. Using `make install`
```bash
make install
```

### 2. Using `pip install .`
```bash
pip install .
```

### 3. Using `pip install -r requirements.txt`
```bash
pip install -r requirements.txt
```

---

## 3.3. Initializing the SDK

All solvers require initializing the SDK with your API key before the first call:

```python
import iq.api.iqrestapi

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")
```

This only needs to be called once per session.

---

## 3.4. Sparse Linear Regression

```python
import numpy as np
import iq.api.iqrestapi
import iq.ml.linear_regression

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(0)
n_samples, n_features, k = 300, 20, 4

X = rng.standard_normal((n_samples, n_features))

# True signal on 4 features only
true_w = np.zeros(n_features)
true_w[[1, 7, 12, 17]] = [2.0, -1.5, 0.8, -1.2]
y = X @ true_w + 0.1 * rng.standard_normal(n_samples)

w, mse = iq.ml.linear_regression.solve_sparse_linreg(
    X, y,
    k=k,
    lambda_l2=0.0,
    options={"copies": 100},
    random_number_generator_seed=42,
    description="Sparse linreg example",
)

w = np.array(w)
print("Selected features:", np.where(w != 0)[0].tolist())
print("MSE:", mse)
```

---

## 3.5. Sparse Logistic Regression

```python
import numpy as np
import iq.api.iqrestapi
import iq.ml.logistic_regression

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(1)
n_samples, n_features, k = 400, 15, 2

X = rng.standard_normal((n_samples, n_features))

# True signal on features 3 and 9
y = np.sign(X[:, 3] - X[:, 9] + 0.3 * rng.standard_normal(n_samples))
y[y == 0] = 1

selected, weights = iq.ml.logistic_regression.solve_logreg_pa(
    X, y,
    k=k,
    lambda_l2=0.01,
    options={"copies": 100},
    random_number_generator_seed=42,
    description="Sparse logreg example",
)

print("Selected features:", list(selected))
print("Model weights (intercept + k features):", np.round(weights, 4))
```

---

## 3.6. Sparse FDR Regression

```python
import numpy as np
import iq.api.iqrestapi
import iq.ml.fdr_regression

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

rng = np.random.default_rng(2)
n_samples, n_features, k = 200, 30, 3

X = rng.standard_normal((n_samples, n_features))

# True signal on features 5, 14, 22
y = np.sign(X[:, 5] - 0.8 * X[:, 14] + 0.5 * X[:, 22]
            + 0.3 * rng.standard_normal(n_samples))
y[y == 0] = 1

selected, weights = iq.ml.fdr_regression.solve_fdr_pa(
    X, y,
    k=k,
    lambda_l2=0.01,
    options={"copies": 100},
    random_number_generator_seed=42,
    description="Sparse FDR example",
)

print("Selected features:", list(selected))
print("Model weights (intercept + k features):", np.round(weights, 4))
```

The FDR criterion is particularly effective in high-dimensional settings where
the number of features is large relative to the number of samples.

You can find more complete examples in the [examples/](../examples/) folder.
