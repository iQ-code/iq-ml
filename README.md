# iQ-ML

Welcome to **iQ-ML**, the SDK and documentation for Inspiration-Q's sparse machine-learning solvers.

---

## Documentation

All details are explained in the `doc/` folder:

1. **[iQ-ML](doc/1-iq-ml.md)** — Problem definitions, API endpoints, and input/output reference for all solvers.
2. **[Accessing the API](doc/2-accessing-the-api.md)** — cURL and Python examples for interacting with the REST API directly.
3. **[iQ-ML SDK](doc/3-iq-ml-sdk.md)** — Installing and using the Python SDK.

---

## Solvers

| Module | Solver | Entry point |
|--------|--------|-------------|
| `iq.ml.linear_regression` | Sparse linear regression | `v1/iq-ml/linear-regression` |
| `iq.ml.logistic_regression` | Sparse logistic regression | `v1/iq-ml/logistic-regression` |
| `iq.ml.fdr_regression` | Sparse FDR regression | `v1/iq-ml/sparse-fdr-regression` |

---

## Installation

```bash
pip install .
```

Or using make:
```bash
make install
```

---

## Quick Start

```python
import numpy as np
import iq.api.iqrestapi
import iq.ml.linear_regression

iq.api.iqrestapi.initialize_credentials("YOUR_API_KEY")

X = np.random.randn(200, 20)
true_w = np.zeros(20)
true_w[[3, 11]] = [2.0, -1.5]
y = X @ true_w + 0.1 * np.random.randn(200)

w, mse = iq.ml.linear_regression.solve_sparse_linreg(X, y, k=2)
print("Selected features:", np.where(np.array(w) != 0)[0].tolist())
print("MSE:", mse)
```

---

## Resources & Support

- **Documentation:** [`doc/`](./doc/) folder
- **Official Website:** [Inspiration-Q](https://www.inspiration-q.com)
- **Contact Support:** [support@inspiration-q.com](mailto:support@inspiration-q.com)
