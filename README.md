# An Interpretable and Scalable Framework for Evaluating Large Language Models
**Authors:** Xinhao Qu, Qiang Heng, Hao Zeng, Xiaoqian Liu (2026)

This repository contains all code needed to reproduce the experiments in the manuscript, along with the benchmark suite dataset `Y_MATH-500.csv`.

## Dependencies

Install dependencies first:

```bash
pip install -r requirement.txt
```

## Quick Start

```python
import numpy as np
from cBMM import cBMM

# Build an N x J binary matrix with entries in {-1, +1}
N, J, r = 50, 40, 5
Y = np.random.choice([-1.0, 1.0], size=(N, J))

result = cBMM(
    Y=Y,
    r=r,
    sigma=1.0,
    ind_omega=None,   # Pass a mask here for missing observations
    tol=1e-4,
    max_iter=500,
    verbose=20,
    num_threads='auto'
)

print("Iterations:", result["n_iter"])
print("X_hat shape:", result["X_hat"].shape)
```

Or run the demo script:

```bash
python demo.py
```



## Input

- `Y`: binary matrix, values are expected to be `-1` or `+1`
- `r`: factorization rank (latent dimension)
- `sigma`: scaling parameter in the loss
- `ind_omega`: optional observation mask, length must be `N*J` (column-major order)
- `init_U1`, `init_V1`, `init_v2`: optional initial values
- `tol`: relative convergence tolerance
- `max_iter`: maximum number of iterations
- `verbose`: print frequency (`0` disables printing)
- `num_threads`: `'auto'`, integer, or `None`

## Output

`cBMM(...)` returns a dictionary with:

- `U1`: left factor matrix, shape `(N, r)`
- `V1`: right factor matrix, shape `(J, r)`, non-negative
- `v2`: column bias vector, shape `(J,)`
- `X_hat`: reconstructed matrix, shape `(N, J)`
- `loss_history`: loss values over iterations
- `n_iter`: number of iterations executed
- `ind_omega`: input observation mask