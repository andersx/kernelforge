# High-Level Model API

KernelForge provides three ready-to-use model classes under `kernelforge.models`.
They follow a scikit-learn-style interface: `fit` / `predict` / `save` / `load`.

## Models

| Class | Representation | Kernel |
|---|---|---|
| `LocalKRRModel` | FCHL19 | Local Gaussian (KRR, exact) |
| `LocalRFFModel` | FCHL19 | Local Gaussian (RFF approximation) |
| `FCHL18KRRModel` | FCHL18 | FCHL18 analytical (KRR, exact) |

All three support the same three **training modes**, inferred automatically from
which of `energies` / `forces` you pass to `fit`:

| Mode | Training data | Predicts |
|---|---|---|
| `energy_only` | energies | E + F |
| `force_only` | forces | F + E |
| `energy_and_force` | energies + forces | E + F |

Forces must be **physical forces** F = −dE/dR. Sign handling is done internally.

---

## Quick start

```python
from kernelforge.models import LocalKRRModel

model = LocalKRRModel(sigma=2.0, l2=1e-9, elements=[1, 6, 8])

# fit — mode inferred automatically from what you pass
model.fit(coords_train, z_train, energies=E_train, forces=F_train)

# predict — always returns (energies, forces)
E_pred, F_pred = model.predict(coords_test, z_test)
# E_pred: shape (n_test,)
# F_pred: shape (n_test, n_atoms*3)

# save / load
model.save("model.npz")
model2 = LocalKRRModel.load("model.npz")
```

`coords_train` is a `list` of `(n_atoms, 3)` float64 arrays; `z_train` is a
`list` of `(n_atoms,)` int32 arrays of nuclear charges.

---

## LocalKRRModel

Kernel Ridge Regression using FCHL19 local representations.  Solves an
N×N (or N*(1+naq) × N*(1+naq) for energy+force) system with a Cholesky
solver on the symmetric RFP-packed kernel matrix.

```python
from kernelforge.models import LocalKRRModel

model = LocalKRRModel(
    sigma=2.0,          # Gaussian kernel length-scale
    l2=1e-9,            # L2 regularisation
    elements=[1, 6, 8], # atomic numbers present in the dataset
    repr_params={},     # extra kwargs forwarded to generate_fchl_acsf_and_gradients
)
```

**Key parameter notes**

- `sigma`: controls the kernel width. Typical range 1–20 for organic molecules.
- `l2`: regularisation on the kernel diagonal. Needs to be larger (`~1e-6`) for
  force-only and energy+force modes due to the less well-conditioned Hessian kernel.
- `elements`: must cover every atomic number in your dataset. Order does not matter.
- `repr_params`: pass any hyperparameter accepted by
  `fchl19_repr.generate_fchl_acsf_and_gradients`, e.g. `nRs2`, `eta2`, `rcut`.

**Training modes and kernels used**

| Mode | Training kernel | Prediction kernels |
|---|---|---|
| `energy_only` | `kernel_gaussian_symm_rfp` | scalar + Jacobian-transpose |
| `force_only` | `kernel_gaussian_hessian_symm_rfp` | Hessian + Jacobian |
| `energy_and_force` | `kernel_gaussian_full_symm_rfp` | full combined |

---

## LocalRFFModel

Random Fourier Features regression using FCHL19 local representations.
Approximates the local Gaussian kernel via the Bochner / Rahimi-Recht
elemental feature mapping with separate random weights per element type.
Solves a `d_rff × d_rff` normal-equations system — much cheaper than
the exact N×N KRR system for large datasets.

```python
from kernelforge.models import LocalRFFModel

model = LocalRFFModel(
    sigma=20.0,         # Gaussian kernel length-scale
    l2=1e-6,            # L2 regularisation on the normal-equations diagonal
    d_rff=4096,         # number of random Fourier features
    seed=42,            # random seed for reproducibility
    elements=[1, 6, 8],
    repr_params={},
)
```

**Key parameter notes**

- `sigma`: typically larger than for exact KRR (range 5–50).
- `d_rff`: more features → better approximation. 2048–8192 is a typical range.
- `seed`: fixing the seed guarantees identical random weights across runs.

---

## FCHL18KRRModel

Kernel Ridge Regression using the FCHL18 analytical kernel, which operates
directly on raw Cartesian coordinates (no pre-computed descriptor array).

```python
from kernelforge.models import FCHL18KRRModel

model = FCHL18KRRModel(
    sigma=2.5,        # Gaussian kernel length-scale
    l2=1e-8,          # L2 regularisation
    max_size=23,      # padding dimension (>= max atoms in any molecule)
    kernel_params={   # FCHL18 kernel hyperparameters (optional)
        "two_body_scaling": 2.828,
        "three_body_scaling": 1.6,
        "two_body_width": 0.2,
        "three_body_width": 3.14159,
        "two_body_power": 4.0,
        "three_body_power": 2.0,
        "cut_start": 1.0,
        "cut_distance": 5.0,
        "fourier_order": 1,
        "use_atm": True,
    },
)
```

**Key parameter notes**

- `max_size`: set to the maximum atom count in your dataset.
- `l2`: force-containing modes require larger values (~`1e-4`) because the
  FCHL18 Hessian kernel is only approximately PSD.
- `use_atm` / `cut_start`: the Hessian kernel requires `use_atm=False` and
  `cut_start >= 1.0`. These constraints are **enforced automatically** when
  `force_only` or `energy_and_force` modes are detected — you do not need to
  set them manually.

---

## Saving and loading

Models are serialised to NumPy `.npz` files. A `.npz` extension is added
automatically if absent.

```python
model.save("my_model.npz")          # saves to my_model.npz
model.save("my_model")              # also saves to my_model.npz

loaded = LocalKRRModel.load("my_model.npz")
E_pred, F_pred = loaded.predict(coords_test, z_test)
```

The saved file contains all hyperparameters, training representations, and
model weights — everything needed for inference without re-fitting.

---

## Pre-computed representations (`fit_from_repr`)

If you have already computed representations (e.g. from a grid search), you
can skip the representation step:

> **Note**: `fit_from_repr` / `predict_from_repr` are not yet implemented in
> the current version. Use `fit` and `predict` with raw coordinates.

---

## Hyperparameter optimisation

The `examples/` directory contains three Optuna-based optimisation scripts
that tune all representation and model hyperparameters jointly:

| Script | Model |
|---|---|
| `examples/optimize_krr_fchl19.py` | `LocalKRRModel` |
| `examples/optimize_rff_fchl19.py` | `LocalRFFModel` |
| `examples/optimize_fchl18_krr.py` | `FCHL18KRRModel` |

All three use the rMD17 ethanol dataset (split 01) and minimise
`energy_MAE + force_MAE` on the test set. Requires `optuna`:

```bash
uv pip install optuna
uv run python examples/optimize_krr_fchl19.py --trials 100
```

---

## Force shape conventions

Forces passed to `fit` and returned by `predict` use the **physical force
convention** F = −dE/dR. Any sign conversion required by the underlying
kernel is handled internally.

Forces can be provided in several shapes:

```python
# (n_mols, n_atoms, 3)  3D array
model.fit(coords, z, forces=F_3d)

# (n_mols, n_atoms*3)  2D array (already flattened)
model.fit(coords, z, forces=F_2d)

# list of per-molecule arrays, each (n_atoms_i, 3) or (n_atoms_i*3,)
model.fit(coords, z, forces=F_list)
```

Predicted forces are always returned as shape `(n_test, n_atoms*3)`.
