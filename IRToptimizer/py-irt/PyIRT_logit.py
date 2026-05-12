from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyro
    import torch
    from py_irt.config import IrtConfig
    from py_irt.dataset import Dataset
    from py_irt.initializers import IrtInitializer
    from py_irt.training import IrtModelTrainer
except Exception as exc:  # pragma: no cover
    pyro = None
    torch = None
    IrtConfig = None
    Dataset = None
    IrtInitializer = object
    IrtModelTrainer = None
    _PYIRT_IMPORT_ERROR = exc
else:
    _PYIRT_IMPORT_ERROR = None


def make_init_method1(m: int, n: int, r_true: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    u0 = rng.normal(size=(m, r_true))
    u0[:, r_true - 1] = 1.0
    v0 = rng.normal(size=(n, r_true))
    if r_true > 1:
        v0[:, : (r_true - 1)] = np.exp(v0[:, : (r_true - 1)])
    return {
        "init_U1": u0[:, : (r_true - 1)].copy(),
        "init_V1": v0[:, : (r_true - 1)].copy(),
        "init_v2": v0[:, r_true - 1].copy(),
    }


class V0ItemInitializer(IrtInitializer):
    """Set py-irt item params from R-style V0 init."""

    def __init__(self, dataset: Dataset, a_init: np.ndarray, d_init: np.ndarray):
        super().__init__(dataset)
        self.a_init = np.asarray(a_init, dtype=float)
        if self.a_init.ndim == 1:
            self.a_init = self.a_init[:, None]
        self.d_init = np.asarray(d_init, dtype=float).reshape(-1)

    def initialize(self) -> None:
        store = pyro.get_param_store()
        if "loc_diff" not in store:
            return

        eps = 1.0e-8
        a = np.clip(self.a_init, eps, None)
        n_dim = a.shape[1]
        if n_dim == 1:
            denom = np.clip(a[:, 0], eps, None)
            diff = (-self.d_init / denom)[:, None]
        else:
            denom = np.sum(a, axis=1)
            denom = np.where(np.abs(denom) < eps, eps, denom)
            diff_scalar = -self.d_init / denom
            diff = np.repeat(diff_scalar[:, None], n_dim, axis=1)

        loc_diff = pyro.param("loc_diff")
        diff_tensor = torch.as_tensor(
            diff[:, 0] if loc_diff.ndim == 1 else diff,
            dtype=loc_diff.dtype,
            device=loc_diff.device,
        )

        if "loc_slope" in store:
            loc_disc = pyro.param("loc_slope")
        elif "loc_disc" in store:
            loc_disc = pyro.param("loc_disc")
        else:
            loc_disc = None

        with torch.no_grad():
            loc_diff.copy_(diff_tensor)
            if loc_disc is not None:
                log_a = np.log(np.clip(a, eps, None))
                disc_tensor = torch.as_tensor(
                    log_a[:, 0] if loc_disc.ndim == 1 else log_a,
                    dtype=loc_disc.dtype,
                    device=loc_disc.device,
                )
                loc_disc.copy_(disc_tensor)


def _extract_ordered_array(
    raw_values: object,
    id_map: Optional[dict],
    wanted_ids: List[str],
    expect_2d: bool,
) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(raw_values, dtype=float)
    except Exception:
        return None

    if arr.ndim == 1:
        arr = arr[:, None]
    if expect_2d and arr.ndim != 2:
        return None
    if not expect_2d and arr.ndim > 2:
        return None

    if id_map is None:
        return arr

    try:
        value_to_idx = {str(v): int(k) for k, v in id_map.items()}
        order = [value_to_idx[str(v)] for v in wanted_ids]
    except Exception:
        return None

    if len(order) != arr.shape[0]:
        return None
    return arr[order, :]


def pyirt_logit(
    seed_val: int,
    y: np.ndarray,
    m: int,
    n: int,
    r_true: int,
    sigma_val: float,
    max_iter_val: int,
) -> Dict[str, np.ndarray]:
    del sigma_val  # kept for parity with R function signature

    n_dim = r_true - 1
    na_theta = np.full((m, n_dim), np.nan, dtype=float)
    na_a = np.full((n, n_dim), np.nan, dtype=float)
    na_b = np.full(n, np.nan, dtype=float)
    if n_dim <= 0:
        return {"runtime": np.nan, "theta": na_theta, "a": na_a, "b": na_b}
    if _PYIRT_IMPORT_ERROR is not None:
        return {"runtime": np.nan, "theta": na_theta, "a": na_a, "b": na_b}

    item_names = [f"item_{j + 1}" for j in range(n)]
    subject_ids = [f"subject_{i + 1}" for i in range(m)]
    y_df = pd.DataFrame(y, columns=item_names)
    y_df.insert(0, "subject_id", subject_ids)

    ik = make_init_method1(m, n, r_true, seed_val)
    v0 = np.column_stack([ik["init_V1"], ik["init_v2"]])

    model_type = "2pl" if n_dim == 1 else "multidim_2pl"
    fit_seed = seed_val + 20000
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)
    pyro.set_rng_seed(fit_seed)

    try:
        dataset = Dataset.from_pandas(
            y_df,
            subject_column="subject_id",
            item_columns=item_names,
        )
        config_kwargs = {
            "model_type": model_type,
            "priors": "vague",
            "epochs": int(max_iter_val),
            "log_every": max(100, int(max_iter_val // 10)),
            "seed": fit_seed,
            "deterministic": False,
        }
        if n_dim > 1:
            config_kwargs["dims"] = int(n_dim)
        config = IrtConfig(**config_kwargs)
        trainer = IrtModelTrainer(
            config=config,
            data_path=Path("."),
            dataset=dataset,
            verbose=False,
        )
        trainer._initializers.append(
            V0ItemInitializer(
                dataset=dataset,
                a_init=v0[:, :n_dim],
                d_init=v0[:, r_true - 1],
            )
        )

        t_fit = time.perf_counter()
        trainer.train(epochs=int(max_iter_val), device="cpu")
        run_time = time.perf_counter() - t_fit
        params = trainer.last_params
    except Exception:
        return {"runtime": np.nan, "theta": na_theta, "a": na_a, "b": na_b}

    theta_arr = _extract_ordered_array(
        params.get("ability") if isinstance(params, dict) else None,
        params.get("subject_ids") if isinstance(params, dict) else None,
        subject_ids,
        expect_2d=True,
    )
    a_arr = _extract_ordered_array(
        params.get("disc") if isinstance(params, dict) else None,
        params.get("item_ids") if isinstance(params, dict) else None,
        item_names,
        expect_2d=True,
    )
    diff_arr = _extract_ordered_array(
        params.get("diff") if isinstance(params, dict) else None,
        params.get("item_ids") if isinstance(params, dict) else None,
        item_names,
        expect_2d=True,
    )

    extract_ok = (
        theta_arr is not None
        and a_arr is not None
        and diff_arr is not None
        and theta_arr.shape[0] == m
        and a_arr.shape[0] == n
        and diff_arr.shape[0] == n
        and theta_arr.shape[1] >= n_dim
        and a_arr.shape[1] >= n_dim
        and diff_arr.shape[1] >= n_dim
    )
    if not extract_ok:
        return {"runtime": np.nan, "theta": na_theta, "a": na_a, "b": na_b}

    theta_hat = theta_arr[:, :n_dim]
    a_hat = a_arr[:, :n_dim]
    diff_hat = diff_arr[:, :n_dim]
    # py-irt uses logits = sum_k a_k * (theta_k - diff_k), while mirt exports intercept d.
    b_hat = -np.sum(a_hat * diff_hat, axis=1)

    if not (
        np.all(np.isfinite(theta_hat))
        and np.all(np.isfinite(a_hat))
        and np.all(np.isfinite(b_hat))
    ):
        return {"runtime": np.nan, "theta": na_theta, "a": na_a, "b": na_b}

    return {"runtime": float(run_time), "theta": theta_hat, "a": a_hat, "b": b_hat}
