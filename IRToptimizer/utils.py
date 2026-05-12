from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def gen_data(
    N: int,
    J: int,
    r_true: int,
    seed: Optional[int] = None,
    sigma: float = 1.0,
    a_zero_prop: float = 0.0,
    spiky: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Simulate binary responses from a low-rank logistic (2PL-style) model.

    linear_arg = theta @ a.T + b; Y ~ Bernoulli(sigmoid(linear_arg / sigma)).
    """
    rng = np.random.default_rng(seed)
    k_dim = r_true - 1
    a = rng.uniform(0.5, 1.0, size=(J, k_dim))
    n_a = J * k_dim
    pz = min(1.0, max(0.0, float(a_zero_prop)))
    n_zero = min(n_a, max(0, int(math.floor(pz * n_a))))
    if n_zero > 0:
        zero_pos = rng.choice(n_a, size=n_zero, replace=False)
        a.reshape(-1)[zero_pos] = 0.0
    if spiky:
        theta = rng.standard_t(df=1.0, size=(N, k_dim))
    else:
        theta = rng.normal(0.0, 1.0, size=(N, k_dim))
    b = rng.normal(0.0, 0.5, size=J)
    linear_arg = theta @ a.T + b[None, :]
    m_true = linear_arg
    p = 1.0 / (1.0 + np.exp(-linear_arg / sigma))
    y0 = rng.binomial(1, p, size=(N, J)).astype(float)
    return {"Y0": y0, "M_true": m_true, "theta": theta, "a": a, "b": b}


def obj_1bit(d, m, f):
    """
    Negative log-likelihood for 1-bit matrix completion.

    Corresponds to R obj_1bit():
        m1 <- m[d==1]; m0 <- m[d==0]
        obj <- -sum(log(f(m1))) - sum(log(1-f(m0)))

    Parameters
    ----------
    d : ndarray  — observations D = (1+Y)/2, entries in {0, 1}
    m : ndarray  — predicted values at observed entries
    f : callable — CDF of noise (logistic: expit(·/σ))

    Note: when f(m)=0 or 1-f(m)=0, log(0)=-inf and the sum is +inf.
    R handles this silently; np.errstate(divide='ignore') replicates
    that behaviour without raising RuntimeWarning.
    """
    m1 = m[d == 1]
    m0 = m[d == 0]
    with np.errstate(divide="ignore"):
        return -np.sum(np.log(f(m1))) - np.sum(np.log(1 - f(m0)))


def hellinger_distance(p: np.ndarray, q: np.ndarray, eps: float = 1.0e-12) -> float:
    p_clip = np.clip(np.asarray(p, dtype=float).reshape(-1), eps, 1.0 - eps)
    q_clip = np.clip(np.asarray(q, dtype=float).reshape(-1), eps, 1.0 - eps)
    return float(np.sqrt(np.mean((np.sqrt(p_clip) - np.sqrt(q_clip)) ** 2)))


def rank_corr_mat(hat: np.ndarray, true: np.ndarray) -> float:
    hat_arr = np.asarray(hat, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    if hat_arr.shape != true_arr.shape:
        return float("nan")
    xh = hat_arr.reshape(-1)
    xt = true_arr.reshape(-1)
    if xh.size < 2:
        return float("nan")
    rho = pd.Series(xt).corr(pd.Series(xh), method="spearman")
    if rho is None or not np.isfinite(rho):
        return float("nan")
    return float(rho)


def compute_metrics(
    theta_hat: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    m_recovered: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    m_true: np.ndarray,
    y0_flat: np.ndarray,
    val_omega: np.ndarray,
    sigma: float,
    val_ratio: float,
    spiky: bool,
    run_time_sec: float,
    final_loss: float,
    iterations: float,
    max_iter: int,
) -> Dict[str, float]:
    """Same metric definitions as ``val_py-irt.py`` / ``val_cbmm_lbfgs.py``."""
    ok_theta_hat = isinstance(theta_hat, np.ndarray) and np.isfinite(theta_hat).all()
    ok_a_hat = isinstance(a_hat, np.ndarray) and np.isfinite(a_hat).all()
    ok_b_hat = isinstance(b_hat, np.ndarray) and np.isfinite(b_hat).all()
    ok_m = isinstance(m_recovered, np.ndarray) and np.isfinite(m_recovered).all()
    ok_truth = (
        np.isfinite(theta).all()
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.isfinite(m_true).all()
    )

    rmse_theta = float(np.sqrt(np.mean((theta_hat - theta) ** 2))) if ok_theta_hat else float("nan")
    rmse_a = float(np.sqrt(np.mean((a_hat - a) ** 2))) if ok_a_hat else float("nan")
    theta_rank_corr = rank_corr_mat(theta_hat, theta)
    a_rank_corr = rank_corr_mat(a_hat, a)
    b_rank_corr = rank_corr_mat(np.asarray(b_hat)[:, None], np.asarray(b)[:, None])
    rmse_b = float(np.sqrt(np.mean((b_hat - b) ** 2))) if ok_b_hat else float("nan")

    fnorm_diff = float(np.linalg.norm(m_recovered - m_true, ord="fro")) if ok_m and ok_truth else float("nan")
    fnorm_true = float(np.linalg.norm(m_true, ord="fro")) if ok_truth else float("nan")
    m_rel_fnorm = (
        float(fnorm_diff / fnorm_true)
        if np.isfinite(fnorm_true) and fnorm_true > 0.0 and np.isfinite(fnorm_diff)
        else float("nan")
    )

    if ok_m and ok_truth:
        with np.errstate(over="ignore", invalid="ignore"):
            p_recovered = 1.0 / (1.0 + np.exp(-m_recovered / sigma))
            p_true = 1.0 / (1.0 + np.exp(-m_true / sigma))
        hellinger = hellinger_distance(p_recovered, p_true)
    else:
        hellinger = float("nan")

    if ok_m and val_omega.size > 0:
        m_flat = np.ravel(m_recovered, order="F")
        m_val = m_flat[val_omega]
        y0_val = y0_flat[val_omega]
        yhat_binary = (m_val >= 0).astype(float)
        class_error_val = float(np.mean(y0_val != yhat_binary))
    else:
        class_error_val = float("nan")

    converged = bool(np.isfinite(iterations) and iterations < max_iter)
    return {
        "rmse_theta": rmse_theta,
        "rmse_a": rmse_a,
        "theta_rank_corr": theta_rank_corr,
        "a_rank_corr": a_rank_corr,
        "rmse_b": rmse_b,
        "b_rank_corr": b_rank_corr,
        "m_rel_fnorm": m_rel_fnorm,
        "hellinger_distance": hellinger,
        "class_error_val": class_error_val,
        "missing_rate": float(val_ratio),
        "run_time_sec": float(run_time_sec),
        "final_loss": float(final_loss) if np.isfinite(final_loss) else float("nan"),
        "iterations": float(iterations) if np.isfinite(iterations) else float("nan"),
        "converged": converged,
        "spiky": bool(spiky),
    }


def normalize_missing_mechanism(missing_mechanism: str) -> str:
    mech = str(missing_mechanism).upper()
    if mech not in {"MCAR", "MAR", "MNAR_0"}:
        raise ValueError("missing_mechanism must be one of 'MCAR', 'MAR', or 'MNAR_0'")
    return mech


def _solve_mar_alpha(centered: np.ndarray, val_ratio: float, mar_slope: float) -> float:
    left = -40.0
    right = 40.0
    for _ in range(120):
        mid = 0.5 * (left + right)
        cur = np.mean(1.0 / (1.0 + np.exp(-(mid + mar_slope * centered))))
        if cur > val_ratio:
            right = mid
        else:
            left = mid
    return 0.5 * (left + right)


def build_missing_partition(
    y0: np.ndarray,
    val_ratio: float,
    missing_mechanism: str = "MCAR",
    mar_slope: float = 2.0,
    mnar_other_scale: float = 0.0,
    mnar_col_prob: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Aligned with ``val_py-irt.build_missing_partition`` (seeded RNG when ``seed`` is set)."""
    if not (0.0 <= float(val_ratio) <= 1.0):
        raise ValueError("val_ratio must be within [0, 1]")
    if not (0.0 <= float(mnar_col_prob) <= 1.0):
        raise ValueError("mnar_col_prob must be within [0, 1]")

    mech = normalize_missing_mechanism(missing_mechanism)
    N, J = y0.shape
    total = N * J
    rng = np.random.default_rng(seed)

    if mech == "MCAR":
        n_val = int(round(val_ratio * total))
        all_indices = rng.permutation(total)
        val_omega = np.sort(all_indices[:n_val]) if n_val > 0 else np.array([], dtype=int)
        mask_train_flat = np.ones(total, dtype=bool)
        mask_train_flat[val_omega] = False
    elif mech == "MAR":
        row_pos = np.linspace(0.0, 1.0, N) if N > 1 else np.zeros(N, dtype=float)
        col_pos = np.linspace(0.0, 1.0, J) if J > 1 else np.zeros(J, dtype=float)
        score_mat = (row_pos[:, None] + col_pos[None, :]) / 2.0
        centered = np.ravel(score_mat - np.mean(score_mat), order="F")
        if val_ratio <= 0.0:
            miss_prob = np.zeros(total, dtype=float)
        elif val_ratio >= 1.0:
            miss_prob = np.ones(total, dtype=float)
        else:
            alpha = _solve_mar_alpha(centered=centered, val_ratio=val_ratio, mar_slope=mar_slope)
            miss_prob = 1.0 / (1.0 + np.exp(-(alpha + mar_slope * centered)))
        miss_draw = rng.random(total) < miss_prob
        val_omega = np.where(miss_draw)[0]
        mask_train_flat = ~miss_draw
    else:
        y_vec = np.ravel(y0, order="F")
        zero_prob = np.full(total, val_ratio, dtype=float)
        other_prob = float(np.clip(val_ratio * mnar_other_scale, 0.0, 1.0))
        base_prob = np.where(y_vec == 0.0, zero_prob, other_prob)
        selected_cols = rng.binomial(1, mnar_col_prob, size=J).astype(int)
        col_gate = np.repeat(selected_cols, N)
        miss_prob = base_prob * col_gate
        miss_draw = rng.random(total) < miss_prob
        val_omega = np.where(miss_draw)[0]
        mask_train_flat = ~miss_draw

    mask_train = np.reshape(mask_train_flat, (N, J), order="F")
    y_vec = np.ravel(y0, order="F")
    miss_vec = ~mask_train_flat
    realized_rate = float(np.mean(miss_vec)) if total > 0 else float("nan")
    idx_y1 = y_vec == 1.0
    idx_y0 = y_vec == 0.0
    realized_rate_y1 = float(np.mean(miss_vec[idx_y1])) if np.any(idx_y1) else float("nan")
    realized_rate_y0 = float(np.mean(miss_vec[idx_y0])) if np.any(idx_y0) else float("nan")

    return {
        "mechanism": mech,
        "val_omega": val_omega.astype(int),
        "n_val": int(val_omega.size),
        "mask_train": mask_train,
        "ind_omega_train": mask_train_flat.astype(int),
        "realized_missing_rate": realized_rate,
        "realized_missing_rate_y1": realized_rate_y1,
        "realized_missing_rate_y0": realized_rate_y0,
    }
