from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from utils import build_missing_partition, compute_metrics, gen_data


ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CONFIG_FILE = ROOT_DIR / "runner_config.json"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    output_prefix: str
    module_dir: Path
    module_name: str
    worker_name: str
    worker_kind: str = "seed_worker"
    convert_zero_to_minus_one: bool = False
    import_error_attr: Optional[str] = None


METHOD_SPECS: Dict[str, MethodSpec] = {
    "py-irt": MethodSpec(
        key="py-irt",
        output_prefix="py-irt",
        module_dir=ROOT_DIR / "py-irt",
        module_name="PyIRT_logit",
        worker_name="pyirt_logit",
        import_error_attr="_PYIRT_IMPORT_ERROR",
    ),
    "cbmm": MethodSpec(
        key="cbmm",
        output_prefix="cbmm",
        module_dir=ROOT_DIR / "cBMM",
        module_name="cBMM_logit",
        worker_name="cBMM",
        worker_kind="cbmm_logit",
        convert_zero_to_minus_one=True,
    ),
}


def load_worker(spec: MethodSpec) -> Callable[..., Dict[str, np.ndarray]]:
    root_dir = str(ROOT_DIR)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    module_dir = str(spec.module_dir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module = importlib.import_module(spec.module_name)

    if spec.import_error_attr is not None:
        import_error = getattr(module, spec.import_error_attr, None)
        if import_error is not None:
            raise RuntimeError(
                "Cannot import py-irt dependencies. Install with: pip install py-irt pyro-ppl torch"
            ) from import_error

    return getattr(module, spec.worker_name)


def make_init_method1(m: int, n: int, r_true: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    u0 = rng.normal(size=(m, r_true))
    u0[:, r_true - 1] = 1.0
    v0 = rng.normal(size=(n, r_true))
    if r_true > 1:
        v0[:, : (r_true - 1)] = np.exp(v0[:, : (r_true - 1)])
    return {
        "init_U": u0.copy(),
        "init_V": v0.copy(),
        "init_U1": u0[:, : (r_true - 1)].copy(),
        "init_V1": v0[:, : (r_true - 1)].copy(),
        "init_v2": v0[:, r_true - 1].copy(),
    }


def _empty_seed_result(m: int, n: int, n_dim: int) -> Dict[str, Any]:
    return {
        "runtime": np.nan,
        "theta": np.full((m, n_dim), np.nan, dtype=float),
        "a": np.full((n, n_dim), np.nan, dtype=float),
        "b": np.full(n, np.nan, dtype=float),
        "m_recovered": np.full((m, n), np.nan, dtype=float),
        "final_loss": np.nan,
        "iterations": np.nan,
    }


def _valid_estimates(
    theta_hat: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    m: int,
    n: int,
    n_dim: int,
) -> bool:
    return (
        theta_hat.ndim == 2
        and a_hat.ndim == 2
        and b_hat.ndim == 1
        and theta_hat.shape == (m, n_dim)
        and a_hat.shape == (n, n_dim)
        and b_hat.shape[0] == n
        and np.all(np.isfinite(theta_hat))
        and np.all(np.isfinite(a_hat))
        and np.all(np.isfinite(b_hat))
    )


def finalize_seed_result(
    spec: MethodSpec,
    result: Dict[str, Any],
    m: int,
    n: int,
    r_true: int,
) -> Dict[str, Any]:
    out = dict(result)
    n_dim = r_true - 1
    if "m_recovered" not in out or out["m_recovered"] is None:
        th = out.get("theta")
        aa = out.get("a")
        bb = out.get("b")
        if (
            isinstance(th, np.ndarray)
            and isinstance(aa, np.ndarray)
            and isinstance(bb, np.ndarray)
            and _valid_estimates(th, aa, bb, m, n, n_dim)
        ):
            out["m_recovered"] = th @ aa.T + np.asarray(bb, dtype=float)[None, :]
        else:
            out["m_recovered"] = np.full((m, n), np.nan, dtype=float)
    if "final_loss" not in out:
        out["final_loss"] = np.nan
    if "iterations" not in out:
        out["iterations"] = np.nan
    return out


def cbmm_logit_one_seed_worker(
    cbmm_fn: Callable[..., Dict[str, np.ndarray]],
    seed_val: int,
    y: np.ndarray,
    m: int,
    n: int,
    r_true: int,
    sigma_val: float,
    max_iter_val: int,
    ind_omega_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n_dim = r_true - 1
    if n_dim <= 0:
        return _empty_seed_result(m, n, max(n_dim, 0))

    init_seed = seed_val + 10000
    ik = make_init_method1(m, n, r_true, init_seed)
    omega = ind_omega_train
    if omega is None:
        omega = np.ones(m * n, dtype=int)
    else:
        omega = np.asarray(omega, dtype=int).reshape(-1)
    try:
        t_fit = time.perf_counter()
        fit = cbmm_fn(
            y,
            r=n_dim,
            sigma=sigma_val,
            ind_omega=omega,
            init_U1=ik["init_U1"],
            init_V1=ik["init_V1"],
            init_v2=ik["init_v2"],
            tol=1e-4,
            max_iter=max_iter_val,
            verbose=0,
        )
        run_time = time.perf_counter() - t_fit
    except Exception:
        return _empty_seed_result(m, n, n_dim)

    theta_hat = np.asarray(fit.get("U1"), dtype=float)
    a_hat = np.asarray(fit.get("V1"), dtype=float)
    b_hat = np.asarray(fit.get("v2"), dtype=float)
    if not _valid_estimates(theta_hat, a_hat, b_hat, m, n, n_dim):
        return _empty_seed_result(m, n, n_dim)
    x_hat = fit.get("X_hat")
    if x_hat is not None:
        m_recovered = np.asarray(x_hat, dtype=float)
    else:
        m_recovered = theta_hat @ a_hat.T + b_hat[None, :]
    loss_hist = np.asarray(fit.get("loss_history", np.array([], dtype=float)), dtype=float)
    final_loss = float(loss_hist[-1]) if loss_hist.size > 0 else float("nan")
    return {
        "runtime": float(run_time),
        "theta": theta_hat,
        "a": a_hat,
        "b": b_hat,
        "m_recovered": m_recovered,
        "final_loss": final_loss,
        "iterations": float(fit.get("n_iter", np.nan)),
    }


def run_one_seed_entry(method_key: str, worker_kwargs: Dict[str, object]) -> Dict[str, Any]:
    spec = METHOD_SPECS[method_key]
    worker = load_worker(spec)
    m = int(worker_kwargs["m"])
    n = int(worker_kwargs["n"])
    r_true = int(worker_kwargs["r_true"])

    if spec.worker_kind == "cbmm_logit":
        raw = cbmm_logit_one_seed_worker(worker, **worker_kwargs)  # type: ignore[arg-type]
    else:
        raw = worker(**worker_kwargs)

    return finalize_seed_result(spec, raw, m, n, r_true)


def build_worker_kwargs(
    method_key: str,
    seed_val: int,
    m: int,
    n: int,
    r_true: int,
    sigma: float,
    max_iter: int,
    y_signed: np.ndarray,
    ind_flat: np.ndarray,
    y_pyirt: np.ndarray,
) -> Dict[str, object]:
    base: Dict[str, object] = {
        "seed_val": seed_val,
        "m": m,
        "n": n,
        "r_true": r_true,
        "sigma_val": sigma,
        "max_iter_val": max_iter,
    }
    if method_key == "py-irt":
        base["y"] = y_pyirt
        return base
    base["ind_omega_train"] = ind_flat
    base["y"] = y_signed
    return base


def append_metrics_snapshot(rows: List[Optional[pd.DataFrame]], out_file: Path) -> None:
    non_null = [r for r in rows if r is not None and not r.empty]
    if not non_null:
        return
    pd.concat(non_null, ignore_index=True).to_csv(out_file, index=False)


def run_method(
    spec: MethodSpec,
    sim_cfg: Dict[str, Any],
    miss_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
    seeds: List[int],
    n_cores: int,
) -> None:
    load_worker(spec)

    N = int(sim_cfg["N"])
    J = int(sim_cfg["J"])
    r_true = int(sim_cfg["r_true"])
    sigma = float(sim_cfg["sigma"])
    spiky = bool(sim_cfg["spiky"])
    a_zero_prop = float(sim_cfg["a_zero_prop"])

    val_ratio = float(miss_cfg["val_ratio"])
    missing_mechanism = str(miss_cfg["missing_mechanism"])
    mar_slope = float(miss_cfg["mar_slope"])
    mnar_other_scale = float(miss_cfg["mnar_other_scale"])
    mnar_col_prob = float(miss_cfg["mnar_col_prob"])

    max_iter = int(opt_cfg["max_iter"])

    metrics_rows: List[Optional[pd.DataFrame]] = [None]
    out_file_metrics = OUT_DIR / f"{spec.output_prefix}_metrics.csv"

    print(f"\n##### method: {spec.key} #####")
    t_outer = time.perf_counter()

    def process_seed(seed_val: int, idx: int) -> pd.DataFrame:
        dat = gen_data(
            N=N,
            J=J,
            r_true=r_true,
            seed=seed_val,
            sigma=sigma,
            spiky=spiky,
            a_zero_prop=a_zero_prop,
        )
        y0 = dat["Y0"]
        m_true = dat["M_true"]
        theta = dat["theta"]
        a = dat["a"]
        b = dat["b"]

        miss_seed = (seed_val + 30000) if seed_val is not None else None  # align val_py-irt
        missing_info = build_missing_partition(
            y0=y0,
            val_ratio=val_ratio,
            missing_mechanism=missing_mechanism,
            mar_slope=mar_slope,
            mnar_other_scale=mnar_other_scale,
            mnar_col_prob=mnar_col_prob,
            seed=miss_seed,
        )
        mask_train = missing_info["mask_train"]
        val_omega = missing_info["val_omega"]
        ind_flat = np.asarray(missing_info["ind_omega_train"], dtype=int).reshape(-1)

        y0_flat = np.ravel(y0, order="F")
        y_signed = 2.0 * y0 - 1.0
        y_signed[~mask_train] = 0.0

        y_pyirt = y0.copy()
        y_pyirt[~mask_train] = np.nan

        kwargs = build_worker_kwargs(
            method_key=spec.key,
            seed_val=seed_val,
            m=N,
            n=J,
            r_true=r_true,
            sigma=sigma,
            max_iter=max_iter,
            y_signed=y_signed,
            ind_flat=ind_flat,
            y_pyirt=y_pyirt,
        )
        seed_result = run_one_seed_entry(spec.key, kwargs)

        metrics = compute_metrics(
            theta_hat=np.asarray(seed_result["theta"], dtype=float),
            a_hat=np.asarray(seed_result["a"], dtype=float),
            b_hat=np.asarray(seed_result["b"], dtype=float),
            m_recovered=np.asarray(seed_result["m_recovered"], dtype=float),
            theta=theta,
            a=a,
            b=b,
            m_true=m_true,
            y0_flat=y0_flat,
            val_omega=val_omega,
            sigma=sigma,
            val_ratio=val_ratio,
            spiky=spiky,
            run_time_sec=float(seed_result["runtime"]),
            final_loss=float(seed_result["final_loss"]),
            iterations=float(seed_result["iterations"]),
            max_iter=max_iter,
        )

        row: Dict[str, Any] = {
            "method": spec.key,
            "seed": seed_val,
            "rep_idx": idx,
            "N": N,
            "J": J,
            "r_true": r_true,
            "sigma": sigma,
            "spiky": spiky,
            "a_zero_prop": a_zero_prop,
            "val_ratio_target": val_ratio,
            "missing_mechanism": missing_info["mechanism"],
            "mar_slope": mar_slope,
            "mnar_other_scale": mnar_other_scale,
            "mnar_col_prob": mnar_col_prob,
            "realized_missing_rate": missing_info["realized_missing_rate"],
            "realized_missing_rate_y1": missing_info["realized_missing_rate_y1"],
            "realized_missing_rate_y0": missing_info["realized_missing_rate_y0"],
            "max_iter": max_iter,
        }
        row.update(metrics)
        return pd.DataFrame([row])

    if n_cores > 1:
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            future_map = {
                pool.submit(
                    _metrics_worker_entry,
                    spec.key,
                    sim_cfg,
                    miss_cfg,
                    opt_cfg,
                    seed_val,
                    idx,
                ): idx
                for idx, seed_val in enumerate(seeds)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                df_one = future.result()
                metrics_rows[0] = (
                    df_one
                    if metrics_rows[0] is None or metrics_rows[0].empty
                    else pd.concat([metrics_rows[0], df_one], ignore_index=True)
                )
                append_metrics_snapshot(metrics_rows, out_file_metrics)
                print(f"  seed slot {idx + 1}/{len(seeds)} completed")
    else:
        for idx, seed_val in enumerate(seeds):
            df_one = process_seed(seed_val, idx)
            metrics_rows[0] = (
                df_one
                if metrics_rows[0] is None or metrics_rows[0].empty
                else pd.concat([metrics_rows[0], df_one], ignore_index=True)
            )
            append_metrics_snapshot(metrics_rows, out_file_metrics)
            print(
                f"  replicate {idx + 1}/{len(seeds)} completed "
                f"(seed={seed_val}, {df_one['run_time_sec'].iloc[0]:.4f}s)"
            )

    elapsed = time.perf_counter() - t_outer
    print(f"  elapsed: {elapsed:.4f}s | Saved to: {out_file_metrics}")


def _metrics_worker_entry(
    method_key: str,
    sim_cfg: Dict[str, Any],
    miss_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
    seed_val: int,
    idx: int,
) -> pd.DataFrame:
    """Picklable top-level entry for ProcessPoolExecutor (loads spec & rebuilds logic)."""
    spec = METHOD_SPECS[method_key]

    N = int(sim_cfg["N"])
    J = int(sim_cfg["J"])
    r_true = int(sim_cfg["r_true"])
    sigma = float(sim_cfg["sigma"])
    spiky = bool(sim_cfg["spiky"])
    a_zero_prop = float(sim_cfg["a_zero_prop"])

    val_ratio = float(miss_cfg["val_ratio"])
    missing_mechanism = str(miss_cfg["missing_mechanism"])
    mar_slope = float(miss_cfg["mar_slope"])
    mnar_other_scale = float(miss_cfg["mnar_other_scale"])
    mnar_col_prob = float(miss_cfg["mnar_col_prob"])

    max_iter = int(opt_cfg["max_iter"])

    dat = gen_data(
        N=N,
        J=J,
        r_true=r_true,
        seed=seed_val,
        sigma=sigma,
        spiky=spiky,
        a_zero_prop=a_zero_prop,
    )
    y0 = dat["Y0"]
    m_true = dat["M_true"]
    theta = dat["theta"]
    a = dat["a"]
    b = dat["b"]

    miss_seed = (seed_val + 30000) if seed_val is not None else None
    missing_info = build_missing_partition(
        y0=y0,
        val_ratio=val_ratio,
        missing_mechanism=missing_mechanism,
        mar_slope=mar_slope,
        mnar_other_scale=mnar_other_scale,
        mnar_col_prob=mnar_col_prob,
        seed=miss_seed,
    )
    mask_train = missing_info["mask_train"]
    val_omega = missing_info["val_omega"]
    ind_flat = np.asarray(missing_info["ind_omega_train"], dtype=int).reshape(-1)

    y0_flat = np.ravel(y0, order="F")
    y_signed = 2.0 * y0 - 1.0
    y_signed[~mask_train] = 0.0

    y_pyirt = y0.copy()
    y_pyirt[~mask_train] = np.nan

    kwargs = build_worker_kwargs(
        method_key=spec.key,
        seed_val=seed_val,
        m=N,
        n=J,
        r_true=r_true,
        sigma=sigma,
        max_iter=max_iter,
        y_signed=y_signed,
        ind_flat=ind_flat,
        y_pyirt=y_pyirt,
    )
    seed_result = run_one_seed_entry(spec.key, kwargs)

    metrics = compute_metrics(
        theta_hat=np.asarray(seed_result["theta"], dtype=float),
        a_hat=np.asarray(seed_result["a"], dtype=float),
        b_hat=np.asarray(seed_result["b"], dtype=float),
        m_recovered=np.asarray(seed_result["m_recovered"], dtype=float),
        theta=theta,
        a=a,
        b=b,
        m_true=m_true,
        y0_flat=y0_flat,
        val_omega=val_omega,
        sigma=sigma,
        val_ratio=val_ratio,
        spiky=spiky,
        run_time_sec=float(seed_result["runtime"]),
        final_loss=float(seed_result["final_loss"]),
        iterations=float(seed_result["iterations"]),
        max_iter=max_iter,
    )

    row: Dict[str, Any] = {
        "method": spec.key,
        "seed": seed_val,
        "rep_idx": idx,
        "N": N,
        "J": J,
        "r_true": r_true,
        "sigma": sigma,
        "spiky": spiky,
        "a_zero_prop": a_zero_prop,
        "val_ratio_target": val_ratio,
        "missing_mechanism": missing_info["mechanism"],
        "mar_slope": mar_slope,
        "mnar_other_scale": mnar_other_scale,
        "mnar_col_prob": mnar_col_prob,
        "realized_missing_rate": missing_info["realized_missing_rate"],
        "realized_missing_rate_y1": missing_info["realized_missing_rate_y1"],
        "realized_missing_rate_y0": missing_info["realized_missing_rate_y0"],
        "max_iter": max_iter,
    }
    row.update(metrics)
    return pd.DataFrame([row])


def load_config(config_file: Path) -> Dict[str, object]:
    if not config_file.exists():
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _config_value(
    config: Dict[str, object],
    cli_value: object,
    key: str,
    default: object,
) -> object:
    return cli_value if cli_value is not None else config.get(key, default)


def _nested_cfg(config: Dict[str, object], *path: str, default: Any = None) -> Any:
    cur: Any = config
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def resolve_sim_config(config: Dict[str, object]) -> Dict[str, Any]:
    def pick(key: str, nested_path: List[str], default: Any) -> Any:
        v = _config_value(config, None, key, None)
        if v is not None:
            return v
        n = _nested_cfg(config, *nested_path, default=None)
        return default if n is None else n

    r_true_v = pick("r_true", ["simulation", "r_true"], None)
    if r_true_v is None:
        r_true_v = pick("r", ["simulation", "r"], None)
    if r_true_v is None:
        r_true_v = config.get("r", 2)

    return {
        "N": int(pick("N", ["simulation", "N"], 100)),
        "J": int(pick("J", ["simulation", "J"], 50)),
        "r_true": int(r_true_v),
        "sigma": float(pick("sigma", ["simulation", "sigma"], 1.0)),
        "spiky": bool(pick("spiky", ["simulation", "spiky"], False)),
        "a_zero_prop": float(pick("a_zero_prop", ["simulation", "a_zero_prop"], 0.0)),
    }


def resolve_miss_config(config: Dict[str, object]) -> Dict[str, Any]:
    def pick(key: str, nested_path: List[str], default: Any) -> Any:
        v = _config_value(config, None, key, None)
        if v is not None:
            return v
        n = _nested_cfg(config, *nested_path, default=None)
        return default if n is None else n

    return {
        "val_ratio": float(pick("val_ratio", ["missingness", "val_ratio"], 0.2)),
        "missing_mechanism": str(pick("missing_mechanism", ["missingness", "missing_mechanism"], "MCAR")),
        "mar_slope": float(pick("mar_slope", ["missingness", "mar_slope"], 2.0)),
        "mnar_other_scale": float(pick("mnar_other_scale", ["missingness", "mnar_other_scale"], 0.0)),
        "mnar_col_prob": float(pick("mnar_col_prob", ["missingness", "mnar_col_prob"], 1.0)),
    }


def resolve_opt_config(config: Dict[str, object], cli_max_iter: Optional[int]) -> Dict[str, Any]:
    v = cli_max_iter
    if v is None:
        v = _nested_cfg(config, "optimizer", "max_iter", default=None)
    if v is None:
        v = config.get("max_iter", 1000)
    return {"max_iter": int(v)}


def resolve_method_keys(config: Dict[str, object], cli_method: Optional[str]) -> List[str]:
    methods_value: object
    if cli_method is not None:
        methods_value = cli_method
    elif "methods" in config:
        methods_value = config["methods"]
    else:
        methods_value = config.get("method", "all")

    if methods_value == "all":
        return list(METHOD_SPECS)
    if isinstance(methods_value, str):
        method_keys = [methods_value]
    elif isinstance(methods_value, list):
        method_keys = [str(method_key) for method_key in methods_value]
    else:
        raise ValueError("Config value 'methods' must be 'all', a string, or a list of method names.")

    unknown = [method_key for method_key in method_keys if method_key not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available: {list(METHOD_SPECS)}")
    return method_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated benchmarks: utils.gen_data + missingness, metrics via utils.compute_metrics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Path to JSON runner config.",
    )
    parser.add_argument(
        "--method",
        choices=[*METHOD_SPECS.keys(), "all"],
        default=None,
        help="Optimizer method to run. Overrides config methods.",
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-rep", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--n-cores", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    sim_cfg = resolve_sim_config(config)
    miss_cfg = resolve_miss_config(config)
    opt_cfg = resolve_opt_config(config, args.max_iter)

    n_rep = int(_config_value(config, args.n_rep, "n_rep", 10))
    base_seed = int(_config_value(config, args.base_seed, "base_seed", 1))
    n_cores = int(_config_value(config, args.n_cores, "n_cores", 1))

    rep_cfg = _nested_cfg(config, "replicates", default=None)
    if isinstance(rep_cfg, dict):
        if args.n_rep is None and "n_rep" in rep_cfg:
            n_rep = int(rep_cfg["n_rep"])
        if args.base_seed is None and "base_seed" in rep_cfg:
            base_seed = int(rep_cfg["base_seed"])

    seeds = [base_seed + i for i in range(n_rep)]
    method_keys = resolve_method_keys(config, args.method)
    for method_key in method_keys:
        run_method(
            spec=METHOD_SPECS[method_key],
            sim_cfg=sim_cfg,
            miss_cfg=miss_cfg,
            opt_cfg=opt_cfg,
            seeds=seeds,
            n_cores=n_cores,
        )


if __name__ == "__main__":
    main()
