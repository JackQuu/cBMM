import numpy as np
from scipy.special import expit
from scipy.optimize._nnls import _nnls as _nnls_low  # bare backend
from scipy.linalg import cho_factor, cho_solve

# ----------------------------------------------------------------------
# Optional thread control for one solver call
# (small-r kernels can slow down under thread contention)
# ----------------------------------------------------------------------
import os as _os
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _have_tpc = True
except Exception:
    _have_tpc = False
    def _threadpool_limits(limits=None):
        # no-op fallback
        class _Null:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Null()


def _auto_num_threads():
    """Use 1 thread on <=4 cores; otherwise keep BLAS default."""
    try:
        n_cpu = _os.cpu_count() or 1
    except Exception:
        n_cpu = 1
    return 1 if n_cpu <= 4 else None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def neg_loglik(X, Y, mask_bool=None, sigma=1):
    """-sum log Phi(Y * X / sigma), on observed entries if mask is provided."""
    if mask_bool is None:
        t = -Y * (X / sigma)
    else:
        t = -Y[mask_bool] * (X[mask_bool] / sigma)
    abs_t = np.abs(t)
    return float(np.sum(np.maximum(t, 0.0) + np.log1p(np.exp(-abs_t))))


def surrogate_and_loss(Y, Xtilde, sigma=1, mask_bool=None):
    """Compute MM surrogate target and current loss in one pass."""
    t = -(Y * Xtilde) * (1.0 / sigma)        
    q = expit(t)                              
    Ytilde = Xtilde + (4.0 * sigma) * (Y * q)
    if mask_bool is None:
        abs_t = np.abs(t)
        loss = float(np.sum(np.maximum(t, 0.0) + np.log1p(np.exp(-abs_t))))
    else:
        tm = t[mask_bool]
        abs_t = np.abs(tm)
        loss = float(np.sum(np.maximum(tm, 0.0) + np.log1p(np.exp(-abs_t))))
    return Ytilde, loss


def surrogate_Y(Y, Xtilde, sigma=1):
    """MM surrogate target: Xtilde + 4*sigma*Y*Phi(-Y*Xtilde/sigma)."""
    z = (Y * Xtilde) * (1.0 / sigma)        
    return Xtilde + (4.0 * sigma) * Y * expit(-z)


def matrix_nnls(U, R, W=None):
    """
    Matrix NNLS: solve min_{Z >= 0} ||U Z - R||_F^2.
    Columns of Z are independent given U, so we solve NNLS column-wise.
    If mask W is provided, each column only uses observed rows.
    """
    n = R.shape[1]
    r = U.shape[1]
    Z = np.zeros((r, n))
    # ensure contiguous float64 inputs
    U = np.ascontiguousarray(U, dtype=np.float64)
    R = np.ascontiguousarray(R, dtype=np.float64)
    maxiter_full = 3 * r
    if W is None:
        for j in range(n):
            x, _, info = _nnls_low(U, R[:, j], maxiter_full)
            Z[:, j] = x
        return Z
    # masked columns
    for j in range(n):
        obs_rows = np.flatnonzero(W[:, j])
        k = obs_rows.size
        if k == 0:
            continue
        U_j = np.ascontiguousarray(U[obs_rows, :])
        R_j = np.ascontiguousarray(R[obs_rows, j])
        x, _, info = _nnls_low(U_j, R_j, 3 * r)
        Z[:, j] = x
    return Z


# ----------------------------------------------------------------------
# cBMM algorithm
# ----------------------------------------------------------------------

def cBMM(Y, r, sigma=1, ind_omega=None,
                   init_U1=None, init_V1=None, init_v2=None,
                   tol=1e-4, max_iter=1000, verbose=10,
                   num_threads='auto'):
    """
    cBMM solver for
        X = U1 @ V1.T + 1_m @ v2.T,   V1 >= 0.
    Y is binary with entries in {-1, +1}; ind_omega is optional mask.
    num_threads: 'auto' | int | None (thread control for this call only).
    """
    if num_threads == 'auto':
        num_threads = _auto_num_threads()
    with _threadpool_limits(limits=num_threads):
        return _cBMM_impl(
            Y, r, sigma, ind_omega, init_U1, init_V1, init_v2,
            tol, max_iter, verbose
        )


def _cBMM_impl(Y, r, sigma, ind_omega,
                         init_U1, init_V1, init_v2,
                         tol, max_iter, verbose):
    m, n = Y.shape
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    use_mask = ind_omega is not None

    if use_mask:
        ind_omega = np.asarray(ind_omega)
        assert len(ind_omega) == m * n, "len(ind_omega) must equal m * n"
        W = (ind_omega > 0).astype(np.float64).reshape(m, n, order='F')
        Wb = W.astype(bool)  # mask for indexing
        if not Wb.any():
            raise ValueError("ind_omega has no observed entries.")
        col_sums_W = W.sum(axis=0)
        col_sums_W_safe = np.where(col_sums_W < 1e-12, 1.0, col_sums_W)
        n_obs_per_row = W.sum(axis=1)
        ok_rows = n_obs_per_row >= r
        all_rows_ok = bool(ok_rows.all())
    else:
        W = None
        Wb = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    if init_U1 is not None and init_V1 is not None and init_v2 is not None:
        U1 = np.atleast_2d(np.asarray(init_U1, dtype=np.float64)).copy()
        V1 = np.atleast_2d(np.asarray(init_V1, dtype=np.float64)).copy()
        v2 = np.asarray(init_v2, dtype=np.float64).ravel().copy()
        assert U1.shape == (m, r)
        assert V1.shape == (n, r)
        assert v2.shape == (n,)
        if np.any(V1 < 0):
            raise ValueError("init_V1 must be non-negative.")
    else:
        Y_svd = Y.copy()
        if use_mask:
            Y_svd[~Wb] = 0
        U_s, d, Vt_s = np.linalg.svd(Y_svd, full_matrices=False)
        U_s  = U_s[:, :r]
        d_r  = d[:r]
        Vt_s = Vt_s[:r, :]
        s = np.sqrt(np.maximum(d_r, 0.0))
        U1 = U_s * s                          
        V1 = np.abs(Vt_s.T * s)
        v2 = np.zeros(n)

    # scale control
    M0 = U1 @ V1.T + v2[None, :]
    scale0 = np.nanmax(np.abs(M0))
    if scale0 > 10:
        scale0 = np.sqrt(scale0 / 10)
        U1 = U1 / scale0
        V1 = V1 / scale0
        v2 = v2 / scale0

    Xtilde       = U1 @ V1.T + v2[None, :]
    # initial surrogate
    Ytilde, loss_old = surrogate_and_loss(Y, Xtilde, sigma, Wb)
    if use_mask:
        Ytilde[~Wb] = 0.0
    loss_history = []

    for iter_num in range(1, max_iter + 1):
        # ------------------------------------------------------------------
        # Block 1: update V1 >= 0  (NNLS)
        #   min_{V1 >= 0} || U1 @ V1.T - R ||_F^2
        #   R = Ytilde - 1_m @ v2.T
        # ------------------------------------------------------------------
        R = Ytilde - v2[None, :]
        V1 = matrix_nnls(U1, R, W if use_mask else None).T   # n x r

        # ------------------------------------------------------------------
        # Block 2: update v2
        #   v2_j = average_i (Ytilde_ij - (U1 @ V1.T)_ij)
        # ------------------------------------------------------------------
        UV = U1 @ V1.T
        R = Ytilde - UV
        if use_mask:
            v2 = (R * W).sum(axis=0) / col_sums_W_safe
        else:
            v2 = R.sum(axis=0) / m

        # ------------------------------------------------------------------
        # Block 3: update U1
        # ------------------------------------------------------------------
        R = Ytilde - v2[None, :]
        if use_mask:
            VVt = np.einsum('jk,jl->jkl', V1, V1)  # (n, r, r)
            A = np.einsum('ij,jkl->ikl', W, VVt)  # (m, r, r)
            b = (R * W) @ V1  # (m, r)
            if all_rows_ok:
                U1 = np.linalg.solve(A, b[..., None])[..., 0]
            else:
                U1_new = U1.copy()
                if ok_rows.any():
                    U1_new[ok_rows] = np.linalg.solve(A[ok_rows], b[ok_rows][..., None])[..., 0]
                U1 = U1_new
        else:
            VtV = V1.T @ V1
            RV1 = R @ V1
            # Cholesky on small r x r SPD Gram matrix
            try:
                c, low = cho_factor(VtV, lower=True, overwrite_a=False, check_finite=False)
                U1 = cho_solve((c, low), RV1.T, overwrite_b=False, check_finite=False).T
            except np.linalg.LinAlgError:
                # fallbacks for near-singular Gram matrix
                try:
                    U1 = np.linalg.solve(VtV, RV1.T).T
                except np.linalg.LinAlgError:
                    # fallbacks for minimum-norm solution
                    U1 = np.linalg.lstsq(VtV, RV1.T, rcond=None)[0].T

        # update Xtilde and recompute surrogate
        Xtilde = U1 @ V1.T + v2[None, :]
        Ytilde, loss_new = surrogate_and_loss(Y, Xtilde, sigma, Wb)
        if use_mask:
            Ytilde[~Wb] = 0.0
        loss_history.append(loss_new)

        rel = abs(loss_new - loss_old) / (abs(loss_old) + 1e-12)

        if verbose > 0 and (iter_num == 1 or iter_num % verbose == 0):
            print(f"iter {iter_num:4d} | loss = {loss_new:12.4f} | rel = {rel:.2e}")

        if rel < tol:
            if verbose > 0:
                print(f"Converged at iter {iter_num}")
            break
        loss_old = loss_new

    return {
        'U1':           U1,
        'V1':           V1,
        'v2':           v2,
        'X_hat':        Xtilde,
        'loss_history': np.array(loss_history),
        'n_iter':       len(loss_history),
        'ind_omega':    ind_omega,
    }
