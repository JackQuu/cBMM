# Utility functions for mirt-based validation runs.
# Mirrors optimizer/utils.py and is sourced from runner_mirt.R.

# -------------------------- Data generation --------------------------
gen_data <- function(N, J, r, seed = NULL, sigma = 1, a_zero_prop = 0,
                     spiky = FALSE) {
  if (!is.null(seed)) set.seed(seed)
  a <- matrix(runif(J * (r - 1), 0.5, 1), nrow = J)
  n_a <- J * (r - 1L)
  pz <- min(1, max(0, as.numeric(a_zero_prop)))
  n_zero <- min(n_a, max(0L, as.integer(floor(pz * n_a))))
  if (n_zero > 0L) {
    zero_pos <- sample.int(n_a, size = n_zero, replace = FALSE)
    a[zero_pos] <- 0
  }
  if (spiky) {
    theta <- matrix(rt(N * (r - 1), df = 1), nrow = N)
  } else {
    theta <- matrix(rnorm(N * (r - 1), 0, 1), nrow = N)
  }
  b <- rnorm(J, 0, 0.5)
  linear_arg <- theta %*% t(a) + matrix(b, N, J, byrow = TRUE)
  M_true <- linear_arg
  P <- plogis(linear_arg / sigma)
  Y0 <- matrix(rbinom(N * J, 1, P), N, J)
  list(Y0 = Y0, M_true = M_true, theta = theta, a = a, b = b)
}

# -------------------------- Helpers --------------------------
hellinger_distance <- function(P, Q, eps = 1e-12) {
  p <- pmax(pmin(as.vector(P), 1 - eps), eps)
  q <- pmax(pmin(as.vector(Q), 1 - eps), eps)
  sqrt(mean((sqrt(p) - sqrt(q))^2))
}

rank_corr_mat <- function(hat, true) {
  if (!is.matrix(hat) || !is.matrix(true)) return(NA_real_)
  if (!identical(dim(hat), dim(true))) return(NA_real_)
  xh <- as.vector(hat)
  xt <- as.vector(true)
  if (length(xh) < 2L) return(NA_real_)
  rho <- tryCatch(
    suppressWarnings(stats::cor(xt, xh, method = "spearman", use = "pairwise.complete.obs")),
    error = function(e) NA_real_
  )
  if (length(rho) != 1L || !is.finite(rho)) NA_real_ else as.numeric(rho)
}

# -------------------------- Metrics --------------------------
# class_error_val: mismatch rate between sign(M) and Y0 on validation cells
compute_metrics <- function(theta_hat, a_hat, b_hat, M_recovered, theta, a, b, M_true, Y0_flat,
                            val_omega, sigma, val_ratio, spiky,
                            run_time_sec, final_loss, iterations, max_iter) {
  ok_theta_hat <- is.matrix(theta_hat) && all(is.finite(theta_hat))
  ok_a_hat <- is.matrix(a_hat) && all(is.finite(a_hat))
  ok_b_hat <- all(is.finite(b_hat))
  ok_M <- is.matrix(M_recovered) && all(is.finite(M_recovered))
  ok_truth <- all(is.finite(theta)) && all(is.finite(a)) && all(is.finite(b)) && all(is.finite(M_true))

  rmse_theta <- if (ok_theta_hat) sqrt(mean((theta_hat - theta)^2)) else NA_real_
  rmse_a <- if (ok_a_hat) sqrt(mean((a_hat - a)^2)) else NA_real_

  theta_rank_corr <- rank_corr_mat(theta_hat, theta)
  a_rank_corr <- rank_corr_mat(a_hat, a)
  b_rank_corr <- rank_corr_mat(as.matrix(b_hat), as.matrix(b))

  rmse_b <- if (ok_b_hat) sqrt(mean((b_hat - b)^2)) else NA_real_
  fnorm_diff <- if (ok_M && ok_truth) norm(M_recovered - M_true, "F") else NA_real_
  fnorm_true <- if (ok_truth) norm(M_true, "F") else NA_real_
  m_rel_fnorm <- if (is.finite(fnorm_true) && fnorm_true > 0 && is.finite(fnorm_diff)) {
    fnorm_diff / fnorm_true
  } else {
    NA_real_
  }

  hellinger <- if (ok_M && ok_truth) {
    P_recovered <- plogis(M_recovered / sigma)
    P_true <- plogis(M_true / sigma)
    hellinger_distance(P_recovered, P_true)
  } else {
    NA_real_
  }

  class_error_val <- if (ok_M && length(val_omega) > 0L) {
    M_flat <- as.vector(M_recovered)
    m_val <- M_flat[val_omega]
    y0_val <- Y0_flat[val_omega]
    yhat_binary <- as.numeric(m_val >= 0)
    mean(y0_val != yhat_binary)
  } else {
    NA_real_
  }

  converged <- if (is.na(iterations)) FALSE else (iterations < max_iter)
  list(
    rmse_theta = rmse_theta,
    rmse_a = rmse_a,
    theta_rank_corr = theta_rank_corr,
    a_rank_corr = a_rank_corr,
    rmse_b = rmse_b,
    b_rank_corr = b_rank_corr,
    m_rel_fnorm = m_rel_fnorm,
    hellinger_distance = hellinger,
    class_error_val = class_error_val,
    missing_rate = val_ratio,
    run_time_sec = run_time_sec,
    final_loss = final_loss,
    iterations = iterations,
    converged = converged,
    spiky = spiky
  )
}

# -------------------------- mirt-specific helpers --------------------------
mirt_pars_fill_from_V0 <- function(sv, Y0_df, V0, r_true, J) {
  n_dim <- r_true - 1L
  item_names <- colnames(Y0_df)
  if (is.null(item_names)) item_names <- names(Y0_df)
  if (is.null(item_names)) item_names <- sprintf("V%d", seq_len(J))
  for (j in seq_len(J)) {
    inm <- item_names[j]
    for (k in seq_len(n_dim)) {
      w <- which(as.character(sv$item) == inm & sv$name == paste0("a", k))
      if (length(w) == 1L) sv$value[w] <- V0[j, k]
    }
    w <- which(as.character(sv$item) == inm & sv$name == "d")
    if (length(w) == 1L) sv$value[w] <- V0[j, r_true]
  }
  sv
}

mirt_na_metrics <- function(val_ratio, spiky, run_time_sec) {
  list(
    rmse_theta = NA_real_, rmse_a = NA_real_,
    theta_rank_corr = NA_real_, a_rank_corr = NA_real_,
    rmse_b = NA_real_, b_rank_corr = NA_real_, m_rel_fnorm = NA_real_, hellinger_distance = NA_real_,
    class_error_val = NA_real_,
    missing_rate = val_ratio, run_time_sec = run_time_sec,
    final_loss = NA_real_, iterations = NA_integer_, converged = FALSE, spiky = spiky
  )
}

# -------------------------- Missing-data partitioning --------------------------
normalize_missing_mechanism <- function(missing_mechanism) {
  mech <- toupper(as.character(missing_mechanism)[1L])
  if (!mech %in% c("MCAR", "MAR", "MNAR_0")) {
    stop("missing_mechanism must be one of 'MCAR', 'MAR', or 'MNAR_0'", call. = FALSE)
  }
  mech
}

build_missing_partition <- function(Y0, val_ratio, missing_mechanism = "MCAR",
                                    mar_slope = 2, mnar_other_scale = 0,
                                    mnar_col_prob = 1) {
  # mar_slope controls how strongly row/column position affects missingness under MAR.
  # Larger values create a steeper gradient (higher-row/higher-column cells become
  # much more likely to be missing); values near 0 make MAR close to uniform missingness.
  #
  # Under MNAR_0, entries with Y=0 use missing probability val_ratio;
  # entries with Y!=0 use val_ratio * mnar_other_scale (clamped to [0, 1]).
  # Missingness only occurs in selected columns, where each column is selected
  # by a Bernoulli draw with probability mnar_col_prob.
  if (!is.finite(val_ratio) || val_ratio < 0 || val_ratio > 1) {
    stop("val_ratio must be within [0, 1]", call. = FALSE)
  }
  if (!is.finite(mnar_col_prob) || mnar_col_prob < 0 || mnar_col_prob > 1) {
    stop("mnar_col_prob must be within [0, 1]", call. = FALSE)
  }
  mech <- normalize_missing_mechanism(missing_mechanism)
  N <- nrow(Y0)
  J <- ncol(Y0)
  total <- N * J

  if (mech == "MCAR") {
    n_val <- round(val_ratio * total)
    all_indices <- sample(total)
    val_omega <- if (n_val > 0L) all_indices[seq_len(n_val)] else integer(0)
    mask_train <- matrix(TRUE, N, J)
    if (n_val > 0L) mask_train[val_omega] <- FALSE
    miss_prob <- rep(val_ratio, total)
  } else if (mech == "MAR") {
    row_pos <- if (N > 1L) (seq_len(N) - 1) / (N - 1) else rep(0, N)
    col_pos <- if (J > 1L) (seq_len(J) - 1) / (J - 1) else rep(0, J)
    score_mat <- (outer(row_pos, rep(1, J)) + outer(rep(1, N), col_pos)) / 2
    centered <- as.vector(score_mat - mean(score_mat))
    if (val_ratio <= 0) {
      miss_prob <- rep(0, total)
    } else if (val_ratio >= 1) {
      miss_prob <- rep(1, total)
    } else {
      target_fn <- function(alpha) {
        mean(plogis(alpha + mar_slope * centered)) - val_ratio
      }
      alpha <- uniroot(target_fn, interval = c(-40, 40), tol = 1e-10)$root
      miss_prob <- plogis(alpha + mar_slope * centered)
    }
    miss_draw <- (runif(total) < miss_prob)
    val_omega <- which(miss_draw)
    mask_train <- matrix(!miss_draw, N, J)
  } else if (mech == "MNAR_0") {
    y_vec <- as.vector(Y0)
    zero_prob <- rep(val_ratio, total)
    other_prob <- pmin(pmax(val_ratio * mnar_other_scale, 0), 1)
    base_prob <- ifelse(y_vec == 0, zero_prob, other_prob)
    selected_cols <- as.integer(stats::rbinom(J, size = 1L, prob = mnar_col_prob))
    col_gate <- rep(selected_cols, each = N)
    miss_prob <- base_prob * col_gate
    miss_draw <- (runif(total) < miss_prob)
    val_omega <- which(miss_draw)
    mask_train <- matrix(!miss_draw, N, J)
  }

  ind_omega_train <- as.integer(as.vector(mask_train))
  y_vec <- as.vector(Y0)
  miss_vec <- !as.vector(mask_train)
  realized_rate <- if (total > 0L) mean(miss_vec) else NA_real_
  idx_y1 <- (y_vec == 1)
  idx_y0 <- (y_vec == 0)
  realized_rate_y1 <- if (any(idx_y1)) mean(miss_vec[idx_y1]) else NA_real_
  realized_rate_y0 <- if (any(idx_y0)) mean(miss_vec[idx_y0]) else NA_real_

  list(
    mechanism = mech,
    val_omega = val_omega,
    n_val = length(val_omega),
    mask_train = mask_train,
    ind_omega_train = ind_omega_train,
    realized_missing_rate = realized_rate,
    realized_missing_rate_y1 = realized_rate_y1,
    realized_missing_rate_y0 = realized_rate_y0
  )
}
