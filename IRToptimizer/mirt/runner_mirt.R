# Validation runner for the mirt optimizer.
# Output filename, schema, and path mirror optimizer/runner.py:
#   - Results saved to <optimizer>/Results/mirt_metrics.csv
#   - CSV columns identical to the Python runner's per-method output.
# Configuration is read from <optimizer>/runner_config.json (same file as runner.py).

rm(list = ls())
library(parallel)
library(mirt)
library(jsonlite)

# -------------------------- Source utility helpers --------------------------
.this_script_dir <- local({
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- cmd_args[grep("^--file=", cmd_args)]
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1L]), mustWork = FALSE)))
  }
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) {
    return(dirname(normalizePath(ofile, mustWork = FALSE)))
  }
  getwd()
})
source(file.path(.this_script_dir, "utils_mirt.R"))

# -------------------------- Constants matching runner.py --------------------------
ROOT_DIR <- normalizePath(file.path(.this_script_dir, ".."), mustWork = FALSE)
OUT_DIR <- file.path(ROOT_DIR, "Results")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
DEFAULT_CONFIG_FILE <- file.path(ROOT_DIR, "runner_config.json")

METHOD_KEY <- "mirt"
OUTPUT_PREFIX <- "mirt"

# -------------------------- mirt fit for a single replicate --------------------------
mirt_fit_one_seed <- function(seed_val, sim_cfg, miss_cfg, opt_cfg) {
  N <- as.integer(sim_cfg$N)
  J <- as.integer(sim_cfg$J)
  r_true <- as.integer(sim_cfg$r_true)
  sigma <- as.numeric(sim_cfg$sigma)
  spiky <- as.logical(sim_cfg$spiky)
  a_zero_prop <- as.numeric(sim_cfg$a_zero_prop)

  val_ratio <- as.numeric(miss_cfg$val_ratio)
  missing_mechanism <- normalize_missing_mechanism(miss_cfg$missing_mechanism)
  mar_slope <- as.numeric(miss_cfg$mar_slope)
  mnar_other_scale <- as.numeric(miss_cfg$mnar_other_scale)
  mnar_col_prob <- as.numeric(miss_cfg$mnar_col_prob)

  max_iter <- as.integer(opt_cfg$max_iter)

  dat <- gen_data(N = N, J = J, r = r_true, seed = seed_val,
                  sigma = sigma, a_zero_prop = a_zero_prop, spiky = spiky)
  Y0 <- dat$Y0
  M_true <- dat$M_true
  theta <- dat$theta
  a <- dat$a
  b <- dat$b

  # Match runner.py: missingness drawn with a separate offset seed (seed + 30000)
  if (!is.null(seed_val)) set.seed(seed_val + 30000L)
  missing_info <- build_missing_partition(
    Y0, val_ratio,
    missing_mechanism = missing_mechanism,
    mar_slope = mar_slope,
    mnar_other_scale = mnar_other_scale,
    mnar_col_prob = mnar_col_prob
  )
  n_val <- missing_info$n_val
  val_omega <- missing_info$val_omega
  mask_train <- missing_info$mask_train
  Y0_flat <- as.vector(Y0)

  if (!is.null(seed_val)) set.seed(seed_val + 10000L)
  V0 <- matrix(rnorm(J * r_true, 0, 1), J, r_true)
  V0[, seq_len(r_true - 1)] <- exp(V0[, seq_len(r_true - 1)])

  K <- r_true - 1L
  na_theta <- matrix(NA_real_, N, K)
  na_a <- matrix(NA_real_, J, K)
  na_b <- rep(NA_real_, J)
  na_M <- matrix(NA_real_, N, J)
  mirt_t0 <- Sys.time()

  bundle <- tryCatch(
    {
      Y_mirt <- Y0
      if (n_val > 0L) Y_mirt[!mask_train] <- NA
      n_dim <- r_true - 1L
      mirt_model_spec <- if (n_dim == 1L) {
        1L
      } else {
        mirt.model(paste(sapply(seq_len(n_dim), function(d) {
          sprintf("F%d = 1-%d", d, J)
        }), collapse = "\n"))
      }
      Y0_df <- as.data.frame(Y_mirt)
      if (!is.null(seed_val)) set.seed(seed_val + 20000L)
      sv_mirt <- tryCatch(
        mirt(data = Y0_df, model = mirt_model_spec, itemtype = "2PL",
             pars = "values", verbose = FALSE, SE = FALSE),
        error = function(e) {
          message("mirt pars=values error: ", conditionMessage(e))
          NULL
        }
      )
      if (!is.null(sv_mirt)) {
        sv_mirt <- mirt_pars_fill_from_V0(sv_mirt, Y0_df, V0, r_true, J)
        if ("lbound" %in% names(sv_mirt)) {
          w_a <- sv_mirt$name %in% paste0("a", seq_len(n_dim))
          sv_mirt$lbound[w_a] <- 0
        }
      }

      t_fit <- Sys.time()
      res_mirt <- tryCatch(
        mirt(data = Y0_df, model = mirt_model_spec, itemtype = "2PL",
             verbose = FALSE, SE = FALSE, pars = sv_mirt,
             technical = list(NCYCLES = max_iter)),
        error = function(e) {
          message("mirt error: ", conditionMessage(e))
          NULL
        }
      )
      run_time <- as.numeric(difftime(Sys.time(), t_fit, units = "secs"))

      if (!is.null(res_mirt)) {
        n_iter <- tryCatch(res_mirt@OptimInfo$iter, error = function(e) NA_integer_)
        ll_val <- tryCatch(as.numeric(logLik(res_mirt)), error = function(e) NULL)
        final_loss <- if (is.null(ll_val) || !is.finite(ll_val)) NA_real_ else -ll_val
        theta_hat <- tryCatch(
          as.matrix(fscores(res_mirt, method = "EAP", full.scores = TRUE, full.scores.SE = FALSE)),
          error = function(e) NULL
        )
        item_params <- tryCatch(coef(res_mirt, simplify = TRUE)$items, error = function(e) NULL)
        a_cols <- paste0("a", seq_len(n_dim))
        cn <- if (!is.null(item_params)) colnames(item_params) else NULL
        extract_ok <- is.matrix(theta_hat) && nrow(theta_hat) == N &&
          !is.null(item_params) && (is.data.frame(item_params) || is.matrix(item_params)) &&
          !is.null(cn) && all(a_cols %in% cn) && ("d" %in% cn)
        if (isTRUE(extract_ok)) {
          a_hat <- as.matrix(item_params[, a_cols, drop = FALSE])
          b_hat <- as.numeric(item_params[, "d"])
          extract_ok <- nrow(a_hat) == J && all(is.finite(b_hat))
        }
        if (isTRUE(extract_ok) && all(is.finite(theta_hat))) {
          M_recovered <- theta_hat %*% t(a_hat) + matrix(b_hat, N, J, byrow = TRUE)
          list(theta_hat = theta_hat, a_hat = a_hat, b_hat = b_hat,
               M_recovered = M_recovered, run_time = run_time,
               final_loss = final_loss, iterations = n_iter)
        } else {
          list(theta_hat = na_theta, a_hat = na_a, b_hat = na_b,
               M_recovered = na_M, run_time = run_time,
               final_loss = final_loss, iterations = n_iter)
        }
      } else {
        list(theta_hat = na_theta, a_hat = na_a, b_hat = na_b,
             M_recovered = na_M, run_time = run_time,
             final_loss = NA_real_, iterations = NA_integer_)
      }
    },
    error = function(e) {
      message("mirt block error (NA recorded, continuing): ", conditionMessage(e))
      list(theta_hat = na_theta, a_hat = na_a, b_hat = na_b,
           M_recovered = na_M,
           run_time = as.numeric(difftime(Sys.time(), mirt_t0, units = "secs")),
           final_loss = NA_real_, iterations = NA_integer_)
    }
  )

  metrics <- compute_metrics(
    theta_hat = bundle$theta_hat, a_hat = bundle$a_hat, b_hat = bundle$b_hat,
    M_recovered = bundle$M_recovered, theta = theta, a = a, b = b,
    M_true = M_true, Y0_flat = Y0_flat, val_omega = val_omega, sigma = sigma,
    val_ratio = val_ratio, spiky = spiky,
    run_time_sec = bundle$run_time, final_loss = bundle$final_loss,
    iterations = bundle$iterations, max_iter = max_iter
  )

  list(metrics = metrics, missing_info = missing_info)
}

# -------------------------- Single CSV row matching runner.py output --------------------------
build_metrics_row <- function(seed_val, rep_idx, sim_cfg, miss_cfg, opt_cfg, fit_result) {
  metrics <- fit_result$metrics
  miss <- fit_result$missing_info
  data.frame(
    method = METHOD_KEY,
    seed = as.integer(seed_val),
    rep_idx = as.integer(rep_idx),
    N = as.integer(sim_cfg$N),
    J = as.integer(sim_cfg$J),
    r_true = as.integer(sim_cfg$r_true),
    sigma = as.numeric(sim_cfg$sigma),
    spiky = as.logical(metrics$spiky),
    a_zero_prop = as.numeric(sim_cfg$a_zero_prop),
    val_ratio_target = as.numeric(miss_cfg$val_ratio),
    missing_mechanism = miss$mechanism,
    mar_slope = as.numeric(miss_cfg$mar_slope),
    mnar_other_scale = as.numeric(miss_cfg$mnar_other_scale),
    mnar_col_prob = as.numeric(miss_cfg$mnar_col_prob),
    realized_missing_rate = miss$realized_missing_rate,
    realized_missing_rate_y1 = miss$realized_missing_rate_y1,
    realized_missing_rate_y0 = miss$realized_missing_rate_y0,
    max_iter = as.integer(opt_cfg$max_iter),
    rmse_theta = metrics$rmse_theta,
    rmse_a = metrics$rmse_a,
    theta_rank_corr = metrics$theta_rank_corr,
    a_rank_corr = metrics$a_rank_corr,
    rmse_b = metrics$rmse_b,
    b_rank_corr = metrics$b_rank_corr,
    m_rel_fnorm = metrics$m_rel_fnorm,
    hellinger_distance = metrics$hellinger_distance,
    class_error_val = metrics$class_error_val,
    missing_rate = metrics$missing_rate,
    run_time_sec = metrics$run_time_sec,
    final_loss = metrics$final_loss,
    iterations = metrics$iterations,
    converged = metrics$converged,
    stringsAsFactors = FALSE
  )
}

append_metrics_snapshot <- function(rows_list, out_file) {
  non_null <- Filter(Negate(is.null), rows_list)
  if (length(non_null) == 0L) return(invisible(NULL))
  out_df <- do.call(rbind, non_null)
  if (nrow(out_df) == 0L) return(invisible(NULL))
  write.csv(out_df, out_file, row.names = FALSE)
  invisible(out_df)
}

# -------------------------- Run all replicates for the mirt method --------------------------
run_method <- function(sim_cfg, miss_cfg, opt_cfg, seeds, n_cores) {
  out_file <- file.path(OUT_DIR, sprintf("%s_metrics.csv", OUTPUT_PREFIX))
  cat(sprintf("\n##### method: %s #####\n", METHOD_KEY))
  t_outer <- Sys.time()
  n_tot <- length(seeds)
  rows_list <- vector("list", n_tot)

  process_one <- function(seed_val, idx) {
    fit <- mirt_fit_one_seed(seed_val, sim_cfg, miss_cfg, opt_cfg)
    build_metrics_row(seed_val, idx - 1L, sim_cfg, miss_cfg, opt_cfg, fit)
  }

  if (n_cores > 1L) {
    cl <- makeCluster(n_cores, outfile = "")
    on.exit(stopCluster(cl), add = TRUE)
    wd <- getwd()
    utils_path <- file.path(.this_script_dir, "utils_mirt.R")
    clusterExport(cl, c(
      "wd", "utils_path", "sim_cfg", "miss_cfg", "opt_cfg", "seeds",
      "mirt_fit_one_seed", "build_metrics_row",
      "METHOD_KEY", "OUTPUT_PREFIX"
    ), envir = environment())
    clusterEvalQ(cl, {
      setwd(wd)
      library(mirt)
      source(utils_path)
    })
    idx_batches <- split(seq_len(n_tot), ceiling(seq_len(n_tot) / n_cores))
    for (idxs in idx_batches) {
      part <- parLapply(cl, idxs, function(i) {
        fit <- mirt_fit_one_seed(seeds[i], sim_cfg, miss_cfg, opt_cfg)
        build_metrics_row(seeds[i], i - 1L, sim_cfg, miss_cfg, opt_cfg, fit)
      })
      for (k in seq_along(idxs)) {
        rows_list[[idxs[k]]] <- part[[k]]
      }
      append_metrics_snapshot(rows_list, out_file)
      for (k in seq_along(idxs)) {
        cat(sprintf("  seed slot %d/%d completed\n", idxs[k], n_tot))
      }
      flush.console()
    }
  } else {
    for (idx in seq_len(n_tot)) {
      seed_val <- seeds[idx]
      row <- process_one(seed_val, idx)
      rows_list[[idx]] <- row
      append_metrics_snapshot(rows_list, out_file)
      cat(sprintf("  replicate %d/%d completed (seed=%d, %.4fs)\n",
                  idx, n_tot, seed_val,
                  ifelse(is.finite(row$run_time_sec), row$run_time_sec, NA_real_)))
      flush.console()
    }
  }

  elapsed <- as.numeric(difftime(Sys.time(), t_outer, units = "secs"))
  cat(sprintf("  elapsed: %.4fs | Saved to: %s\n", elapsed, out_file))
  invisible(out_file)
}

# -------------------------- Configuration loaders mirroring runner.py --------------------------
load_config <- function(config_file) {
  if (!file.exists(config_file)) return(list())
  jsonlite::fromJSON(config_file, simplifyVector = TRUE)
}

resolve_sim_config <- function(config) {
  sim <- if (is.list(config)) config$simulation else NULL
  pick <- function(key, default) {
    v <- if (is.list(config)) config[[key]] else NULL
    if (!is.null(v)) return(v)
    if (is.list(sim) && !is.null(sim[[key]])) return(sim[[key]])
    default
  }
  r_true_v <- pick("r_true", NULL)
  if (is.null(r_true_v)) r_true_v <- pick("r", NULL)
  if (is.null(r_true_v)) r_true_v <- 2L
  list(
    N = as.integer(pick("N", 100L)),
    J = as.integer(pick("J", 50L)),
    r_true = as.integer(r_true_v),
    sigma = as.numeric(pick("sigma", 1.0)),
    spiky = as.logical(pick("spiky", FALSE)),
    a_zero_prop = as.numeric(pick("a_zero_prop", 0.0))
  )
}

resolve_miss_config <- function(config) {
  miss <- if (is.list(config)) config$missingness else NULL
  pick <- function(key, default) {
    v <- if (is.list(config)) config[[key]] else NULL
    if (!is.null(v)) return(v)
    if (is.list(miss) && !is.null(miss[[key]])) return(miss[[key]])
    default
  }
  list(
    val_ratio = as.numeric(pick("val_ratio", 0.2)),
    missing_mechanism = as.character(pick("missing_mechanism", "MCAR")),
    mar_slope = as.numeric(pick("mar_slope", 2.0)),
    mnar_other_scale = as.numeric(pick("mnar_other_scale", 0.0)),
    mnar_col_prob = as.numeric(pick("mnar_col_prob", 1.0))
  )
}

resolve_opt_config <- function(config, cli_max_iter = NULL) {
  v <- cli_max_iter
  if (is.null(v) && is.list(config) && is.list(config$optimizer)) {
    v <- config$optimizer$max_iter
  }
  if (is.null(v) && is.list(config)) v <- config$max_iter
  if (is.null(v)) v <- 1000L
  list(max_iter = as.integer(v))
}

# -------------------------- Main --------------------------
if (!exists("RUNNER_MIRT_LIBRARY_MODE", inherits = FALSE) ||
    !isTRUE(RUNNER_MIRT_LIBRARY_MODE)) {
  config <- load_config(DEFAULT_CONFIG_FILE)
  sim_cfg <- resolve_sim_config(config)
  miss_cfg <- resolve_miss_config(config)
  opt_cfg <- resolve_opt_config(config, cli_max_iter = NULL)

  rep_cfg <- if (is.list(config)) config$replicates else NULL
  n_rep <- if (is.list(rep_cfg) && !is.null(rep_cfg$n_rep)) {
    as.integer(rep_cfg$n_rep)
  } else if (is.list(config) && !is.null(config$n_rep)) {
    as.integer(config$n_rep)
  } else {
    10L
  }
  base_seed <- if (is.list(rep_cfg) && !is.null(rep_cfg$base_seed)) {
    as.integer(rep_cfg$base_seed)
  } else if (is.list(config) && !is.null(config$base_seed)) {
    as.integer(config$base_seed)
  } else {
    1L
  }
  n_cores <- if (is.list(config) && !is.null(config$n_cores)) {
    as.integer(config$n_cores)
  } else {
    1L
  }

  seeds <- base_seed + seq_len(n_rep) - 1L

  run_method(
    sim_cfg = sim_cfg,
    miss_cfg = miss_cfg,
    opt_cfg = opt_cfg,
    seeds = seeds,
    n_cores = n_cores
  )
}
