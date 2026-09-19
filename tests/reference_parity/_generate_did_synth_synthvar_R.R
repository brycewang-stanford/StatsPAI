#!/usr/bin/env Rscript
# Frozen R references for tests/reference_parity/test_did_synth_synthvar_parity.py
# (did_synth family, cluster "synthvar", phase-3 cross-language campaign).
#
#   sp.demeaned_synth  vs augsynth::augsynth(progfunc = "None", fixedeff = TRUE)
#                         (augsynth 0.2.0 — de-meaned SCM; weights + gap path)
#   sp.robust_synth    vs scpi::scest(w.constr = list(name = "ols")) on
#                         scdata(constant = TRUE) (scpi 4.0.1), and
#                         glmnet::glmnet (ridge / lasso / elastic net with an
#                         unpenalised intercept)
#   sp.staggered_synth vs augsynth::multisynth (augsynth 0.2.0): nu, lambda,
#                         fixedeff, n_leads, n_lags, time_cohort, jackknife SE
#   sp.discos          vs DiSCos::DiSCo (DiSCos 0.1.4): quantile weights,
#                         counterfactual quantile functions, CDF-mixture
#                         weights, permutation statistics
#
# Random draws. DiSCo draws its quantile nodes (runif(M) per period) and its
# CDF grid (runif(G) per period) from R's stream after
# RNGkind("L'Ecuyer-CMRG"); set.seed(seed). With num.cores = 1 the periods are
# processed by a plain lapply, so the stream is replayed below call for call
# (period by period: runif(G) inside getGrid, then runif(M) inside
# DiSCo_weights_reg; then, for the permutation test, runif(M) per
# (control, pre-period)). The replayed draws are checked against DiSCo's own
# per-period weights and grids before they are written out, and the Python
# side feeds them to sp.discos(q_nodes=..., cdf_grid=...). In mixture mode the
# grids are read back from DiSCo's result instead (the SCS solve between
# periods consumes R random numbers, so a replay would drift).
#
# multisynth's QP is solved by OSQP; eps_abs = eps_rel = 1e-12 are passed so
# that the reference is not limited by the solver's default 1e-4 tolerance.
#
# CVXR versions. DiSCos solves its CDF-mixture LP through CVXR (SCS); this
# script loads CVXR from the shared site library (1.8.2, what DiSCos was
# installed against) at the top. scpi 4.0.1 needs CVXR > 1.9, so the scpi
# reference is computed in a separate Rscript process that puts the private
# library _fixtures/_rlib_did_synth_scpi (CVXR 1.9.2 + highs, git-ignored;
# see _generate_did_synth_scpi_R.R) first on .libPaths().
#
# DiSCo's mixture LP. DiSCos hands min ||F w - f||_1 s.t. sum(w) = 1 (w >= 0
# with simplex) to SCS, a first-order conic solver stopped at eps 1e-6, so
# its weights carry ~1e-6 solver error. The same LP (DiSCo's own CDF matrix)
# is also solved with GLPK's simplex method via CVXR — an exact LP solver
# independent of the HiGHS solver StatsPAI uses — and both L1 objectives are
# frozen, so the weights can be held to machine precision against GLPK and
# to SCS's tolerance against DiSCo itself.
#
# Regenerate (after _generate_did_synth_synthvar_data.py), from the repo root:
#   Rscript tests/reference_parity/_generate_did_synth_synthvar_R.R

script_dir <- tryCatch(
  dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(CVXR)
  library(DiSCos)
  library(augsynth)
  library(glmnet)
})

num <- function(x) as.numeric(x)

# ---------------------------------------------------------------------------
# 1. demeaned_synth  vs  augsynth(progfunc = "None", fixedeff = TRUE)
# ---------------------------------------------------------------------------
single <- read.csv(file.path(fixtures, "did_synth_synthvar_single.csv"))
aug <- suppressMessages(augsynth(y ~ treated, unit, time, single,
                                 progfunc = "None", scm = TRUE, fixedeff = TRUE))
aug_gap <- num(predict(aug, att = TRUE))
demeaned <- list(
  control_units = as.integer(rownames(aug$weights)),
  weights = num(aug$weights),
  gap = aug_gap,
  att_post_mean = mean(aug_gap[21:30]),
  treatment_time = 21L
)

# ---------------------------------------------------------------------------
# 2. robust_synth  vs  scpi OLS-with-constant and glmnet
# ---------------------------------------------------------------------------
W <- reshape(single[, c("unit", "time", "y")], idvar = "time", timevar = "unit",
             direction = "wide")
W <- W[order(W$time), ]
y_pre <- W[["y.1"]][1:20]
X_pre <- as.matrix(W[1:20, paste0("y.", 2:9)])
X_all <- as.matrix(W[, paste0("y.", 2:9)])
n_pre <- length(y_pre)
s_y <- sqrt(mean((y_pre - mean(y_pre))^2))  # 1/n SD: glmnet's internal y scale
glmnet_fit <- function(l2, l1) {
  # Target objective  ||y - mu - X w||^2 + l2 ||w||^2 + l1 ||w||_1.
  # glmnet minimises (1/2n) RSS + lambda[(1 - a)/2 ||b||^2 + a ||b||_1]; with
  # y scaled to unit (1/n) SD (so glmnet's own standardisation of y is the
  # identity) this is the target objective for  n lambda (1 - a) = l2,
  # 2 n lambda a = l1 / s_y.
  a_num <- l1 / (2 * s_y)
  lam <- (l2 + a_num) / n_pre
  al <- a_num / (l2 + a_num)
  f <- glmnet(X_pre, y_pre / s_y, family = "gaussian", alpha = al, lambda = lam,
              standardize = FALSE, intercept = TRUE, thresh = 1e-24, maxit = 1e7)
  list(l2 = l2, l1 = l1, glmnet_lambda = lam, glmnet_alpha = al,
       intercept = s_y * num(f$a0), weights = s_y * num(as.matrix(f$beta)))
}
glmnet_cases <- list(glmnet_fit(0.01, 0), glmnet_fit(5, 0), glmnet_fit(5, 3),
                     glmnet_fit(0, 2))

scpi_lib <- normalizePath(file.path(fixtures, "_rlib_did_synth_scpi"))
scpi_json <- tempfile(fileext = ".json")
scpi_code <- sprintf(paste(
  '.libPaths(c("%s", .libPaths()))',
  'suppressPackageStartupMessages({library(scpi); library(jsonlite)})',
  'single <- read.csv("%s")',
  'sdat <- scdata(df = single, id.var = "unit", time.var = "time", outcome.var = "y",',
  '  period.pre = 1:20, period.post = 21:30, unit.tr = 1, unit.co = 2:9, constant = TRUE)',
  'sols <- scest(sdat, w.constr = list(name = "ols"))',
  'writeLines(toJSON(list(b = as.numeric(sols$est.results$b),',
  '  y_pre_fit = as.numeric(sols$est.results$Y.pre.fit),',
  '  y_post_fit = as.numeric(sols$est.results$Y.post.fit),',
  '  scpi = as.character(packageVersion("scpi")),',
  '  CVXR = as.character(packageVersion("CVXR"))), digits = I(17)), "%s")',
  sep = "\n"), scpi_lib, normalizePath(file.path(fixtures, "did_synth_synthvar_single.csv")),
  scpi_json)
stopifnot(system2("Rscript", c("-e", shQuote(scpi_code))) == 0)
sc_out <- fromJSON(scpi_json)
b <- sc_out$b
ols_lm <- lm(y_pre ~ X_pre)
robust <- list(
  ols = list(weights = b[1:8], intercept = b[9],
             y_pre_fit = sc_out$y_pre_fit, y_post_fit = sc_out$y_post_fit,
             scpi_version = sc_out$scpi, scpi_CVXR_version = sc_out$CVXR),
  ols_lm = list(intercept = num(coef(ols_lm)[1]), weights = num(coef(ols_lm)[-1])),
  glmnet = glmnet_cases,
  y_sd_1n = s_y
)

# ---------------------------------------------------------------------------
# 3. staggered_synth  vs  multisynth
# ---------------------------------------------------------------------------
stag <- read.csv(file.path(fixtures, "did_synth_synthvar_stag.csv"))
ms_case <- function(name, jackknife = FALSE, ...) {
  args <- list(...)
  m <- do.call(multisynth, c(list(form = y ~ treated, unit = quote(unit),
                                   time = quote(time), data = stag,
                                   eps_abs = 1e-12, eps_rel = 1e-12), args))
  p <- predict(m, att = TRUE)
  d <- ncol(m$data$X)
  ev <- seq(-d, nrow(p) - 2 - d)
  out <- list(
    name = name,
    args = args,
    nu = m$nu, n_leads = m$n_leads, n_lags = m$n_lags,
    global_l2 = m$global_l2, ind_l2 = m$ind_l2,
    att = p[nrow(p), 1],
    group_att = num(p[nrow(p), -1]),
    event_time = ev,
    event_att = num(p[-nrow(p), 1]),
    weights = unname(m$weights)
  )
  if (jackknife) {
    s <- summary(m, inf_type = "jackknife")
    av <- s$att[s$att$Level == "Average" & is.na(s$att$Time), ]
    out$jackknife_se <- av$Std.Error
  }
  out
}
ms <- list(
  ms_case("sep_nu0_nofe", nu = 0, fixedeff = FALSE, n_leads = 6),
  ms_case("nu05_fe", nu = 0.5, fixedeff = TRUE, n_leads = 6),
  ms_case("auto_fe", jackknife = TRUE, fixedeff = TRUE, n_leads = 6),
  ms_case("cohort_auto_fe", time_cohort = TRUE, fixedeff = TRUE, n_leads = 6),
  ms_case("auto_fe_lags5_lambda", jackknife = TRUE, fixedeff = TRUE, n_leads = 6,
          n_lags = 5, lambda = 0.1),
  ms_case("r_defaults")
)

# ---------------------------------------------------------------------------
# 4. discos  vs  DiSCos::DiSCo
# ---------------------------------------------------------------------------
micro <- fread(file.path(fixtures, "did_synth_synthvar_micro.csv"))
dt <- data.table(id_col = micro$id, time_col = micro$time, y_col = micro$y)
M <- 200L; G <- 100L; T0 <- 3L; periods <- 1:6
ctrl_ids <- unique(dt[id_col != 1]$id_col)
stopifnot(identical(as.numeric(ctrl_ids), c(2, 3, 4, 5, 6)))
J <- length(ctrl_ids)

replay <- function(seed, mixture, n_perm_fits) {
  RNGkind("L'Ecuyer-CMRG"); set.seed(seed)
  grids <- list(); nodes <- list()
  for (p in periods) {
    tgt <- dt[id_col == 1 & time_col == p]$y_col
    ctr <- lapply(ctrl_ids, function(j) dt[id_col == j & time_col == p]$y_col)
    gmin <- floor(min(c(tgt, unlist(ctr))) * 10) / 10
    gmax <- ceiling(max(c(tgt, unlist(ctr))) * 10) / 10
    grids[[p]] <- sort(runif(G, gmin - 0.25, gmax + 0.25))
    if (!mixture) nodes[[p]] <- runif(M, 0, 1)
  }
  perm <- lapply(seq_len(n_perm_fits), function(k) lapply(1:T0, function(t) runif(M, 0, 1)))
  list(grids = grids, nodes = nodes, perm = perm)
}

disco_case <- function(name, seed, mixture, simplex, permutation) {
  res <- DiSCo(dt, id_col.target = 1, t0 = 4, M = M, G = G, seed = seed,
               num.cores = 1L, simplex = simplex, mixture = mixture,
               permutation = permutation, graph = FALSE)
  rp <- replay(seed, mixture, if (permutation && !mixture) J else 0L)
  # Sanity check of the replay against DiSCo's own per-period weights.
  if (!mixture) {
    for (t in 1:T0) {
      tgt <- sort(dt[id_col == 1 & time_col == t]$y_col)
      ctr <- lapply(ctrl_ids, function(j) sort(dt[id_col == j & time_col == t]$y_col))
      cs <- sapply(ctr, function(x) DiSCos:::quant7_sorted(x, rp$nodes[[t]]))
      ts <- DiSCos:::quant7_sorted(tgt, rp$nodes[[t]])
      sc <- norm(cs, "2")
      w_rep <- pracma::lsqlincon(cs / sc, ts / sc, Aeq = matrix(1, 1, J), beq = 1,
                                 lb = if (simplex) 0 else NULL, ub = 1)
      stopifnot(max(abs(w_rep - res$results.periods[[t]]$DiSCo$weights)) < 1e-12)
    }
  }
  out <- list(
    name = name, seed = seed, mixture = mixture, simplex = simplex,
    M = M, G = G, t0 = 4L, control_ids = as.integer(ctrl_ids),
    weights = num(res$weights),
    period_weights = lapply(1:T0, function(t)
      num(if (mixture) res$results.periods[[t]]$mixture$weights
          else res$results.periods[[t]]$DiSCo$weights)),
    target_quantiles = lapply(periods, function(t) num(res$results.periods[[t]]$target$quantiles)),
    counterfactual_quantiles = lapply(periods, function(t) num(res$results.periods[[t]]$DiSCo$quantile)),
    q_nodes = rp$nodes[1:T0],
    # the CDF grid DiSCo actually used (read back, not replayed: in mixture
    # mode the CVXR/SCS solve between periods touches R's RNG stream)
    cdf_grid = lapply(periods, function(t) num(res$results.periods[[t]]$target$grid))
  )
  if (!mixture) {
    stopifnot(max(abs(unlist(rp$grids) - unlist(out$cdf_grid))) == 0)
  }
  if (mixture) {
    out$mixture_distance <- lapply(1:T0, function(t) res$results.periods[[t]]$mixture$distance)
    # Same LP, exact simplex solver (GLPK), and both L1 objectives.
    glpk <- lapply(1:T0, function(t) {
      Fm <- res$results.periods[[t]]$controls$cdf  # col 1 = target
      w <- Variable(J)
      prob <- Problem(Minimize(cvxr_norm(Fm[, -1] %*% w - Fm[, 1], 1)),
                      if (simplex) list(w >= 0, sum_entries(w) == 1)
                      else list(sum_entries(w) == 1))
      val <- psolve(prob, solver = "GLPK")
      wg <- num(value(w))
      list(weights = wg,
           objective_glpk = sum(abs(Fm[, -1] %*% wg - Fm[, 1])),
           objective_scs = sum(abs(Fm[, -1] %*% res$results.periods[[t]]$mixture$weights - Fm[, 1])))
    })
    out$glpk <- glpk
  }
  tea <- DiSCoTEA(res, agg = "quantileDiff", graph = FALSE)
  out$quantile_diff <- lapply(periods, function(t) num(tea$treats[[t]]))
  if (permutation) {
    out$perm_p_value <- res$perm$p_overall
    out$perm_target_dist <- num(res$perm$distt)
    out$perm_control_dist <- unname(lapply(res$perm$distp, num))
    out$perm_nodes <- rp$perm
  }
  out
}

# Patched permutation. DiSCos 0.1.4's DiSCo_per_iter builds the pseudo-
# controls' quantile matrix perc.q with column 1 (the original treated unit,
# which IS in the pseudo-donor pool and gets a weight) left at zero:
#   perc.q[[i]] <- matrix(0, ...); perc.q[[i]][, j + 1] <- c_df.q[[i]][, keepcon[j]]
# so every placebo counterfactual drops that unit's quantile contribution.
# The function below is DiSCo_per_iter verbatim (quantile branch) with that
# one column filled; it is evaluated on the SAME replayed nodes, so the
# Python permutation (which uses every pool member) can be compared with it.
perm_iter_patched <- function(c_df, c_df.q, t_df, T0, peridx, evgrid, idx,
                              nodes_idx, simplex) {
  keepcon <- peridx[-idx]
  n_per <- length(c_df)
  perc <- list(); perc.q <- list(); pert <- list()
  for (i in 1:n_per) {
    perc[[i]] <- c(list(t_df[[i]]), lapply(keepcon, function(k) c_df[[i]][[k]]))
    perc.q[[i]] <- cbind(DiSCos:::quant7_sorted(t_df[[i]], evgrid),
                         c_df.q[[i]][, keepcon, drop = FALSE])
    pert[[i]] <- c_df[[i]][[idx]]
  }
  lambda_tp <- lapply(1:T0, function(t) {
    cs <- sapply(perc[[t]], function(x) DiSCos:::quant7_sorted(x, nodes_idx[[t]]))
    ts <- DiSCos:::quant7_sorted(pert[[t]], nodes_idx[[t]])
    sc <- norm(cs, "2")
    pracma::lsqlincon(cs / sc, ts / sc, Aeq = matrix(1, 1, ncol(cs)), beq = 1,
                      lb = if (simplex) 0 else NULL, ub = 1)
  })
  lambda.opt <- matrix(unlist(lambda_tp), ncol = T0) %*% rep(1 / T0, T0)
  vapply(1:n_per, function(t) {
    bc <- perc.q[[t]] %*% lambda.opt
    tq <- DiSCos:::quant7_sorted(pert[[t]], evgrid)
    mean((bc - tq)^2)
  }, numeric(1))
}

dc_q <- disco_case("quantile", 11L, mixture = FALSE, simplex = FALSE, permutation = TRUE)
dc_qs <- disco_case("quantile_simplex", 12L, mixture = FALSE, simplex = TRUE, permutation = FALSE)
dc_mix <- disco_case("mixture_simplex", 13L, mixture = TRUE, simplex = TRUE, permutation = FALSE)

evgrid <- seq(from = 0, to = 1, length.out = G + 1)
c_df <- lapply(periods, function(t) lapply(ctrl_ids, function(j) sort(dt[id_col == j & time_col == t]$y_col)))
t_df <- lapply(periods, function(t) sort(dt[id_col == 1 & time_col == t]$y_col))
c_df.q <- lapply(periods, function(t) sapply(c_df[[t]], function(x) DiSCos:::quant7_sorted(x, evgrid)))
patched <- lapply(1:J, function(idx) perm_iter_patched(c_df, c_df.q, t_df, T0, 1:J,
                                                      evgrid, idx, dc_q$perm_nodes[[idx]],
                                                      simplex = FALSE))
distall <- rbind(do.call(rbind, patched), dc_q$perm_target_dist)
Rstat <- apply(distall, 1, function(x) sqrt(mean(x[(T0 + 1):length(x)])) / sqrt(mean(x[1:T0])))
dc_q$patched_perm_control_dist <- lapply(patched, num)
dc_q$patched_perm_p_value <- (rank(-Rstat)[length(Rstat)]) / nrow(distall)

out <- list(
  meta = list(
    R_version = R.version.string,
    augsynth = as.character(packageVersion("augsynth")),
    osqp = as.character(packageVersion("osqp")),
    scpi = sc_out$scpi,
    CVXR = as.character(packageVersion("CVXR")),
    Rglpk = as.character(packageVersion("Rglpk")),
    glmnet = as.character(packageVersion("glmnet")),
    DiSCos = as.character(packageVersion("DiSCos")),
    pracma = as.character(packageVersion("pracma")),
    quadprog = as.character(packageVersion("quadprog")),
    generated = format(Sys.Date())
  ),
  demeaned = demeaned,
  robust = robust,
  multisynth = ms,
  discos = list(dc_q, dc_qs, dc_mix)
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"),
           file.path(fixtures, "did_synth_synthvar_R.json"))
cat("wrote", file.path(fixtures, "did_synth_synthvar_R.json"), "\n")
