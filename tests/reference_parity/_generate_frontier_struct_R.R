# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_frontier_struct_R_parity.py
#
# Run _generate_frontier_struct_data.py first, then from this directory:
#     Rscript _generate_frontier_struct_R.R
# Writes _fixtures/frontier_struct_R.json (numbers at 17 significant digits).
#
# Packages: sfaR (sfalcmcross, sfacross), metafrontier (metafrontier).
#
# Conventions pinned here
# -----------------------
# * sfaR parameterises the variance components as log VARIANCES
#   (Zu_(Intercept) = log sigma_u^2); StatsPAI reports log standard
#   deviations. The test maps ln_sigma = Zu / 2 and se(ln_sigma) = se(Zu) / 2.
# * sfalcmcross class probability: P(class 1) = logistic(Cl1' z), the same
#   link as sp.lcsf; vcov = inverse of sfaR's analytic Hessian (OIM).
# * The latent-class likelihood is flat in class-1 sigma_u, so four sfaR
#   optimisers are run and the one with the smallest max |gradient| is kept
#   (its loglik is written, so the test can check StatsPAI is at least as high).
# * metafrontier: method = "sfa", engine = "sfaR" (group frontiers by
#   sfaR::sfacross, half-normal), meta_type = "deterministic",
#   objective = "lp" (lpSolveAPI, constraints x_i'b* >= x_i'b_{k(i)} against
#   the OWN group frontier only), estimator = "bc88" (TE_group).
# * malmquist: one sfaR::sfacross half-normal frontier per period (best of
#   four optimisers by max |gradient|, as for lcsf); the index
#   arithmetic (D^s(x, y) = exp(y - x'b_s)) is re-done here in R from sfaR's
#   betas -- it replicates the formula, it is not a second package.
# ---------------------------------------------------------------------------
suppressMessages({
  library(sfaR)
  library(metafrontier)
  library(jsonlite)
})

fx <- "_fixtures"
out <- list(
  versions = list(
    R = R.version.string,
    sfaR = as.character(packageVersion("sfaR")),
    metafrontier = as.character(packageVersion("metafrontier")),
    lpSolveAPI = as.character(packageVersion("lpSolveAPI"))
  )
)

# ---------------------------------------------------------------- lcsf -----
lc_fit <- function(formula, thet, data, S) {
  best <- NULL
  for (m in c("bfgs", "nr", "ucminf", "bhhh")) {
    f <- try(sfalcmcross(formula, thet = thet, data = data, S = S,
                         udist = "hnormal", lcmClasses = 2, method = m,
                         gradtol = 1e-10, tol = 1e-14, itermax = 10000),
             silent = TRUE)
    if (inherits(f, "try-error")) next
    g <- max(abs(f$gradient))
    if (is.null(best) || g < best$g) best <- list(fit = f, g = g, method = m)
  }
  f <- best$fit
  list(
    method = best$method,
    max_abs_gradient = best$g,
    loglik = f$mlLoglik,
    coef = as.list(setNames(unname(coef(f)), make.unique(names(coef(f))))),
    se = as.list(setNames(unname(sqrt(diag(vcov(f)))), make.unique(names(coef(f)))))
  )
}
lc <- read.csv(file.path(fx, "frontier_struct_lcsf.csv"))
out$lcsf_prod_z <- lc_fit(y ~ x1 + x2, ~ z, lc, S = 1)
lc_cost <- lc
lc_cost$c <- -lc_cost$y
out$lcsf_cost_noz <- lc_fit(c ~ x1 + x2, ~ 1, lc_cost, S = -1)

# -------------------------------------------------------- metafrontier -----
md <- read.csv(file.path(fx, "frontier_struct_meta.csv"))
mf <- metafrontier(y ~ x1 + x2, data = md, group = "group", method = "sfa",
                   meta_type = "deterministic", dist = "hnormal",
                   engine = "sfaR", objective = "lp", estimator = "bc88")
out$meta <- list(
  meta_coef = as.list(mf$meta_coef),
  group_coef = lapply(mf$group_coef, function(b) as.list(b)),
  tgr = unname(mf$tgr),
  te_group = unname(mf$te_group),
  te_meta = unname(mf$te_meta),
  group = as.character(md$group)
)

# ----------------------------------------------------------- malmquist -----
pd_ <- read.csv(file.path(fx, "frontier_struct_malm.csv"))
periods <- sort(unique(pd_$t))
betas <- list()
for (tt in periods) {
  f <- NULL
  for (m in c("bfgs", "nr", "ucminf", "bhhh")) {
    ff <- try(sfacross(y ~ x1 + x2, data = pd_[pd_$t == tt, ], S = 1,
                       udist = "hnormal", method = m, gradtol = 1e-10,
                       tol = 1e-14), silent = TRUE)
    if (inherits(ff, "try-error")) next
    if (is.null(f) || max(abs(ff$gradient)) < max(abs(f$gradient))) f <- ff
  }
  b <- coef(f)
  betas[[as.character(tt)]] <- list(
    beta = unname(b[1:3]),
    Zu = unname(b["Zu_(Intercept)"]),
    Zv = unname(b["Zv_(Intercept)"]),
    loglik = f$mlLoglik,
    max_abs_gradient = max(abs(f$gradient))
  )
}
rows <- list()
for (k in seq_len(length(periods) - 1)) {
  t1 <- periods[k]; t2 <- periods[k + 1]
  b1 <- betas[[as.character(t1)]]$beta; b2 <- betas[[as.character(t2)]]$beta
  a <- pd_[pd_$t == t1, ]; b <- pd_[pd_$t == t2, ]
  b <- b[match(a$id, b$id), ]
  X1 <- cbind(1, a$x1, a$x2); X2 <- cbind(1, b$x1, b$x2)
  lD_t_t   <- a$y - X1 %*% b1
  lD_tp_tp <- b$y - X2 %*% b2
  lD_t_tp  <- b$y - X2 %*% b1
  lD_tp_t  <- a$y - X1 %*% b2
  lM  <- 0.5 * ((lD_t_tp - lD_t_t) + (lD_tp_tp - lD_tp_t))
  lEC <- lD_tp_tp - lD_t_t
  rows[[k]] <- data.frame(id = a$id, t_from = t1, t_to = t2,
                          m_index = exp(lM), ec = exp(lEC), tc = exp(lM - lEC))
}
it <- do.call(rbind, rows)
it <- it[order(it$id, it$t_from), ]
out$malmquist <- list(betas = betas, index = as.list(it))

writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE),
           file.path(fx, "frontier_struct_R.json"))
cat("wrote", file.path(fx, "frontier_struct_R.json"), "\n")
