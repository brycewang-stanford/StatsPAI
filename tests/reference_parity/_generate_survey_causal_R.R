#!/usr/bin/env Rscript
# Frozen reference for design= on the causal estimators (the shared
# linearisation helper statspai.survey.linearization and sp.ipw / sp.aipw /
# sp.regress / sp.logit with design=SurveyDesign) vs R survey 4.x.
#
# Regenerate (from the repository root):
#   Rscript tests/reference_parity/_generate_survey_causal_R.R
#
# The data file _fixtures/survey_causal_data.csv is written once by this
# script (fixed seed, values rounded to 6 decimals so the CSV bytes are the
# input both sides read) and never overwritten afterwards.
#
# What is frozen
# --------------
# * helper: survey::svyrecvar on a fixed contribution matrix u for seven
#   designs (strata + PSU + fpc as count / fraction, no fpc, clusters only,
#   element sampling with fpc, weights only, four lonely-PSU rules) and a
#   raked design (svyrecvar with the design's postStrata).
# * IPW (Hajek ATE and ATT) with the propensity logit from svyglm
#   (quasibinomial): the stacked estimating equations (logit score, the two
#   weighted means) written out by hand, their Jacobian A in closed form
#   (checked against numDeriv::jacobian), linearised contributions
#   -A^{-1} w psi_i, and svyrecvar of those (Binder 1983 linearisation).
# * AIPW ATE, full-sample logit + per-arm least squares, no trimming: the
#   same construction with the five-block stacked system; and the
#   "influence" variance that holds the nuisance fits fixed.
# * TMLE (tmle::tmle, SL.glm nuisances, gbound = 0.025, cvQinit = FALSE,
#   obsWeights = design weights scaled to mean one): its weighted
#   influence curves IC.ATE / IC.EY1 / IC.EY0 divided by n are the
#   contributions, and svyrecvar of those is the design variance.
# * DML (PLR partialling out, IRM ATE) cross-fitted on three folds of whole
#   PSUs (fold = PSU number mod 3) with weighted lm / glm nuisances; the
#   weighted DML2 solution and svyrecvar of -w psi / sum(w psi_a).
# * Survival (_fixtures/survey_causal_surv.csv, a time / status pair per
#   row of the main file, written once): svycoxph(Surv(time, status) ~ d +
#   x1) (Efron ties; times rounded to 2 decimals so there are ties), and
#   the Kaplan-Meier curve overall and by d: survfit(weights = w) for the
#   product-limit estimate and resid(survfit, type = "surv") for each
#   row's infinitesimal-jackknife influence r_i(t); svyrecvar of w_i r_i(t)
#   is the design variance. (svykm(se = TRUE) is not used: on this file its
#   curve differs from the product-limit one by up to 0.019, while
#   svykm(se = FALSE) equals survfit to 2e-15. R packages here are used as
#   black boxes: outputs and documentation only.)
# * svyglm gaussian y ~ d + x1 + x2 and quasibinomial yb ~ d + x1 (tight
#   control, refitted at the MLE as in _generate_survey_design_R.R).
suppressPackageStartupMessages({
  library(survey); library(jsonlite); library(numDeriv); library(tmle)
  library(survival)
})

csv <- "tests/reference_parity/_fixtures/survey_causal_data.csv"
if (!file.exists(csv)) {
  set.seed(20261019)
  n_psu <- c(3, 4, 5, 3, 6)
  pop_mult <- c(6, 3, 9, 4, 5)
  rows <- list()
  for (h in seq_along(n_psu)) {
    for (j in seq_len(n_psu[h])) {
      m <- sample(10:20, 1)
      u <- rnorm(1, sd = 0.8)
      x1 <- round(rnorm(m, mean = 0.2 * h + 0.5 * u, sd = 1), 6)
      x2 <- rbinom(m, 1, 0.4 + 0.05 * h)
      d <- rbinom(m, 1, plogis(-0.3 + 0.6 * x1 - 0.5 * x2 + 0.3 * u))
      y <- round(1 + 2 * d + 0.5 * d * x1 + x1 + 0.5 * x2 + u + rnorm(m), 6)
      yb <- rbinom(m, 1, plogis(-0.5 + 0.8 * d + 0.5 * x1 + 0.3 * u))
      # informative weights: treated units and stratum both move them
      w <- round(pop_mult[h] * runif(m, 5, 15) * (1 + 0.4 * d), 4)
      rows[[length(rows) + 1]] <- data.frame(
        stratum = h, psu = j, w = w, d = d, y = y, yb = yb, x1 = x1, x2 = x2,
        fpc_N = n_psu[h] * pop_mult[h],
        stratum_l = ifelse(h == 2 & j == 4, 9L, h))
    }
  }
  df <- do.call(rbind, rows)
  df$fpc_f <- round(ave(df$psu, df$stratum, FUN = function(p) length(unique(p))) /
                      df$fpc_N, 12)
  df$fpc_el <- ave(df$y, df$stratum, FUN = length) * 8
  write.csv(df, csv, row.names = FALSE)
}
df <- read.csv(csv)
df$psu_global <- df$stratum * 100 + df$psu
n <- nrow(df)
dig <- function(x) unname(as.numeric(x))

designs <- list(
  full = function() svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                              fpc = ~fpc_N, nest = TRUE, data = df),
  frac = function() svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                              fpc = ~fpc_f, nest = TRUE, data = df),
  nofpc = function() svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                               nest = TRUE, data = df),
  cluster_only = function() svydesign(ids = ~psu_global, weights = ~w, data = df),
  element_fpc = function() svydesign(ids = ~1, strata = ~stratum, weights = ~w,
                                     fpc = ~fpc_el, data = df),
  weights_only = function() svydesign(ids = ~1, weights = ~w, data = df))
lonely <- c("remove", "certainty", "adjust", "average")
lonely_design <- function() svydesign(ids = ~psu, strata = ~stratum_l,
                                      weights = ~w, nest = TRUE, data = df)
recvar <- function(u, des) {
  unname(svyrecvar(u, des$cluster, des$strata, des$fpc,
                   postStrata = des$postStrata))
}

# ---------------------------------------------------------------- helper
u_fixed <- cbind(df$w * df$y / sum(df$w), df$w * (df$x1 - 0.3) * df$d)
res <- list(helper = list())
for (nm in names(designs)) {
  des <- designs[[nm]]()
  res$helper[[nm]] <- list(vcov = recvar(u_fixed, des), degf = degf(des))
}
for (lp in lonely) {
  options(survey.lonely.psu = lp)
  des <- lonely_design()
  res$helper[[paste0("lonely_", lp)]] <- list(vcov = recvar(u_fixed, des),
                                             degf = degf(des))
}
options(survey.lonely.psu = "fail")
# raking to x2 counts: contributions are the calibrated weight times a
# per-unit quantity z
pop_x2 <- data.frame(x2 = c(0, 1), Freq = c(5200, 3800))
cal <- calibrate(designs$nofpc(), list(~x2), list(pop_x2), calfun = "raking",
                 epsilon = 1e-13, maxit = 1000)
z <- cbind(df$y, (df$x1 - 0.3) * df$d)
res$helper$raked <- list(vcov = recvar(weights(cal) * z / sum(weights(cal)), cal),
                         weights = dig(weights(cal)),
                         margins = list(x2_0 = 5200, x2_1 = 3800))

# Logit fits at the MLE: svyglm's one-pass bread is one IRLS step stale
# (see _generate_survey_design_R.R), so refit from the converged values.
tight <- glm.control(epsilon = 1e-14, maxit = 100)
fit_at_mle <- function(form, des, coef_only = TRUE) {
  g <- do.call(svyglm, list(formula = form, design = des,
                            family = quasibinomial(), control = tight))
  g <- do.call(svyglm, list(formula = form, design = des,
                            family = quasibinomial(), control = tight,
                            start = unname(coef(g))))
  if (coef_only) unname(coef(g)) else g
}

# ------------------------------------------------------------- IPW / AIPW
X <- cbind(1, df$x1, df$x2)
Y <- df$y; D <- df$d
k <- ncol(X)

ipw_block <- function(des, estimand) {
  w <- weights(des)
  gam <- fit_at_mle(d ~ x1 + x2, des)
  arms <- function(e) {
    if (estimand == "ATE") list(a1 = D / e, a0 = (1 - D) / (1 - e))
    else list(a1 = D, a0 = (1 - D) * e / (1 - e))
  }
  psi <- function(theta) {          # n x (k + 2) estimating functions
    e <- plogis(drop(X %*% theta[1:k]))
    a <- arms(e)
    cbind(X * (D - e), a$a1 * (Y - theta[k + 1]), a$a0 * (Y - theta[k + 2]))
  }
  e <- plogis(drop(X %*% gam))
  a <- arms(e)
  mu1 <- sum(w * a$a1 * Y) / sum(w * a$a1)
  mu0 <- sum(w * a$a0 * Y) / sum(w * a$a0)
  theta <- c(gam, mu1, mu0)
  # closed-form Jacobian of U(theta) = sum_i w_i psi_i(theta)
  A <- matrix(0, k + 2, k + 2)
  A[1:k, 1:k] <- -crossprod(X * (w * e * (1 - e)), X)
  if (estimand == "ATE") {
    da1 <- -D * (1 - e) / e; da0 <- (1 - D) * e / (1 - e)
  } else {
    da1 <- rep(0, n); da0 <- (1 - D) * e / (1 - e)
  }
  A[k + 1, 1:k] <- colSums(X * (w * da1 * (Y - mu1)))
  A[k + 2, 1:k] <- colSums(X * (w * da0 * (Y - mu0)))
  A[k + 1, k + 1] <- -sum(w * a$a1)
  A[k + 2, k + 2] <- -sum(w * a$a0)
  A_num <- jacobian(function(th) colSums(w * psi(th)), theta)
  stopifnot(max(abs(A - A_num)) / max(abs(A)) < 1e-7)
  u <- -(w * psi(theta)) %*% t(solve(A))      # rows: -A^{-1} w_i psi_i
  V <- recvar(u, des)
  cvec <- c(rep(0, k), 1, -1)
  list(estimate = mu1 - mu0, mu = c(mu1, mu0), gamma = gam,
       se = sqrt(drop(t(cvec) %*% V %*% cvec)),
       se_mu = sqrt(diag(V)[k + 1:2]),
       jacobian_check = max(abs(A - A_num)) / max(abs(A)),
       contributions_tau = dig(u[, k + 1] - u[, k + 2]),
       degf = degf(des))
}

aipw_block <- function(des) {
  w <- weights(des)
  gam <- fit_at_mle(d ~ x1 + x2, des)
  b1 <- unname(coef(lm(y ~ x1 + x2, data = df, weights = w, subset = d == 1)))
  b0 <- unname(coef(lm(y ~ x1 + x2, data = df, weights = w, subset = d == 0)))
  nb <- 3 * k
  psi <- function(theta) {
    e <- plogis(drop(X %*% theta[1:k]))
    m1 <- drop(X %*% theta[k + 1:k]); m0 <- drop(X %*% theta[2 * k + 1:k])
    cbind(X * (D - e), X * (D * (Y - m1)), X * ((1 - D) * (Y - m0)),
          m1 + D * (Y - m1) / e - theta[nb + 1],
          m0 + (1 - D) * (Y - m0) / (1 - e) - theta[nb + 2])
  }
  e <- plogis(drop(X %*% gam))
  m1 <- drop(X %*% b1); m0 <- drop(X %*% b0)
  phi1 <- m1 + D * (Y - m1) / e; phi0 <- m0 + (1 - D) * (Y - m0) / (1 - e)
  M1 <- sum(w * phi1) / sum(w); M0 <- sum(w * phi0) / sum(w)
  theta <- c(gam, b1, b0, M1, M0)
  A <- matrix(0, nb + 2, nb + 2)
  A[1:k, 1:k] <- -crossprod(X * (w * e * (1 - e)), X)
  A[k + 1:k, k + 1:k] <- -crossprod(X * (w * D), X)
  A[2 * k + 1:k, 2 * k + 1:k] <- -crossprod(X * (w * (1 - D)), X)
  A[nb + 1, 1:k] <- colSums(X * (w * (-D * (Y - m1) * (1 - e) / e)))
  A[nb + 1, k + 1:k] <- colSums(X * (w * (1 - D / e)))
  A[nb + 2, 1:k] <- colSums(X * (w * ((1 - D) * (Y - m0) * e / (1 - e))))
  A[nb + 2, 2 * k + 1:k] <- colSums(X * (w * (1 - (1 - D) / (1 - e))))
  A[nb + 1, nb + 1] <- -sum(w); A[nb + 2, nb + 2] <- -sum(w)
  A_num <- jacobian(function(th) colSums(w * psi(th)), theta)
  stopifnot(max(abs(A - A_num)) / max(abs(A)) < 1e-7)
  u <- -(w * psi(theta)) %*% t(solve(A))
  V <- recvar(u, des)
  cvec <- c(rep(0, nb), 1, -1)
  # nuisance fits held fixed: contributions w (psi - tau) / sum(w)
  u_inf <- cbind(w * (phi1 - phi0 - (M1 - M0)) / sum(w),
                 w * (phi1 - M1) / sum(w), w * (phi0 - M0) / sum(w))
  V_inf <- recvar(u_inf, des)
  list(estimate = M1 - M0, mu = c(M1, M0),
       se_sandwich = sqrt(drop(t(cvec) %*% V %*% cvec)),
       se_mu_sandwich = sqrt(diag(V)[nb + 1:2]),
       se_influence = sqrt(V_inf[1, 1]),
       se_mu_influence = sqrt(diag(V_inf)[2:3]),
       jacobian_check = max(abs(A - A_num)) / max(abs(A)),
       degf = degf(des))
}

glm_block <- function(des) {
  g <- svyglm(y ~ d + x1 + x2, design = des)
  b <- fit_at_mle(yb ~ d + x1, des, coef_only = FALSE)
  list(gaussian = list(coef = dig(coef(g)), se = dig(SE(g)),
                       vcov = unname(vcov(g))),
       logit = list(coef = dig(coef(b)), se = dig(SE(b)),
                    vcov = unname(vcov(b))),
       degf = degf(des))
}

tmle_block <- function(des) {
  w <- weights(des); ow <- w * n / sum(w)
  out <- list()
  for (fam in c("gaussian", "binomial")) {
    Yv <- if (fam == "gaussian") df$y else df$yb
    f <- suppressWarnings(suppressMessages(tmle(
      Y = Yv, A = D, W = df[, c("x1", "x2")], family = fam,
      Q.SL.library = "SL.glm", g.SL.library = "SL.glm", gbound = 0.025,
      cvQinit = FALSE, obsWeights = ow)))
    ic <- cbind(f$estimates$IC$IC.ATE, f$estimates$IC$IC.EY1,
                f$estimates$IC$IC.EY0) / n
    V <- recvar(ic, des)
    out[[fam]] <- list(ate = f$estimates$ATE$psi,
                       ey1 = f$estimates$EY1$psi, ey0 = f$estimates$EY0$psi,
                       se = sqrt(diag(V)))
  }
  out
}

df$fold <- (as.integer(factor(df$psu_global)) - 1) %% 3
dml_block <- function(des) {
  w <- weights(des)
  yres <- dres <- g1 <- g0 <- m <- numeric(n)
  for (k in 0:2) {
    tr <- df$fold != k; te <- !tr
    tdf <- df[tr, ]; tdf$w_ <- w[tr]
    yres[te] <- df$y[te] - predict(lm(y ~ x1 + x2, data = tdf, weights = w_),
                                   df[te, ])
    dres[te] <- df$d[te] - predict(lm(d ~ x1 + x2, data = tdf, weights = w_),
                                   df[te, ])
    g1[te] <- predict(lm(y ~ x1 + x2, data = tdf[tdf$d == 1, ], weights = w_),
                      df[te, ])
    g0[te] <- predict(lm(y ~ x1 + x2, data = tdf[tdf$d == 0, ], weights = w_),
                      df[te, ])
    pm <- suppressWarnings(glm(d ~ x1 + x2, data = tdf, weights = w_,
                               family = quasibinomial(), control = tight))
    m[te] <- predict(pm, df[te, ], type = "response")
  }
  one <- function(psi_a, psi_b) {
    den <- sum(w * psi_a)
    theta <- -sum(w * psi_b) / den
    u <- -w * (psi_a * theta + psi_b) / den
    list(estimate = theta, se = sqrt(recvar(cbind(u), des)[1, 1]))
  }
  m <- pmin(pmax(m, 0.01), 0.99)
  list(plr = one(-dres^2, dres * yres),
       irm = one(rep(-1, n), g1 - g0 + D * (Y - g1) / m -
                   (1 - D) * (Y - g0) / (1 - m)),
       degf = degf(des))
}

surv_csv <- "tests/reference_parity/_fixtures/survey_causal_surv.csv"
if (!file.exists(surv_csv)) {
  set.seed(20261022)
  ev <- rexp(n, 0.15 * exp(0.5 * df$d + 0.3 * df$x1))
  ce <- rexp(n, 0.05)
  write.csv(data.frame(time = pmax(round(pmin(ev, ce), 2), 0.01),
                       status = as.integer(ev <= ce)), surv_csv,
            row.names = FALSE)
}
sv <- read.csv(surv_csv)
df$time <- sv$time; df$status <- sv$status
surv_block <- function(des) {
  w <- weights(des)
  cx <- svycoxph(Surv(time, status) ~ d + x1, design = des)
  km_one <- function(mask) {
    d_ <- df[mask, ]; d_$w_ <- w[mask]
    # do.call puts the data in the call, which resid() re-evaluates
    fit <- do.call(survfit, list(formula = Surv(time, status) ~ 1, data = d_,
                                 weights = d_$w_))
    tt <- fit$time[fit$n.event > 0]
    r <- matrix(0, n, length(tt))
    r[mask, ] <- resid(fit, times = tt, type = "surv")
    V <- recvar(w * r, des)
    list(time = tt, surv = summary(fit, times = tt)$surv,
         se = sqrt(diag(V)))
  }
  list(cox = list(coef = dig(coef(cx)), se = dig(SE(cx))),
       km_all = km_one(rep(TRUE, n)),
       km_d1 = km_one(df$d == 1), km_d0 = km_one(df$d == 0),
       degf = degf(des))
}

res$causal <- list()
for (nm in c("full", "nofpc", "cluster_only", "weights_only")) {
  des <- designs[[nm]]()
  res$causal[[nm]] <- list(ipw_ate = ipw_block(des, "ATE"),
                           ipw_att = ipw_block(des, "ATT"),
                           aipw = aipw_block(des),
                           tmle = tmle_block(des),
                           dml = dml_block(des),
                           surv = surv_block(des),
                           glm = glm_block(des))
}
options(survey.lonely.psu = "adjust")
des <- lonely_design()
res$causal$lonely_adjust <- list(ipw_ate = ipw_block(des, "ATE"),
                                 aipw = aipw_block(des), glm = glm_block(des))
options(survey.lonely.psu = "fail")

res$provenance <- list(
  r_version = R.version.string,
  survey_version = as.character(packageVersion("survey")),
  numDeriv_version = as.character(packageVersion("numDeriv")),
  data = "tests/reference_parity/_fixtures/survey_causal_data.csv",
  generated_by = "tests/reference_parity/_generate_survey_causal_R.R")

writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_causal_R.json")
cat("wrote survey_causal_R.json\n")
