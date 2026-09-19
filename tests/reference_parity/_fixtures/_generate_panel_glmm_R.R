#!/usr/bin/env Rscript
# R reference values for tests/reference_parity/test_panel_glmm_parity.py.
#
# Reads panel_glmm_data.csv (written by _generate_panel_glmm_data.py) and
# writes panel_glmm_R.json.
#
# References and the convention each one pins
# -------------------------------------------
# * lme4::glmer, Poisson-log / binomial-logit, nAGQ = 1 (Laplace) and
#   nAGQ = 7 (mode-curvature adaptive Gauss-Hermite).  Canonical links, so
#   lme4's PIRLS (Fisher) weights equal the observed curvature.  lme4's
#   inner PIRLS tolerance is tightened (tolPwrss = 1e-13): at the default
#   1e-7 the Laplace deviance is noisy enough to leave the optimum 2.7e-4
#   (relative) away on this sample.  lme4 reports the nAGQ > 1 logLik
#   without the -log(y!) constant, so logLik is only compared for nAGQ = 1.
# * glmmTMB, NB-2 (nbinom2) and Gamma(log): the Laplace approximation with
#   the exact (automatic-differentiation) Hessian of the joint
#   log-likelihood -- the observed curvature -- and vcov from the AD
#   Hessian of the Laplace objective.  betadisp is log(1/alpha) for
#   nbinom2 and log(1/phi) for Gamma; theta is log sd(_cons).
# * lme4::glmer.nb: NB-2 with lme4's Fisher-weight Laplace, the
#   curvature = "expected" counterpart.
# * ordinal::clmm, cumulative logit, nAGQ = 1 and nAGQ = 7.
# * lme4::lmer(REML = FALSE): the Gaussian meglm is the ML linear mixed
#   model; anova() on ML fits for the likelihood-ratio tests.
# * performance::icc and psych::ICC for the intraclass correlation.
# * pglm::pglm(family = negbin, model = "within" / "random"): the
#   Hausman-Hall-Griliches conditional-FE / beta-RE negative binomial
#   (Stata xtnbreg, fe / re).
# * phtt::Eup (archived from CRAN; installed from the CRAN archive,
#   phtt_3.1.2.tar.gz): Bai's interactive-fixed-effects estimator.
#
# * gmm::gmm: exponential-mean IV moment conditions (two-step, iterated).
#
# Requires: lme4, glmmTMB, TMB, ordinal, performance, psych, pglm, phtt,
#           gmm, jsonlite
# Run:      Rscript _generate_panel_glmm_R.R   (from this directory)

suppressMessages({
  library(TMB)
  library(lme4)
  library(glmmTMB)
  library(ordinal)
  library(pglm)  # attaches maxLik, which pglm calls unqualified
  library(jsonlite)
})

here <- tryCatch({
  a <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f[1]))) else getwd()
}, error = function(e) getwd())

d <- read.csv(file.path(here, "panel_glmm_data.csv"))
d$gid <- factor(d$gid)
d$y_ordf <- factor(d$y_ord, ordered = TRUE)

ctl <- glmerControl(optimizer = "bobyqa", tolPwrss = 1e-13,
                    optCtrl = list(maxfun = 2e5, rhobeg = 2e-3, rhoend = 1e-12))

named <- function(v) as.list(setNames(unname(v), names(v)))

lme4_row <- function(m) {
  b <- fixef(m)
  se <- sqrt(diag(as.matrix(vcov(m))))
  names(se) <- names(b)
  list(b = named(b), se = named(se), ll = as.numeric(logLik(m)),
       var_cons = unname(as.numeric(VarCorr(m)$gid)[1]))
}

# glmmTMB's nlminb stops with an AD gradient of ~1e-4 on these fits
# (rel.tol does not move it), i.e. ~2e-6 from the optimum.  The fit is
# finished with Newton steps on TMB's own AD gradient and Hessian of the
# Laplace objective (gradient 1e-12 afterwards); logLik and the standard
# errors (sdreport, i.e. the inverse AD Hessian) are re-evaluated at the
# finished parameters.  Parameter order: beta (3), betadisp, theta.
tmb_row <- function(m) {
  obj <- m$obj
  par <- m$fit$par
  for (i in 1:6) {
    g <- as.numeric(obj$gr(par))
    H <- optimHess(par, obj$fn, obj$gr)
    par <- par - solve(H, g)
  }
  sdr <- TMB::sdreport(obj, par.fixed = par)
  b <- par[names(par) == "beta"]
  names(b) <- names(fixef(m)$cond)
  se <- sqrt(diag(sdr$cov.fixed))[names(par) == "beta"]
  names(se) <- names(b)
  list(b = named(b), se = named(se), ll = -obj$fn(par),
       max_abs_gradient = max(abs(as.numeric(obj$gr(par)))),
       betadisp = unname(par[names(par) == "betadisp"]),
       theta = unname(par[names(par) == "theta"]),
       var_cons = exp(2 * unname(par[names(par) == "theta"])),
       nlminb_b = named(fixef(m)$cond),
       nlminb_ll = as.numeric(logLik(m)))
}

out <- list()

m <- glmer(y_pois ~ x1 + x2 + offset(lexpo) + (1 | gid), d,
           family = poisson, control = ctl)
out$pois_laplace <- lme4_row(m)
m <- glmer(y_pois ~ x1 + x2 + offset(lexpo) + (1 | gid), d,
           family = poisson, nAGQ = 7L, control = ctl)
out$pois_agq7 <- lme4_row(m)

m <- glmer(cbind(y_succ, n_trials - y_succ) ~ x1 + x2 + (1 | gid), d,
           family = binomial, control = ctl)
out$binom_laplace <- lme4_row(m)

m <- glmer(y_bin ~ x1 + x2 + (1 | gid), d, family = binomial, control = ctl)
out$bin_laplace <- lme4_row(m)

m <- glmmTMB(y_nb ~ x1 + x2 + (1 | gid), d, family = nbinom2)
out$nb_laplace_tmb <- tmb_row(m)

m <- glmer.nb(y_nb ~ x1 + x2 + (1 | gid), d, control = ctl,
              nb.control = list(tol = 1e-12))
out$nb_laplace_expected_lme4 <- c(lme4_row(m),
                                  list(theta = unname(getME(m, "glmer.nb.theta"))))

m <- glmmTMB(y_gam ~ x1 + x2 + (1 | gid), d, family = Gamma(link = "log"))
out$gamma_laplace_tmb <- tmb_row(m)

for (k in c(1L, 7L)) {
  m <- clmm(y_ordf ~ x1 + x2 + (1 | gid), data = d, link = "logit", nAGQ = k,
            control = clmm.control(gradTol = 1e-10, maxIter = 1000,
                                   method = "nlminb"))
  cf <- coef(m)
  se <- sqrt(diag(vcov(m)))
  keep <- setdiff(names(cf), "ST1")
  out[[paste0("ologit_clmm_agq", k)]] <- list(
    b = named(cf[keep]), se = named(se[keep]), ll = as.numeric(logLik(m)),
    var_cons = unname(as.numeric(VarCorr(m)$gid)[1]),
    max_grad = max(abs(m$gradient)))
}

m <- lmer(y_gau ~ x1 + x2 + (1 | gid), d, REML = FALSE,
          control = lmerControl(optimizer = "bobyqa",
                                optCtrl = list(rhoend = 1e-12, maxfun = 2e5)))
out$gauss_lmer_ml <- c(lme4_row(m), list(sigma2 = sigma(m)^2))

# ---- likelihood-ratio tests: anova() on ML fits (naive chi2(df)) -----------
lr_row <- function(m0, m1) {
  a <- anova(m0, m1)
  list(chi2 = a$Chisq[2], df = a$Df[2], p = a[["Pr(>Chisq)"]][2],
       ll0 = as.numeric(logLik(m0)), ll1 = as.numeric(logLik(m1)),
       npar0 = attr(logLik(m0), "df"), npar1 = attr(logLik(m1), "df"))
}
lctl <- lmerControl(optimizer = "bobyqa", optCtrl = list(rhoend = 1e-12, maxfun = 2e5))
m_full <- lmer(y_gau ~ x1 + x2 + (1 | gid), d, REML = FALSE, control = lctl)
m_x1 <- lmer(y_gau ~ x1 + (1 | gid), d, REML = FALSE, control = lctl)
out$lr_fixed <- lr_row(m_x1, m_full)
r_int <- lmer(y_rs ~ x1 + x2 + (1 | gid), d, REML = FALSE, control = lctl)
r_un <- lmer(y_rs ~ x1 + x2 + (1 + x1 | gid), d, REML = FALSE, control = lctl)
r_ind <- lmer(y_rs ~ x1 + x2 + (1 + x1 || gid), d, REML = FALSE, control = lctl)
out$lr_slope_un <- lr_row(r_int, r_un)
out$lr_slope_ind <- lr_row(r_int, r_ind)
out$aic_mixed_ml <- AIC(m_full)
out$bic_mixed_ml <- BIC(m_full)

# ---- ICC -----------------------------------------------------------------------
# performance::icc "adjusted" ICC = var(_cons) / (var(_cons) + residual
# variance); for a logit GLMM the residual variance is pi^2/3.
m_reml <- lmer(y_gau ~ x1 + x2 + (1 | gid), d, REML = TRUE, control = lctl)
m_bin <- glmer(y_bin ~ x1 + x2 + (1 | gid), d, family = binomial, control = ctl)
out$icc_performance <- list(
  mixed_ml = performance::icc(m_full)$ICC_adjusted,
  mixed_reml = performance::icc(m_reml)$ICC_adjusted,
  melogit = performance::icc(m_bin)$ICC_adjusted
)
# Balanced one-way layout (first 4 rows of every group, intercept only): the
# REML ICC equals the ANOVA ICC(1) (MSB - MSW) / (MSB + (k - 1) MSW)
# whenever that is positive -- psych::ICC reports it as "ICC1".
bal <- do.call(rbind, lapply(split(d, d$gid), function(g) g[1:4, ]))
wide <- matrix(bal$y_gau, ncol = 4, byrow = TRUE)
ic <- psych::ICC(wide, lmer = FALSE)
m_bal <- lmer(y_gau ~ 1 + (1 | gid), bal, REML = TRUE, control = lctl)
vc_bal <- as.data.frame(VarCorr(m_bal))
out$icc_balanced <- list(
  psych_icc1 = ic$results$ICC[ic$results$type == "ICC1"],
  lmer_reml = vc_bal$vcov[1] / sum(vc_bal$vcov),
  n_groups = nrow(wide), k = 4L
)

# ---- xtnbreg: pglm (Hausman-Hall-Griliches), panel_count_data.csv ------------
pc <- read.csv(file.path(here, "panel_count_data.csv"))
for (mod in c("within", "random")) {
  m <- suppressWarnings(pglm::pglm(y ~ z1 + z2, data = pc, index = c("pid", "year"),
                                   family = pglm::negbin, model = mod,
                                   method = "nr", iterlim = 1000,
                                   tol = 1e-14, reltol = 1e-16, gradtol = 1e-12))
  cf <- coef(m)
  out[[paste0("xtnbreg_pglm_", mod)]] <- list(
    b = named(cf), se = named(setNames(sqrt(diag(vcov(m))), names(cf))),
    ll = as.numeric(logLik(m)))
}

# ---- interactive fixed effects: phtt::Eup, panel_ife_data.csv ----------------
# additive.effects = "none", no intercept, r = 2 factors.  phtt stores the
# panel as a T x N matrix (one column per unit).  Its error.type = 1 SE is
# Bai's sig2.hat * (Z'Z)^{-1} with Z = M_F X M_A (read unrounded from
# Eup.inference; summary() rounds to 3 digits).  sig2.hat is
# sum(diag(var(residuals))) * (T - 1) / df: the residuals are demeaned
# unit by unit before squaring even though the model has no unit effect.
ife <- read.csv(file.path(here, "panel_ife_data.csv"))
ife <- ife[order(ife$unit, ife$period), ]
Nn <- length(unique(ife$unit)); Tt <- length(unique(ife$period))
mk <- function(v) matrix(ife[[v]], nrow = Tt, ncol = Nn)
Yi <- mk("y"); XA <- mk("xa"); XB <- mk("xb")
e <- phtt::Eup(Yi ~ -1 + XA + XB, additive.effects = "none", factor.dim = 2,
               convergence = 1e-12, max.iteration = 10000)
inf1 <- phtt:::Eup.inference(Eup.Obj = e, error.type = 1, kernel.weights = NULL)$inf.result
out$ife_phtt <- list(
  b = list(xa = unname(e$slope.para[1]), xb = unname(e$slope.para[2])),
  se_type1 = list(xa = unname(inf1[1, 2]), xb = unname(inf1[2, 2])),
  sig2_hat = e$sig2.hat, rss = sum(e$residuals^2),
  degrees_of_freedom = e$degrees.of.freedom, iterations = e$Nbr.iteration)

# ---- gmm::gmm, exponential-mean IV moments (gmm_nl_data.csv) ----------------
# E[z (y exp(-b1 x1 - b0) - 1)] = 0, z = (z1, z2, z3, 1).  twoStep's first
# step uses the identity weight; vcov = "MDS"; centeredVcov both ways.
gd <- as.matrix(read.csv(file.path(here, "gmm_nl_data.csv")))
g_nl <- function(theta, x) {
  u <- x[, "y"] * exp(-(theta[1] * x[, "x1"] + theta[2])) - 1
  cbind(x[, "z1"], x[, "z2"], x[, "z3"], 1) * u
}
GCTRL <- list(reltol = 1e-15, abstol = 1e-15, maxit = 100000)
gpack <- function(fit) {
  s <- summary(fit)
  co <- s$coefficients
  list(coef = list(x1 = unname(co[1, 1]), `_cons` = unname(co[2, 1])),
       se = list(x1 = unname(co[1, 2]), `_cons` = unname(co[2, 2])),
       J = as.numeric(s$stest$test[1]))
}
for (cv in c(TRUE, FALSE)) {
  tag <- if (cv) "centered" else "uncentered"
  f2 <- gmm::gmm(g_nl, gd, t0 = c(0, 0), type = "twoStep", wmatrix = "optimal",
                 vcov = "MDS", centeredVcov = cv, control = GCTRL)
  out[[paste0("gmm_nl_twostep_", tag)]] <- gpack(f2)
  fi <- gmm::gmm(g_nl, gd, t0 = c(0, 0), type = "iterative", wmatrix = "optimal",
                 vcov = "MDS", centeredVcov = cv, crit = 1e-13, itermax = 1000,
                 control = GCTRL)
  out[[paste0("gmm_nl_iterative_", tag)]] <- gpack(fi)
}

out$versions <- list(
  R = R.version.string,
  lme4 = as.character(packageVersion("lme4")),
  glmmTMB = as.character(packageVersion("glmmTMB")),
  TMB = as.character(packageVersion("TMB")),
  ordinal = as.character(packageVersion("ordinal")),
  performance = as.character(packageVersion("performance")),
  psych = as.character(packageVersion("psych")),
  pglm = as.character(packageVersion("pglm")),
  phtt = as.character(packageVersion("phtt")),
  gmm = as.character(packageVersion("gmm"))
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = 17, pretty = TRUE),
           file.path(here, "panel_glmm_R.json"))
