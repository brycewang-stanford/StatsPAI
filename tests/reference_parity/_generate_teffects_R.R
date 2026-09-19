#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_teffects_R_parity.py
#
# Reads the CSVs written by _fixtures/_generate_teffects_data.py and writes
# _fixtures/teffects_R.json (17 significant digits) with package versions.
#
#   Rscript tests/reference_parity/_generate_teffects_R.R
#
# Requires R 4.5 and CRAN: AIPW, SuperLearner, tmle, ipw, geepack, sandwich,
# ltmle, DTRreg, geex, survival, AER, jsonlite; GitHub: CMAverse
# (BS1125/CMAverse).
#
# Conventions pinned here (each decides a number below)
# -----------------------------------------------------
# * AIPW (0.6.9.3): stratified_fit() with k_split = 1 (no sample splitting;
#   per-arm outcome fits, the arm indicator is constant within each fit).
#   AIPW's stratified_fit() omits `Q.model = FALSE` when fitting the
#   propensity, so SL.glm would be fitted with the *outcome's* gaussian
#   family (a linear probability model); the propensity learner is therefore
#   SL.glm forced to binomial ("SL.glm.logit" below). g.bound = 0.01, inert
#   on this fixture (propensities in [0.08, 0.94]). SE = sd(EIF)/sqrt(n).
#   AIPW's ATT divides the control term by P(A = 0) instead of P(A = 1);
#   the test reconstructs this from its own quantities.
# * tmle (2.1.1): cvQinit = FALSE (by default the initial Q is 10-fold
#   cross-validated), SL.glm for Q and g, gbound = 0.025, default
#   alpha = 0.9995, i.e. initial Q truncated to [5e-4, 1 - 5e-4] on the
#   [0, 1] scale.
# * ipw::ipwtm (1.3.0): type = "all", logit link, trunc = 0.01 (quantile
#   type 7). Gaussian family is geepack::geeglm with corstr =
#   "independence"; its dispersion is RSS / N (ML).
# * MSM outcome fit: lm / glm(quasibinomial) with the truncated weights,
#   sandwich::vcovCL(cluster = ~id, type = "HC1") (G/(G-1) (N-1)/(N-k)).
#   The glm is run to epsilon = 1e-14: vcovCL's bread uses the IRLS working
#   weights of the last iteration, evaluated at the *previous* coefficient
#   vector, so at glm's default 1e-8 the SEs sit ~1e-7 from the converged
#   sandwich (the coefficients themselves agree to 1e-12 either way).
# * ltmle (1.3-0): SL.library = NULL (glm), default gbounds c(0.01, 1),
#   default Qform / gform (all parents), variance.method = "ic".
# * DTRreg (2.4): method = "gest", treat.type = "bin", weight = "none"
#   (treat.fam is ignored for binary treatments: the propensity is logit).
# * CMAverse (0.1.0): cmest(model = "rb", estimation = "paramfunc",
#   inference = "delta") for the four-way decomposition (basec fixed at its
#   mean); cmest(model = "gformula", postc = "l") for interventional
#   analogues (rpnde, rpnie).
# * ICE g-formula: sequential lm() by hand, sandwich SE from
#   geex::m_estimate (1.1.1) on the stacked estimating equations.
# * IPCW: survival::coxph(Surv(time, 1 - event), ties = "breslow"),
#   basehaz(centered = FALSE), survival evaluated just before each time.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(AIPW); library(SuperLearner); library(tmle); library(ipw)
  library(geepack); library(sandwich); library(ltmle); library(DTRreg)
  library(geex); library(survival); library(AER); library(CMAverse)
  library(jsonlite)
})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
HERE <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
FIX <- file.path(HERE, "_fixtures")

cs <- read.csv(file.path(FIX, "teffects_cs.csv"))
wide <- read.csv(file.path(FIX, "teffects_wide.csv"))
long <- read.csv(file.path(FIX, "teffects_long.csv"))
surv <- read.csv(file.path(FIX, "teffects_surv.csv"))
out <- list()

# ---- AIPW -------------------------------------------------------------------
SL.glm.logit <- function(Y, X, newX, family, obsWeights, ...) {
  SuperLearner::SL.glm(Y, X, newX, family = binomial(), obsWeights = obsWeights, ...)
}
set.seed(1)
a <- AIPW$new(Y = cs$y, A = cs$d, W = cs[, c("x1", "x2", "x3")],
              Q.SL.library = "SL.glm", g.SL.library = "SL.glm.logit",
              k_split = 1, verbose = FALSE)
suppressWarnings(a$stratified_fit())
a$summary(g.bound = 0.01)
res <- a$result
out$aipw <- list(
  ate = res["Mean Difference", "Estimate"], ate_se = res["Mean Difference", "SE"],
  po1 = res["Mean of Exposure", "Estimate"], po1_se = res["Mean of Exposure", "SE"],
  po0 = res["Mean of Control", "Estimate"], po0_se = res["Mean of Control", "SE"],
  att = res["ATT Mean Difference", "Estimate"],
  att_se = res["ATT Mean Difference", "SE"],
  p_score_range = range(a$obs_est$raw_p_score))

# ---- TMLE (tmle::tmle, glm nuisances) ----------------------------------------
W <- cs[, c("x1", "x2", "x3")]
tm <- function(Y, family) {
  set.seed(1)
  f <- tmle(Y = Y, A = cs$d, W = W, family = family, Q.SL.library = "SL.glm",
            g.SL.library = "SL.glm", gbound = 0.025, cvQinit = FALSE)
  list(psi = f$estimates$ATE$psi, se = sqrt(f$estimates$ATE$var.psi),
       att = f$estimates$ATT$psi, att_se = sqrt(f$estimates$ATT$var.psi),
       epsilon = unname(f$epsilon))
}
out$tmle_binary <- tm(cs$yb, "binomial")
out$tmle_gaussian <- tm(cs$y, "gaussian")

# ---- MSM / stabilized weights (ipw::ipwtm) -----------------------------------
long <- long[order(long$id, long$t), ]
lag0 <- function(x) c(0, head(x, -1))
long$a_lag <- ave(long$a, long$id, FUN = lag0)
long$ac_lag <- ave(long$ac, long$id, FUN = lag0)
long$cum_a <- ave(long$a, long$id, FUN = cumsum)
long$ever_a <- ave(long$a, long$id, FUN = function(x) as.numeric(cummax(x) > 0))
wb <- ipwtm(exposure = a, family = "binomial", link = "logit",
            numerator = ~ a_lag + v, denominator = ~ a_lag + v + l,
            id = id, timevar = t, type = "all", data = long, trunc = 0.01)
wc <- ipwtm(exposure = ac, family = "gaussian",
            numerator = ~ ac_lag + v, denominator = ~ ac_lag + v + l,
            id = id, timevar = t, type = "all", data = long,
            corstr = "independence")
long$w <- wb$weights.trunc
msm_fit <- function(expo, family) {
  f <- as.formula(paste(if (family == "gaussian") "y" else "yb", "~", expo, "+ v"))
  m <- if (family == "gaussian") lm(f, data = long, weights = w)
       else suppressWarnings(glm(f, data = long, weights = w, family = quasibinomial(),
                                 control = glm.control(epsilon = 1e-14, maxit = 100)))
  list(coef = unname(coef(m)),
       se = unname(sqrt(diag(vcovCL(m, cluster = ~id, type = "HC1")))))
}
out$msm <- list(
  sw_binary = wb$ipw.weights, sw_binary_trunc = wb$weights.trunc,
  sw_gaussian = wc$ipw.weights,
  cumulative = msm_fit("cum_a", "gaussian"),
  ever = msm_fit("ever_a", "gaussian"),
  current = msm_fit("a", "gaussian"),
  cumulative_binomial = msm_fit("cum_a", "binomial"))

# ---- LTMLE --------------------------------------------------------------------
lt <- function(ycol) {
  d <- wide[, c("v", "L0", "A0", "L1", "A1", ycol)]
  r <- suppressMessages(ltmle(d, Anodes = c("A0", "A1"), Lnodes = "L1",
                              Ynodes = ycol, abar = list(c(1, 1), c(0, 0)),
                              SL.library = NULL, estimate.time = FALSE,
                              variance.method = "ic"))
  s <- summary(r)
  list(psi1 = s$effect.measures$treatment$estimate,
       psi0 = s$effect.measures$control$estimate,
       ate = s$effect.measures$ATE$estimate,
       ate_se = s$effect.measures$ATE$std.dev)
}
out$ltmle_binary <- lt("Yb")
out$ltmle_gaussian <- lt("Y")
widec <- read.csv(file.path(FIX, "teffects_wide_cens.csv"))
ltc <- function(ycol) {
  d <- widec[, c("v", "L0", "A0", "C0", "L1", "A1", "C1", ycol)]
  d$C0 <- BinaryToCensoring(is.uncensored = d$C0)
  d$C1 <- BinaryToCensoring(is.uncensored = d$C1)
  r <- suppressMessages(ltmle(d, Anodes = c("A0", "A1"), Cnodes = c("C0", "C1"),
                              Lnodes = "L1", Ynodes = ycol,
                              abar = list(c(1, 1), c(0, 0)), SL.library = NULL,
                              estimate.time = FALSE, variance.method = "ic"))
  s <- summary(r)
  list(psi1 = s$effect.measures$treatment$estimate,
       psi0 = s$effect.measures$control$estimate,
       ate = s$effect.measures$ATE$estimate,
       ate_se = s$effect.measures$ATE$std.dev)
}
out$ltmle_cens_binary <- ltc("Yb")
out$ltmle_cens_gaussian <- ltc("Y")

# ---- g-estimation (DTRreg) ----------------------------------------------------
dg <- function(tm) {
  f <- suppressMessages(DTRreg(outcome = wide$Y, blip.mod = list(~1, ~1),
                               treat.mod = tm,
                               tf.mod = list(~L0 + v, ~L0 + v + A0 + L1),
                               data = wide, method = "gest",
                               treat.type = "bin", weight = "none"))
  unname(unlist(f$psi))
}
out$g_estimation <- list(
  psi_logit = dg(list(A0 ~ L0 + v, A1 ~ L0 + v + A0 + L1)),
  psi_logit_pcov = dg(list(A0 ~ L0, A1 ~ L1 + A0)))

# ---- Mediation (CMAverse) -------------------------------------------------------
fw <- cmest(data = cs[, c("ym", "d", "m", "x1")], model = "rb", outcome = "ym",
            exposure = "d", mediator = "m", basec = "x1", EMint = TRUE,
            mreg = list("linear"), yreg = "linear", astar = 0, a = 1,
            mval = list(0), estimation = "paramfunc", inference = "delta")
out$four_way <- list(
  te = unname(fw$effect.pe["te"]), cde = unname(fw$effect.pe["cde"]),
  intref = unname(fw$effect.pe["intref"]), intmed = unname(fw$effect.pe["intmed"]),
  pie = unname(fw$effect.pe["pnie"]),
  se_te = unname(fw$effect.se["te"]), se_cde = unname(fw$effect.se["cde"]),
  se_intref = unname(fw$effect.se["intref"]),
  se_intmed = unname(fw$effect.se["intmed"]), se_pie = unname(fw$effect.se["pnie"]))
dmi <- cs[, c("yl", "d", "m2", "x1", "l")]
set.seed(1)
gi <- suppressMessages(cmest(data = dmi, model = "gformula", outcome = "yl",
            exposure = "d", mediator = "m2", basec = "x1", postc = "l",
            EMint = FALSE, mreg = list("linear"), yreg = "linear",
            postcreg = list("linear"), astar = 0, a = 1, mval = list(0),
            estimation = "imputation", inference = "bootstrap", nboot = 2))
rb <- cmest(data = dmi[, c("yl", "d", "m2", "x1")], model = "rb", outcome = "yl",
            exposure = "d", mediator = "m2", basec = "x1", EMint = FALSE,
            mreg = list("linear"), yreg = "linear", astar = 0, a = 1,
            mval = list(0), estimation = "paramfunc", inference = "delta")
out$interventional <- list(
  tv_ide = unname(gi$effect.pe["rpnde"]), tv_iie = unname(gi$effect.pe["rpnie"]),
  tv_te = unname(gi$effect.pe["te"]),
  notv_ide = unname(rb$effect.pe["pnde"]), notv_iie = unname(rb$effect.pe["pnie"]),
  notv_te = unname(rb$effect.pe["te"]))

# ---- ICE g-formula (hand lm + geex sandwich) -----------------------------------
f1 <- lm(Y ~ v + L0 + A0 + L1 + A1, data = wide)
nd1 <- wide; nd1$A1 <- 1
wide$q1 <- predict(f1, newdata = nd1)
f0 <- lm(q1 ~ v + L0 + A0, data = wide)
nd0 <- wide; nd0$A0 <- 1
psi_ice <- mean(predict(f0, newdata = nd0))
ef <- function(data) {
  X1 <- cbind(1, data$v, data$L0, data$A0, data$L1, data$A1)
  X1s <- cbind(1, data$v, data$L0, data$A0, data$L1, 1)
  X0 <- cbind(1, data$v, data$L0, data$A0)
  X0s <- cbind(1, data$v, data$L0, 1)
  Y <- data$Y
  function(theta) {
    b1 <- theta[1:6]; b0 <- theta[7:10]; m <- theta[11]
    c(X1 * (Y - sum(X1 * b1)), X0 * (sum(X1s * b1) - sum(X0 * b0)),
      sum(X0s * b0) - m)
  }
}
gx <- m_estimate(estFUN = ef, data = wide, units = "id",
                 root_control = setup_root_control(start = c(coef(f1), coef(f0), psi_ice)))
out$ice <- list(psi = psi_ice, psi_geex = unname(coef(gx)[11]),
                se = sqrt(vcov(gx)[11, 11]))

# ---- IPCW (Cox, Breslow) ---------------------------------------------------------
sc <- function(formula) {
  fit <- coxph(formula, data = surv, ties = "breslow")
  bh <- basehaz(fit, centered = FALSE)
  lp <- if (length(coef(fit))) as.vector(model.matrix(fit) %*% coef(fit)) else rep(0, nrow(surv))
  Hm <- sapply(surv$time, function(t) { k <- bh$time < t; if (any(k)) max(bh$hazard[k]) else 0 })
  list(S = exp(-Hm * exp(lp)), coef = unname(coef(fit)))
}
den <- sc(Surv(time, 1 - event) ~ z1 + z2)
num <- sc(Surv(time, 1 - event) ~ 1)
out$ipcw <- list(
  coef = den$coef,
  w_unstab = ifelse(surv$event == 1, 1 / den$S, 0),
  w_stab = ifelse(surv$event == 1, num$S / den$S, 0))

# ---- Principal strata: Wald LATE -------------------------------------------------
iv <- ivreg(y ~ s | d, data = cs)
out$principal_strat <- list(late = unname(coef(iv)["s"]))

out$versions <- list(
  R = R.version.string,
  AIPW = as.character(packageVersion("AIPW")),
  SuperLearner = as.character(packageVersion("SuperLearner")),
  tmle = as.character(packageVersion("tmle")),
  ipw = as.character(packageVersion("ipw")),
  geepack = as.character(packageVersion("geepack")),
  sandwich = as.character(packageVersion("sandwich")),
  ltmle = as.character(packageVersion("ltmle")),
  DTRreg = as.character(packageVersion("DTRreg")),
  CMAverse = as.character(packageVersion("CMAverse")),
  geex = as.character(packageVersion("geex")),
  survival = as.character(packageVersion("survival")),
  AER = as.character(packageVersion("AER")))

writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null"),
           file.path(FIX, "teffects_R.json"))
cat("wrote", file.path(FIX, "teffects_R.json"), "\n")
