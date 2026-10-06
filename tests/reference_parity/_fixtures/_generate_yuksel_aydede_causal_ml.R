# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/
#   test_yuksel_aydede_causal_ml_parity.py
#
# The causal chapters of Yuksel and Aydede, "Causal Inference and Machine
# Learning: In Economics, Social, and Health Sciences"
# (https://www.causalmlbook.com), run with the R packages
# the book uses, on the book's data-generating processes at a smaller sample
# size. Both sides read the CSV files this script writes.
#
# Requires: estimatr, MatchIt, optmatch, sandwich, hdm, glmnet, AER,
#           gsynth (>= 1.4.0, with fect), grf, jsonlite.
# The ivreg package is deliberately not loaded: it and AER both register
# S3 methods for class "ivreg" and sandwich::vcovHC then fails.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_yuksel_aydede_causal_ml.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(estimatr); library(MatchIt); library(sandwich); library(hdm)
  library(glmnet); library(AER); library(gsynth); library(grf)
  library(jsonlite); library(MASS)
})
here <- "tests/reference_parity/_fixtures"
ref <- list(meta = list(R = R.version.string,
  packages = sapply(c("estimatr", "MatchIt", "optmatch", "sandwich", "hdm",
    "glmnet", "AER", "gsynth", "fect", "grf"),
    function(p) as.character(packageVersion(p)))))

# --- 1. selection on observables: the book's college / earnings design -----
set.seed(42)
N <- 600; NT <- 150; NC <- N - NT
d <- data.frame(
  D = c(rep(1, NT), rep(0, NC)),
  income = round(c(rnorm(NT, 95, 20), rnorm(NC, 70, 20)), 3),
  cont = round(c(rnorm(NT, 6, 2), rnorm(NC, 5, 2)), 3),
  bin = c(rbinom(NT, 1, 0.6), rbinom(NC, 1, 0.4)))
d$Y <- round(50 + 3 * d$D + 0.05 * d$income + 0.5 * d$cont + rnorm(N, 0, 10), 3)
write.csv(d, file.path(here, "yuksel_aydede_selection.csv"), row.names = FALSE)
fml <- D ~ income + cont + bin

# Neyman / HC2 / Lin
d$inc_c <- d$income - mean(d$income)
m1 <- lm_robust(Y ~ D, d, se_type = "HC2")
m3 <- lm_robust(Y ~ D * inc_c, d, se_type = "HC2")
ref$rct <- list(
  neyman_se = sqrt(var(d$Y[d$D == 1]) / NT + var(d$Y[d$D == 0]) / NC),
  hc2 = c(coef(m1)[2], m1$std.error[2]), lin = c(coef(m3)[2], m3$std.error[2]))

# subclassification on income quintiles
st <- dplyr::ntile(d$income, 5)
tau_s <- sapply(1:5, function(k) mean(d$Y[st == k & d$D == 1]) - mean(d$Y[st == k & d$D == 0]))
ref$subclass <- list(strata = st, att = sum(tau_s * tapply(d$D, st, sum)) / NT,
                     ate = sum(tau_s * table(st)) / N)

# matching (MatchIt)
att <- function(md) weighted.mean(md$Y[md$D == 1], md$weights[md$D == 1]) -
  weighted.mean(md$Y[md$D == 0], md$weights[md$D == 0])
ps <- predict(glm(fml, binomial, d), type = "response")
nn <- matchit(fml, d, method = "nearest")
mh <- matchit(fml, d, method = "nearest", distance = "mahalanobis")
cal <- matchit(fml, d, method = "nearest", caliper = 0.1)
ref$match <- list(ps = ps, nearest = att(match.data(nn)),
  mahalanobis = att(match.data(mh)), caliper = att(match.data(cal)),
  caliper_n = sum(match.data(cal)$D))

# full matching: optmatch at its default tolerance and at a tight one
full <- function(est, tol) {
  fu <- if (is.null(tol)) matchit(fml, d, method = "full", estimand = est)
        else matchit(fml, d, method = "full", estimand = est, tol = tol)
  md <- match.data(fu); md <- md[order(as.integer(rownames(md))), ]
  fit <- lm(Y ~ D, md, weights = weights)
  sc <- as.integer(md$subclass)
  tot <- sum(sapply(unique(sc), function(s) {
    i <- which(sc == s); sum(abs(outer(ps[i[md$D[i] == 1]], ps[i[md$D[i] == 0]], "-"))) }))
  list(estimate = unname(coef(fit)[2]), se = sqrt(vcovCL(fit, cluster = ~subclass)[2, 2]),
       subclass = sc, total_distance = tot, n_sets = length(unique(sc)))
}
ref$full <- list(att = full("ATT", NULL), ate = full("ATE", NULL),
                 att_tight = full("ATT", 1e-9), ate_tight = full("ATE", 1e-9))

# IPW (Hajek) and AIPW as the book writes them
w <- ifelse(d$D == 1, 1 / ps, 1 / (1 - ps))
hajek <- weighted.mean(d$Y[d$D == 1], w[d$D == 1]) - weighted.mean(d$Y[d$D == 0], w[d$D == 0])
y1 <- predict(lm(Y ~ income + cont + bin, d[d$D == 1, ]), d)
y0 <- predict(lm(Y ~ income + cont + bin, d[d$D == 0, ]), d)
psi <- d$D * (d$Y - y1) / ps - (1 - d$D) * (d$Y - y0) / (1 - ps) + y1 - y0
ref$weighting <- list(hajek = hajek, aipw = mean(psi), aipw_se = sd(psi) / sqrt(N),
                      n_outside_01_99 = sum(ps < 0.01 | ps > 0.99), ps_min = min(ps))

# --- 2. double machine learning --------------------------------------------
set.seed(123)
n <- 800; p <- 6
X <- round(mvrnorm(n, rep(0, p), diag(p)), 3)
Z <- round(0.7 * X[, 1] + 0.6 * rnorm(n), 3)
Dm <- rbinom(n, 1, plogis(1.5 * Z + X %*% runif(p, -0.5, 0.5)))
Ym <- round(as.numeric(2 * Dm + X %*% runif(p, -1, 1) + rnorm(n)), 3)
fold <- sample(rep(1:5, length.out = n))
dm <- data.frame(Y = Ym, D = Dm, Z = Z, X, fold = fold)
names(dm)[4:(3 + p)] <- paste0("X", 1:p)
write.csv(dm, file.path(here, "yuksel_aydede_dml.csv"), row.names = FALSE)
Yr <- Dr <- Zr <- rep(NA, n)
for (k in 1:5) {
  tr <- fold != k; va <- fold == k
  Yr[va] <- Ym[va] - predict(rlasso(X[tr, ], Ym[tr]), X[va, ])
  Dr[va] <- Dm[va] - predict(rlasso(X[tr, ], Dm[tr]), X[va, ])
  Zr[va] <- Z[va] - predict(rlasso(X[tr, ], Z[tr]), X[va, ])
}
th <- sum(Yr * Dr) / sum(Dr^2)
thiv <- sum(Zr * Yr) / sum(Zr * Dr)
res <- data.frame(Yr = Yr, Dr = Dr, Zr = Zr)
iv_aer <- AER::ivreg(Yr ~ Dr | Zr, data = res)
ref$dml <- list(
  plr = c(th, sqrt(mean(((Yr - th * Dr) * Dr)^2) / mean(Dr^2)^2 / n)),
  pliv = c(thiv, sqrt(mean(((Yr - thiv * Dr) * Zr)^2) / mean(Dr * Zr)^2 / n)),
  resid = res,
  iv_se = list(
    estimatr_hc2 = iv_robust(Yr ~ Dr | Zr, data = res, se_type = "HC2")$std.error[2],
    estimatr_hc3 = iv_robust(Yr ~ Dr | Zr, data = res, se_type = "HC3")$std.error[2],
    aer_hc3 = sqrt(vcovHC(iv_aer, type = "HC3")[2, 2]),
    hc1 = sqrt(vcovHC(iv_aer, type = "HC1")[2, 2])))
# glmnet at fixed penalties (lasso, ridge)
lam <- c(0.5, 0.1, 0.01)
ref$glmnet <- list(lambda = lam, sd_y = sqrt(mean((Ym - mean(Ym))^2)),
  lasso = as.matrix(coef(glmnet(X, Ym, alpha = 1, lambda = lam, thresh = 1e-15))),
  ridge = as.matrix(coef(glmnet(X, Ym, alpha = 0, lambda = lam, thresh = 1e-15))))

# --- 3. meta-learners with a different learner in each arm ------------------
set.seed(42)
n <- 400
X1 <- round(pmax(pmin(rnorm(n, 2, 1), 4), 0), 3)
X2 <- round(pmax(pmin(rnorm(n, 4, 0.5), 6), 0), 3)
ml <- data.frame(X1, X2)
ml$W <- rbinom(n, 1, plogis(0.5 * X1 - 0.25 * X2))
ml$Y <- round(1 + X1 + X2 + 2 * X1 * ml$W + rnorm(n), 3)
write.csv(ml, file.path(here, "yuksel_aydede_meta.csv"), row.names = FALSE)
e <- predict(glm(W ~ X1 + X2, binomial, ml), type = "response")
m1 <- lm(Y ~ X1 + I(X1^2) + X2, ml[ml$W == 1, ]); m0 <- lm(Y ~ X1 + X2, ml[ml$W == 0, ])
t1 <- ml[ml$W == 1, ]; t1$D1 <- t1$Y - predict(m0, t1)
t0 <- ml[ml$W == 0, ]; t0$D0 <- predict(m1, t0) - t0$Y
tau1 <- predict(lm(D1 ~ X1 + X2, t1), ml); tau0 <- predict(lm(D0 ~ X1 + I(X1^2) + X2, t0), ml)
ref$metalearner <- list(t = predict(m1, ml) - predict(m0, ml), x = e * tau0 + (1 - e) * tau1)

# --- 4. generalized synthetic control (the gsynth example data) -------------
data(gsynth)
gs <- simdata[, c("id", "time", "Y", "D", "X1", "X2")]
ids <- unique(gs$id[gs$D == 1])
gs$D_stag <- gs$D
gs$D_stag[gs$id == ids[1] & gs$time <= 25] <- 0
gs$D_stag[gs$id == ids[2] & gs$time <= 23] <- 0
write.csv(gs, file.path(here, "yuksel_aydede_gsynth.csv"), row.names = FALSE)
gfit <- function(f, r) {
  o <- suppressMessages(gsynth(f, data = gs, index = c("id", "time"), force = "two-way",
    CV = FALSE, r = r, se = FALSE, tol = 1e-12, parallel = FALSE))
  list(att_avg = o$att.avg, beta = as.numeric(o$beta), att = as.numeric(o$att))
}
ref$gsynth <- list(
  cov = lapply(0:3, function(r) gfit(Y ~ D + X1 + X2, r)),
  nocov = lapply(0:3, function(r) gfit(Y ~ D, r)),
  staggered = gfit(Y ~ D_stag + X1 + X2, 2),
  one_unit = (function() {
    one <- gs[!(gs$id %in% ids[-1]), ]
    o <- suppressMessages(gsynth(Y ~ D + X1 + X2, data = one, index = c("id", "time"),
      force = "two-way", CV = FALSE, r = 2, se = FALSE, tol = 1e-12, parallel = FALSE))
    list(unit = ids[1], att_avg = o$att.avg, beta = as.numeric(o$beta)) })())
set.seed(1)
cv <- suppressMessages(gsynth(Y ~ D + X1 + X2, data = gs, index = c("id", "time"),
  force = "two-way", CV = TRUE, r = c(0, 5), se = TRUE, inference = "parametric",
  nboots = 500, parallel = FALSE))
ref$gsynth$cv <- list(r_cv = cv$r.cv, att_avg = cv$att.avg, se = cv$est.avg[2])

# --- 5. grf::average_treatment_effect(subset=) as an operator ---------------
gd <- read.csv(file.path(here, "grf_cluster_operator_data.csv"))
Xg <- as.matrix(gd[, c("x1", "x2", "x3")])
sub <- gd$x1 > 0
ref$grf_subset <- list()
for (key in c("rows", "clusters", "clusters_equalized")) {
  cf <- if (key == "rows") causal_forest(Xg, gd$Y, gd$W, num.trees = 50, seed = 1)
        else causal_forest(Xg, gd$Y, gd$W, clusters = gd$cluster,
          equalize.cluster.weights = key == "clusters_equalized", num.trees = 50, seed = 1)
  cf$Y.hat <- gd$Y_hat; cf$W.hat <- gd$W_hat; cf$predictions <- gd$tau_oob
  ref$grf_subset[[key]] <- lapply(c("all", "treated", "control", "overlap"), function(t) {
    r <- average_treatment_effect(cf, target.sample = t, subset = sub)
    list(target = t, estimate = unname(r[1]), se = unname(r[2])) })
}

write_json(ref, file.path(here, "yuksel_aydede_causal_ml_R.json"),
           digits = NA, auto_unbox = TRUE, pretty = FALSE)
