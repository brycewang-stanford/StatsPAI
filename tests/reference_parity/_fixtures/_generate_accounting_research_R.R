# Reference values for tests/reference_parity/test_accounting_research_parity.py
# Run from this folder:  Rscript _generate_accounting_research_R.R
# Reads accounting_research_{panel,cross}.csv, writes accounting_research_R.json.
suppressMessages({
  library(plm); library(sandwich); library(lmtest); library(fixest)
  library(robustbase); library(MASS); library(jsonlite)
})
out <- list()
tab <- function(b, V) list(names = names(b), coef = unname(b), se = unname(sqrt(diag(V))))

# ---------------------------------------------------------------- panel
p <- read.csv("accounting_research_panel.csv")
pg <- p[p$gap == 0, ]
fm <- lm(y ~ x + z, data = p)
out$ols <- tab(coef(fm), vcov(fm))
out$hc1 <- tab(coef(fm), vcovHC(fm, type = "HC1"))
pl <- plm(y ~ x + z, p, index = c("firm", "year"), model = "pooling")
for (L in c(1, 3)) out[[paste0("panel_nw_", L)]] <- tab(coef(pl), vcovNW(pl, maxlag = L))
for (d in list(list("fm", p), list("fm_gap", pg))) {
  m <- pmg(y ~ x + z, d[[2]], index = "year")
  out[[d[[1]]]] <- c(tab(coef(m), vcov(m)), list(vcov = unname(vcov(m))))
  per <- do.call(rbind, lapply(split(d[[2]], d[[2]]$year), function(s) coef(lm(y ~ x + z, s))))
  out[[paste0(d[[1]], "_r2")]] <- mean(sapply(split(d[[2]], d[[2]]$year), function(s) summary(lm(y ~ x + z, s))$r.squared))
  for (L in c(1, 3)) {
    se <- sapply(colnames(per), function(v) {
      f <- lm(per[, v] ~ 1)
      sqrt(diag(NeweyWest(f, lag = L, prewhite = FALSE, adjust = TRUE)))
    })
    out[[paste0(d[[1]], "_nw_", L)]] <- list(names = colnames(per), se = unname(se))
  }
}
fx <- function(m) list(names = names(coef(m)), coef = unname(coef(m)), se = unname(se(m)))
out$cl_firm <- fx(feols(y ~ x + z, vcov = ~firm, data = p))
out$cl_year <- fx(feols(y ~ x + z, vcov = ~year, data = p))
out$cl_two <- fx(feols(y ~ x + z, vcov = ~firm + year, data = p))
out$fe_two <- fx(feols(y ~ x + z | firm + year, vcov = ~firm + year, data = p))

# ------------------------------------------------------- cross-section
d <- read.csv("accounting_research_cross.csv")
f3 <- y_out ~ x1 + x2 + x3
tight <- function(psi) lmrob.control(tuning.psi = psi, rel.tol = 1e-14, refine.tol = 1e-14,
                                     solve.tol = 1e-14, scale.tol = 1e-14, max.it = 5000,
                                     k.max = 5000, maxit.scale = 5000)
set.seed(1)
for (psi in c(3.4437, 4.685061)) {
  m <- lmrob(f3, data = d, method = "MM", control = tight(psi))
  out[[paste0("lmrob_", psi)]] <- c(tab(coef(m), vcov(m)), list(
    scale = m$scale, s_coef = unname(m$init.S$coefficients),
    weights = unname(weights(m, type = "robustness"))))
}
for (k in list(list("rlm_huber", psi.huber), list("rlm_bisquare", psi.bisquare))) {
  m <- rlm(f3, data = d, psi = k[[2]], acc = 1e-13, maxit = 1000)
  out[[k[[1]]]] <- c(tab(coef(m), vcov(m)), list(scale = m$s))
}

# factor() spelling, through the origin and with an interaction
m <- lm(y ~ x1 + factor(ind), data = d)
out$lm_factor <- tab(coef(m), vcov(m))
m <- lm(y ~ factor(ind) * x1, data = d)
out$lm_factor_int <- tab(coef(m), vcov(m))

# Poisson with a regressor that picks out only zeros
m <- glm(fines ~ x1 + x2 + rare + factor(ind), family = poisson, data = d,
         control = glm.control(maxit = 200, epsilon = 1e-13))
keep <- c("(Intercept)", "x1", "x2", "factor(ind)2", "factor(ind)3", "factor(ind)4", "factor(ind)5")
out$poisson_sep <- list(names = keep, coef = unname(coef(m)[keep]),
                        se = unname(sqrt(diag(vcov(m)))[keep]),
                        se_hc1 = unname(sqrt(diag(vcovHC(m, type = "HC1")))[keep]),
                        rare = unname(coef(m)["rare"]), deviance = deviance(m))

# winsorize / truncate at the type-2 quantile (farr::winsorize, farr::truncate)
q <- quantile(d$tail, c(0.01, 0.99), type = 2, na.rm = TRUE)
w <- d$tail; w[!is.na(w) & w < q[1]] <- q[1]; w[!is.na(w) & w > q[2]] <- q[2]
tr <- d$tail; tr[!is.na(tr) & (tr < q[1] | tr > q[2])] <- NA
q5 <- quantile(d$tail, c(0.05, 0.95), type = 2, na.rm = TRUE)
out$winsor <- list(cuts = unname(q), cuts5 = unname(q5), w = w, tr = tr)

# NDCG at k (farr::ndcg) and the rank AUC (farr::auc)
ndcg <- function(score, resp, k) {
  rk <- sort(score, index.return = TRUE, decreasing = TRUE)$ix
  kn <- round(length(resp) * k); kz <- min(kn, sum(resp))
  z <- sum(c(rep(1, kz), rep(0, kn - kz)) / log(1:kn + 1, 2))
  sum((resp[rk][1:kn] == 1) / log(1:kn + 1, 2)) / z
}
n_neg <- sum(!d$event); n_pos <- sum(d$event)
U <- sum(rank(d$score)[!d$event]) - n_neg * (n_neg + 1) / 2
out$ranking <- list(ndcg_01 = ndcg(d$score, d$event, 0.01), ndcg_05 = ndcg(d$score, d$event, 0.05),
                    ndcg_20 = ndcg(d$score, d$event, 0.2), auc = 1 - U / n_neg / n_pos)

# ITCV as computed in the book (Frank 2000), classical and HC1 t statistics
m <- lm(y ~ x1 + x2 + x3 + factor(ind), data = d)
itcv <- function(tstat, df, alpha) {
  r <- tstat / sqrt(df + tstat^2); cv <- sign(tstat) * qt(1 - alpha / 2, df)
  rc <- cv / sqrt(df + cv^2); c(r_obs = r, r_crit = rc, itcv = (r - rc) / (1 - abs(rc)))
}
t_iid <- coef(summary(m))[, 3]; t_hc1 <- coeftest(m, vcov = vcovHC(m, type = "HC1"))[, 3]
pc <- function(M) { P <- solve(cov(M)); -P[1, -1] / sqrt(P[1, 1] * diag(P)[-1]) }
X <- model.matrix(m)[, -1]
others <- X[, colnames(X) != "x2"]
out$itcv <- list(df = df.residual(m),
                 x2_iid = lapply(c(0.01, 0.05, 0.1), function(a) unname(itcv(t_iid["x2"], df.residual(m), a))),
                 x2_hc1 = unname(itcv(t_hc1["x2"], df.residual(m), 0.05)),
                 x3_iid = unname(itcv(t_iid["x3"], df.residual(m), 0.05)),
                 controls = colnames(others),
                 r_yz = unname(pc(cbind(d$y, others))), r_xz = unname(pc(cbind(d$x2, others))))

out$versions <- list(R = R.version.string,
                     plm = as.character(packageVersion("plm")),
                     sandwich = as.character(packageVersion("sandwich")),
                     fixest = as.character(packageVersion("fixest")),
                     robustbase = as.character(packageVersion("robustbase")),
                     MASS = as.character(packageVersion("MASS")))
write_json(out, "accounting_research_R.json", digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null")
cat("wrote accounting_research_R.json\n")
