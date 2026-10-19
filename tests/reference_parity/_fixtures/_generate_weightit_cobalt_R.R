# Reference outputs for sp.weightit / sp.bal_tab
# (tests/reference_parity/test_weightit_cobalt_parity.py).
#
# WeightIt 1.7.0 and cobalt 4.6.2 are GPL: used as black boxes (outputs on
# this fixture only). Run from this directory:
#   Rscript _generate_weightit_cobalt_R.R
suppressMessages({library(WeightIt); library(cobalt); library(jsonlite)})
set.seed(20261019)
n <- 400
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rbinom(n, 1, 0.4)
eta2 <- 0.5 * x1 - 0.3 * x2 + 0.4 * x3; eta3 <- -0.4 * x1 + 0.6 * x2 - 0.2 * x3 + 0.2
p <- cbind(1, exp(eta2), exp(eta3)); p <- p / rowSums(p)
a <- apply(p, 1, function(pr) sample(1:3, 1, prob = pr))
dose <- 1 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3 + rnorm(n)
d <- data.frame(treat = c("a", "b", "c")[a], bin = as.integer(a == 1),
                dose = round(dose, 6), x1 = round(x1, 6), x2 = round(x2, 6), x3 = x3)
write.csv(d, "weightit_cobalt_data.csv", row.names = FALSE)
d <- read.csv("weightit_cobalt_data.csv"); d$treat <- factor(d$treat)

out <- list(`_versions` = list(WeightIt = as.character(packageVersion("WeightIt")),
                               cobalt = as.character(packageVersion("cobalt"))))
fm <- treat ~ x1 + x2 + x3; fb <- bin ~ x1 + x2 + x3; fc <- dose ~ x1 + x2 + x3
wcase <- function(name, treat, method, estimand, W, focal = NULL, extra = list())
  out$weights[[name]] <<- c(list(treat = treat, method = method, estimand = estimand,
                                 focal = focal, w = unname(W$weights)), extra)
for (m in c("glm", "cbps", "ebal")) {
  wcase(paste0("multi_", m, "_ATE"), "treat", m, "ATE", weightit(fm, data = d, method = m, estimand = "ATE"))
  wcase(paste0("multi_", m, "_ATT"), "treat", m, "ATT", weightit(fm, data = d, method = m, estimand = "ATT", focal = "b"), focal = "b")
  wcase(paste0("bin_", m, "_ATE"), "bin", m, "ATE", weightit(fb, data = d, method = m, estimand = "ATE"))
  wcase(paste0("bin_", m, "_ATT"), "bin", m, "ATT", weightit(fb, data = d, method = m, estimand = "ATT"))
  wcase(paste0("bin_", m, "_ATC"), "bin", m, "ATC", weightit(fb, data = d, method = m, estimand = "ATC"))
}
for (e in c("ATO", "ATM")) {
  wcase(paste0("multi_glm_", e), "treat", "glm", e, weightit(fm, data = d, method = "glm", estimand = e))
  wcase(paste0("bin_glm_", e), "bin", "glm", e, weightit(fb, data = d, method = "glm", estimand = e))
}
wcase("multi_ebal_ATE_m2int", "treat", "ebal", "ATE",
      weightit(fm, data = d, method = "ebal", moments = 2, int = TRUE), extra = list(moments = 2, int = TRUE))
wcase("multi_cbps_ATE_m2", "treat", "cbps", "ATE",
      weightit(fm, data = d, method = "cbps", moments = 2), extra = list(moments = 2))
wcase("cont_glm", "dose", "glm", "ATE", weightit(fc, data = d, method = "glm"))
wcase("cont_ebal", "dose", "ebal", "ATE", weightit(fc, data = d, method = "ebal"))
wcase("cont_ebal_m2", "dose", "ebal", "ATE", weightit(fc, data = d, method = "ebal", moments = 2), extra = list(moments = 2))
wcase("cont_ebal_d3", "dose", "ebal", "ATE", weightit(fc, data = d, method = "ebal", d.moments = 3), extra = list(d_moments = 3))

# cobalt: balance of the glm weights
tab <- function(b) { b <- as.data.frame(b); b$var <- rownames(b); b[b$var != "prop.score", ] }
st <- c("m", "v", "ks")
bcase <- function(name, W, ...) {
  b <- bal.tab(W, un = TRUE, ...)
  rec <- list(w = unname(W$weights), args = list(...))
  if (!is.null(b$Balance.Across.Pairs)) {
    rec$summary <- tab(b$Balance.Across.Pairs)
    rec$pairs <- lapply(b$Pair.Balance, function(p) tab(p$Balance))
  } else rec$summary <- tab(b$Balance)
  rec$obs <- as.data.frame(b$Observations)
  out$balance[[name]] <<- rec
}
Wm <- weightit(fm, data = d, method = "glm", estimand = "ATE")
Wmt <- weightit(fm, data = d, method = "glm", estimand = "ATT", focal = "b")
bcase("multi_ATE", Wm, stats = st)
bcase("multi_ATE_binstd", Wm, stats = "m", binary = "std")
bcase("multi_ATE_nonpair", Wm, stats = st, pairwise = FALSE)
bcase("multi_ATT", Wmt, stats = st)
bcase("multi_ATT_nonpair", Wmt, stats = st, pairwise = FALSE)
for (e in c("ATE", "ATT", "ATC")) bcase(paste0("bin_", e), weightit(fb, data = d, method = "glm", estimand = e), stats = st)
bcase("bin_ATE_binstd", weightit(fb, data = d, method = "glm", estimand = "ATE"), stats = "m", binary = "std")
Wc <- weightit(fc, data = d, method = "glm")
bcase("cont", Wc, stats = c("cor", "sp"))
bcase("cont_binstd", Wc, stats = "cor", binary = "std")
write_json(out, "weightit_cobalt_R.json", digits = NA, auto_unbox = TRUE, null = "null")
