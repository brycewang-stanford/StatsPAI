#!/usr/bin/env Rscript
# Frozen R references for the inference / sensitivity family:
#   tests/reference_parity/test_inference_sens_R_parity.py
#
# Regenerate (from the repository root, after _generate_inference_sens_data.py):
#   Rscript tests/reference_parity/_generate_inference_sens_R.R
#
# Every block calls the canonical R implementation on the committed CSVs and
# writes values at full precision (17 significant digits).
# Package versions are recorded in "_meta".
suppressPackageStartupMessages({
  library(jsonlite)
  library(sandwich)
  library(clubSandwich)
  library(summclust)
  library(fwildclusterboot)
  library(ri2)
  library(DOS2)
  library(rbounds)
  library(EValue)
  library(robomit)
})

FIX <- "tests/reference_parity/_fixtures"
rd <- function(f) read.csv(file.path(FIX, f))
num <- function(x) unname(as.numeric(x))
mat <- function(m) unname(lapply(seq_len(nrow(m)), function(i) num(m[i, ])))

reg <- rd("inference_sens_reg.csv")
out <- list()

# ---------------------------------------------------------------------------
# 1. Cluster-robust variance: sandwich::vcovCL / vcovJK, clubSandwich, summclust
# ---------------------------------------------------------------------------
fit <- lm(y ~ d + x1 + x2, data = reg)
out$cluster <- list(
  coef = num(coef(fit)),
  # one-way CR1: G/(G-1) * (n-1)/(n-k)  (Stata regress, vce(cluster))
  V_cr1_s12 = mat(vcovCL(fit, cluster = ~s12, type = "HC1", cadjust = TRUE)),
  # one-way CR0 without any finite-sample factor
  V_cr0_s12 = mat(vcovCL(fit, cluster = ~s12, type = "HC0", cadjust = FALSE)),
  # two-way CGM (2011): each component carries its own G/(G-1)
  V_cr1_s12_yr = mat(vcovCL(fit, cluster = ~ s12 + yr, type = "HC1",
                            cadjust = TRUE, multi0 = FALSE)),
  V_cr1_s12_yr_fix = mat(vcovCL(fit, cluster = ~ s12 + yr, type = "HC1",
                                cadjust = TRUE, multi0 = FALSE, fix = TRUE)),
  # delete-one-cluster jackknife, (G-1)/G, centred at the replicate mean
  V_jk_mean = mat(vcovJK(fit, cluster = ~s12, center = "mean")),
  # ... centred at the full-sample estimate (MacKinnon-Nielsen-Webb CV3)
  V_jk_estimate = mat(vcovJK(fit, cluster = ~s12, center = "estimate")),
  # clubSandwich CR3 (analytic (I - H_gg)^-1 form, no (G-1)/G factor)
  V_clubsandwich_cr3 = mat(as.matrix(vcovCR(fit, cluster = reg$s12, type = "CR3")))
)
sc <- summclust(fit, cluster = ~s12, params = c("d", "x1", "x2"))
out$cluster$summclust_vcov_cv3 <- mat(sc$vcov)
out$cluster$summclust_coef_names <- rownames(sc$vcov)

# ---------------------------------------------------------------------------
# 2. Restricted wild cluster bootstrap: fwildclusterboot::boottest.
#    G = 12 and B = 9999 >= 2^12, so fwildclusterboot enumerates all 4096
#    Rademacher sign vectors (check_set_full_enumeration): the bootstrap
#    distribution, p-value and test-inversion CI are exact, not Monte Carlo.
#    tol / maxiter tighten the uniroot search for the CI endpoints, which are
#    the jump locations of a step function of the null value.
# ---------------------------------------------------------------------------
set.seed(1)
dqrng::dqset.seed(1)
wcb <- function(param, r) {
  b <- suppressWarnings(suppressMessages(boottest(
    fit, param = param, r = r, clustid = "s12", B = 9999, type = "rademacher",
    sign_level = 0.05, tol = 1e-13, maxiter = 1000
  )))
  list(param = param, r = r, p = num(b$p_val), t = num(b$t_stat),
       B = num(b$boot_iter), ci = num(b$conf_int))
}
out$wild <- list(d_h0 = wcb("d", 0), x1_h0 = wcb("x1", 0), d_h02 = wcb("d", 0.2),
                 x2_h0 = wcb("x2", 0))

# ---------------------------------------------------------------------------
# 3. Randomization inference: ri2::conduct_ri. Every design has at most 4900
#    assignments and sims = 10000, so randomizr::obtain_permutation_matrix
#    enumerates all of them -- the randomization distribution is exact.
#    Statistics: difference in means (ri2's default, the Y ~ Z coefficient),
#    Welch t (stats::t.test), Kolmogorov-Smirnov D (stats::ks.test) and the
#    standardized Wilcoxon rank-sum z (normal approximation, no tie or
#    continuity correction -- the statistic scipy.stats.ranksums returns).
# ---------------------------------------------------------------------------
ri_simple <- rd("inference_sens_ri_simple.csv")
ri_cluster <- rd("inference_sens_ri_cluster.csv")
ri_strat <- rd("inference_sens_ri_strat.csv")
stat_t <- function(dat) unname(t.test(dat$y[dat$d == 1], dat$y[dat$d == 0])$statistic)
stat_ks <- function(dat) unname(suppressWarnings(ks.test(dat$y[dat$d == 1], dat$y[dat$d == 0]))$statistic)
stat_rs <- function(dat) {
  n1 <- sum(dat$d == 1); n0 <- sum(dat$d == 0); n <- n1 + n0
  w <- sum(rank(dat$y)[dat$d == 1])
  (w - n1 * (n + 1) / 2) / sqrt(n1 * n0 * (n + 1) / 12)
}
ri_run <- function(dat, decl, test_function = NULL) {
  r <- if (is.null(test_function)) {
    conduct_ri(y ~ d, assignment = "d", outcome = "y", declaration = decl,
               sharp_hypothesis = 0, data = dat, sims = 10000)
  } else {
    conduct_ri(test_function = test_function, assignment = "d", outcome = "y",
               declaration = decl, sharp_hypothesis = 0, data = dat, sims = 10000)
  }
  two <- summary(r, p = "two-tailed")
  up <- summary(r, p = "upper")
  list(observed = num(two$estimate), p_two = num(two$two_tailed_p_value),
       p_upper = num(up$upper_p_value), n_assign = nrow(r$sims_df))
}
decl_simple <- randomizr::declare_ra(N = nrow(ri_simple), m = sum(ri_simple$d))
decl_cluster <- randomizr::declare_ra(clusters = ri_cluster$cl, m = 4)
decl_strat <- randomizr::declare_ra(blocks = ri_strat$st, block_m = c(4, 4))
out$ri <- list(
  simple_diff = ri_run(ri_simple, decl_simple),
  simple_t = ri_run(ri_simple, decl_simple, stat_t),
  simple_ks = ri_run(ri_simple, decl_simple, stat_ks),
  simple_ranksum = ri_run(ri_simple, decl_simple, stat_rs),
  cluster_diff = ri_run(ri_cluster, decl_cluster),
  strat_diff = ri_run(ri_strat, decl_strat)
)

# ---------------------------------------------------------------------------
# 4. Rosenbaum bounds on matched-pair differences (integer outcomes: the
#    differences contain ties and exact zeros).
#    * DOS2::senWilcox -- Rosenbaum's own implementation of the Wilcoxon
#      signed-rank bound: zeros ranked then weighted 0, no continuity
#      correction; upper-bound p for "greater" / "less" / "twosided".
#    * rbounds::psens -- same statistic but zeros dropped BEFORE ranking;
#      returns lower and upper bounds ROUNDED TO 4 DECIMALS by the package.
#    * stats::binom.test -- the exact sign-test bound P(B >= T),
#      B ~ Binomial(n_nonzero, Gamma / (1 + Gamma)).
# ---------------------------------------------------------------------------
pairs <- rd("inference_sens_pairs.csv")
dd <- pairs$diff
gammas <- c(1, 1.5, 2, 2.5, 3)
sw <- function(alt) sapply(gammas, function(g) senWilcox(dd, gamma = g, alternative = alt)$pval)
ps <- psens(pairs$y_t, pairs$y_c, Gamma = 3, GammaInc = 0.5)$bounds
nz <- dd[dd != 0]
bt <- function(k, n, g) binom.test(k, n, p = g / (1 + g), alternative = "greater")$p.value
out$rosenbaum <- list(
  gamma = gammas,
  senwilcox_greater = num(sw("greater")),
  senwilcox_less = num(sw("less")),
  senwilcox_twosided = num(sw("twosided")),
  psens_gamma = num(ps$Gamma),
  psens_lower = num(ps[["Lower bound"]]),
  psens_upper = num(ps[["Upper bound"]]),
  sign_upper_greater = num(sapply(gammas, function(g) bt(sum(nz > 0), length(nz), g))),
  sign_lower_greater = num(sapply(gammas, function(g) bt(sum(nz > 0), length(nz), 1 / g))),
  sign_upper_less = num(sapply(gammas, function(g) bt(sum(nz < 0), length(nz), g)))
)

# ---------------------------------------------------------------------------
# 5. E-values (EValue): evalues.RD (exact risk-difference E-value, grid
#    search), the confounding bias factor through multi_bound(confounding()),
#    and the relative-risk path of evalue_from_result: twoXtwoRR (Katz-log CI)
#    fed to evalues.RR.
# ---------------------------------------------------------------------------
rd_case <- function(n11, n10, n01, n00, true = 0, alpha = 0.05, grid = 1e-4) {
  e <- evalues.RD(n11, n10, n01, n00, true = true, alpha = alpha, grid = grid)
  list(cells = c(n11, n10, n01, n00), true = true, alpha = alpha, grid = grid,
       est = num(e$est.Evalue), lower = num(e$lower.Evalue))
}
bf_case <- function(a, b) list(rr_eu = a, rr_ud = b,
  bias = num(suppressWarnings(multi_bound(confounding(), RRAUc = a, RRUcY = b))))
rr_case <- function(n11, n10, n01, n00) {
  tt <- twoXtwoRR(n11, n10, n01, n00)
  e <- suppressMessages(evalues.RR(tt[["point"]], tt[["lower"]], tt[["upper"]]))
  list(cells = c(n11, n10, n01, n00), rr = num(tt[["point"]]),
       lo = num(tt[["lower"]]), hi = num(tt[["upper"]]),
       e_point = num(e["E-values", "point"]),
       e_ci = num(if (tt[["point"]] >= 1) e["E-values", "lower"] else e["E-values", "upper"]))
}
out$evalue <- list(
  rd = list(rd_case(200, 150, 100, 250),
            rd_case(200, 150, 100, 250, true = 0.1),
            rd_case(200, 150, 100, 250, alpha = 0.1),
            rd_case(40, 60, 30, 70),
            rd_case(500, 500, 300, 700, grid = 1e-3)),
  bias_factor = list(bf_case(2, 2), bf_case(3, 3), bf_case(1.5, 4),
                     bf_case(1, 5), bf_case(10, 1.2)),
  from_result_rr = list(rr_case(30, 70, 20, 80), rr_case(60, 40, 30, 70),
                        rr_case(20, 80, 40, 60))
)

# ---------------------------------------------------------------------------
# 6. Oster (2019): robomit::o_delta / o_beta (R port of Oster's psacalc).
#    NOTE robomit rounds delta* and beta* to 6 decimals (round(x, 6)) before
#    returning them, so these values carry +-5e-7 absolute rounding; Stata
#    psacalc (in the Stata fixture) is the full-precision reference.
# ---------------------------------------------------------------------------
fit_long <- lm(y ~ t + x1 + x2, data = reg)
rm13 <- min(1, 1.3 * summary(fit_long)$r.squared)
ov <- function(tab, name) num(tab$Value[tab$Name == name])
out$oster <- list(
  rmax13 = rm13,
  delta_rm13 = ov(o_delta(y = "y", x = "t", con = "x1 + x2", beta = 0, R2max = rm13,
                          type = "lm", data = reg), "delta*"),
  delta_rm1 = ov(o_delta(y = "y", x = "t", con = "x1 + x2", beta = 0, R2max = 1,
                         type = "lm", data = reg), "delta*"),
  beta_rm13_d1 = ov(o_beta(y = "y", x = "t", con = "x1 + x2", delta = 1, R2max = rm13,
                           type = "lm", data = reg), "beta*"),
  beta_rm13_d05 = ov(o_beta(y = "y", x = "t", con = "x1 + x2", delta = 0.5, R2max = rm13,
                            type = "lm", data = reg), "beta*"),
  beta_rm13_d2 = ov(o_beta(y = "y", x = "t", con = "x1 + x2", delta = 2, R2max = rm13,
                           type = "lm", data = reg), "beta*")
)

# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
pv <- function(p) as.character(utils::packageVersion(p))
out[["_meta"]] <- list(
  R = R.version.string,
  sandwich = pv("sandwich"),
  clubSandwich = pv("clubSandwich"),
  summclust = pv("summclust"),
  fwildclusterboot = pv("fwildclusterboot"),
  ri2 = pv("ri2"),
  DOS2 = pv("DOS2"),
  rbounds = pv("rbounds"),
  EValue = pv("EValue"),
  robomit = pv("robomit")
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null"),
           file.path(FIX, "inference_sens_R.json"))
cat("wrote inference_sens_R.json\n")
