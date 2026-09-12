#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_decomp_R_parity.py
#
# Requires: R 4.5 + DasGuptR + ddecompose + cdgd + rifreg + dineq + jsonlite. Run
# _generate_decomp_data.py first (it writes decomp_gap / _ye / _cps.csv),
# then this file from any directory.
#
# Conventions this fixture pins, each of which decides a number below
# -------------------------------------------------------------------
# * DasGuptR reports standardised rates per population; a factor's effect
#   is the difference of its two standardised rates, first population minus
#   second. The cross-classified example uses
#   ratefunction = "sum(A*B*C*D)" -- the aggregate is a SUM over age groups
#   of the product of the four factors.
# * Kitagawa is Das Gupta with two factors, composition (normalised within
#   population) and rate: ratefunction = "sum(size*rate)/sum(size)".
# * ddecompose::dfl_decompose(reference_0 = TRUE) reweights group 0 to
#   group 1's covariate distribution with a logit of group on the
#   covariates; its composition effect is the counterfactual mean minus
#   group 0's mean. reference_0 = FALSE reweights group 1 instead.
#   No trimming.
# * cdgd0_manual is fed nuisance predictions fitted HERE, independently of
#   StatsPAI: E[Y | R, T, X] by lm() within each of the four (R, T) cells
#   and P(T = 1 | R, X) by glm(binomial) within each group -- the nuisance
#   specification sp.yu_elwert_decompose uses.
# * RIF quantiles: rifreg / ddecompose use Hmisc::wtd.quantile, density() at
#   the quantile and the indicator y <= q (StatsPAI quantile_convention =
#   "rifreg"; dineq::rif uses y < q).
# * RIF Gini: rifreg::get_rif_gini integrates the piecewise-linear Lorenz
#   curve with integrate() (rel.tol 1.2e-4). With the exact area its formula
#   IS dineq::rif(method = "gini"), so the Gini references are
#   dineq::rif + lm and ob_decompose with dineq's RIF as
#   custom_rif_function; the stock rifreg Gini is emitted too, to show
#   the size of its quadrature error.
# * ob_decompose(reweighting = TRUE, trimming = FALSE) is the FFL (2018)
#   reweighted RIF decomposition; reference_0 = TRUE reweights group 0 to
#   group 1's covariates. Terms: observed, composition, structure,
#   specification error, reweighting error.
# * dfl_decompose on variance / quantiles uses Hmisc's weighted variance
#   (denominator sum(w) - 1) and weighted quantile; the Gini is supplied
#   exactly through custom_statistic_function (see the DFL block).
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(DasGuptR); library(ddecompose); library(cdgd); library(rifreg)
  library(dineq); library(jsonlite)
})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
out <- list()

effects <- function(res, p1, p2) {
  res <- res[res$factor != "crude", ]
  f <- unique(res$factor)
  setNames(lapply(f, function(k)
    res$rate[res$factor == k & res$pop == p1] - res$rate[res$factor == k & res$pop == p2]), f)
}

# ---- Das Gupta: two factors, one row per population (Table 2.1) ------------
write.csv(dgeg2_1, file.path(OUT, "decomp_dg2_1.csv"), row.names = FALSE)
r <- dgnpop(dgeg2_1, pop = "pop", factors = c("avg_earnings", "earner_prop"))
out$dg2_1 <- effects(r, "black", "white")

# ---- Das Gupta: four factors x six age groups, 1963 vs 1968 (Table 6.5) ----
x <- subset(dgeg6_5, pop %in% c(1963, 1968))
write.csv(x, file.path(OUT, "decomp_dg6_5.csv"), row.names = FALSE)
r <- dgnpop(x, pop = "pop", factors = c("A", "B", "C", "D"), id_vars = "agegroup",
            ratefunction = "sum(A*B*C*D)")
out$dg6_5 <- effects(r, "1968", "1963")
out$dg6_5_crude <- list(r1968 = sum(with(x[x$pop == 1968, ], A * B * C * D)),
                        r1963 = sum(with(x[x$pop == 1963, ], A * B * C * D)))

# ---- Kitagawa: composition x rate, 1970 vs 1985 (Table 5.1) ----------------
write.csv(dgeg5_1, file.path(OUT, "decomp_dg5_1.csv"), row.names = FALSE)
r <- dgnpop(dgeg5_1, pop = "pop", factors = c("size", "rate"), id_vars = "age_group",
            ratefunction = "sum(size*rate)/sum(size)")
out$kitagawa <- effects(r, "1970", "1985")

# ---- DFL reweighting (gap_closing, method = "ipw") --------------------------
d <- read.csv(file.path(OUT, "decomp_gap.csv"))
d$group <- factor(d$group)
for (ref in c(TRUE, FALSE)) {
  dd <- dfl_decompose(y ~ x1 + x2, data = d, group = group, reference_0 = ref,
                      statistics = "mean", trimming = FALSE)
  s <- dd$decomposition_other_statistics
  out[[if (ref) "dfl_ref0" else "dfl_ref1"]] <- list(
    observed = s[["Observed difference"]], composition = s[["Composition effect"]],
    structure = s[["Structure effect"]])
}
# Oaxaca-Blinder counterfactual (gap_closing, method = "regression")
ob <- ob_decompose(y ~ x1 + x2, data = d, group = group, reference_0 = TRUE)
out$ob_ref0 <- list(observed = ob$ob_decompose$decomposition_terms$Observed_difference[1],
                    composition = ob$ob_decompose$decomposition_terms$Composition_effect[1],
                    structure = ob$ob_decompose$decomposition_terms$Structure_effect[1])

# ---- Yu-Elwert efficient estimator (cdgd0_manual) ---------------------------
e <- read.csv(file.path(OUT, "decomp_ye.csv"))
p1 <- p0 <- ps <- rep(NA_real_, nrow(e))
for (g in 0:1) {
  ig <- e$r == g
  for (dv in 0:1) {
    fit <- lm(y ~ x1 + x2, data = e[ig & e$t == dv, ])
    pr <- predict(fit, newdata = e[ig, ])
    if (dv == 1) p1[ig] <- pr else p0[ig] <- pr
  }
  ps[ig] <- predict(glm(t ~ x1 + x2, family = binomial, data = e[ig, ]),
                    newdata = e[ig, ], type = "response")
}
ce <- cdgd0_manual(Y = "y", D = "t", G = "r", YgivenGX.Pred_D1 = p1,
                   YgivenGX.Pred_D0 = p0, DgivenGX.Pred = ps, data = e)
out$cdgd <- list(point = setNames(as.list(ce$results$point), rownames(ce$results)),
                 se = setNames(as.list(ce$results$se), rownames(ce$results)))

# ---- RIF regression (rifreg package, dineq) --------------------------------
c <- read.csv(file.path(OUT, "decomp_cps.csv"))
fm <- log_wage ~ education + experience
rr <- function(st, ...) unname(rifreg(fm, data = c, statistic = st, ...)$estimates[, 1])
out$rifreg_variance <- rr("variance")
out$rifreg_gini_stock <- rr("gini")
out$rifreg_quantiles <- lapply(c(0.1, 0.5, 0.9), function(p) rr("quantiles", probs = p))
out$dineq_gini_lm <- unname(coef(lm(dineq::rif(c$log_wage, method = "gini") ~ education + experience, data = c)))

# ---- FFL reweighted RIF decomposition (ddecompose::ob_decompose) -----------
c$female <- factor(c$female)
fx <- log_wage ~ education + experience + tenure
gini_exact <- function(dep_var, weights, probs = NULL)
  data.frame(rif_gini = dineq::rif(dep_var, weights = weights, method = "gini"), weights = weights)
ffl <- function(ref, ...) {
  r <- suppressWarnings(ob_decompose(fx, data = c, group = female, reweighting = TRUE,
                                     reference_0 = ref, trimming = FALSE, ...))
  x <- r[[1]]$decomposition_terms[1, ]
  list(observed = x$Observed_difference, composition = x$Composition_effect,
       structure = x$Structure_effect, specification = x$Specification_error,
       reweighting = x$Reweighting_error)
}
for (ref in c(TRUE, FALSE)) {
  tag <- if (ref) "ref0" else "ref1"
  out[[paste0("ffl_variance_", tag)]] <- ffl(ref, rifreg_statistic = "variance")
  out[[paste0("ffl_gini_", tag)]] <- ffl(ref, rifreg_statistic = "custom",
                                         custom_rif_function = gini_exact)
  out[[paste0("ffl_gini_stock_", tag)]] <- ffl(ref, rifreg_statistic = "gini")
  for (p in c(0.1, 0.5, 0.9))
    out[[sprintf("ffl_q%02d_%s", round(100 * p), tag)]] <-
      ffl(ref, rifreg_statistic = "quantiles", rifreg_probs = p)
}

# ---- DFL on non-mean statistics (ddecompose::dfl_decompose) ----------------
# Hmisc::wtd.var / wtd.quantile define the reweighted counterfactual's
# variance and quantile (StatsPAI stat_convention = "hmisc"). The stock Gini
# integrates the Lorenz curve with integrate(); the exact plug-in Gini is
# passed as custom_statistic_function so the comparison is not of quadrature.
gini_stat <- function(dep_var, weights) dineq::gini.wtd(dep_var, weights)
for (ref in c(TRUE, FALSE)) {
  tag <- if (ref) "ref0" else "ref1"
  r <- suppressWarnings(dfl_decompose(fx, data = c, group = female, reference_0 = ref,
                                      statistics = c("variance", "quantiles"),
                                      probs = c(0.1, 0.5, 0.9),
                                      custom_statistic_function = gini_stat,
                                      trimming = FALSE))
  s <- r$decomposition_other_statistics
  q <- r$decomposition_quantiles
  row <- function(df, i) list(observed = df[i, "Observed difference"],
                              composition = df[i, "Composition effect"],
                              structure = df[i, "Structure effect"])
  out[[paste0("dfl_variance_", tag)]] <- row(s, which(s$statistic == "Variance"))
  out[[paste0("dfl_gini_", tag)]] <- row(s, which(s$statistic == "Custom statistic"))
  for (i in seq_len(nrow(q)))
    out[[sprintf("dfl_q%02d_%s", round(100 * q$probs[i]), tag)]] <- row(q, i)
}

out$provenance <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  DasGuptR = as.character(packageVersion("DasGuptR")),
  ddecompose = as.character(packageVersion("ddecompose")),
  cdgd = as.character(packageVersion("cdgd")),
  rifreg = as.character(packageVersion("rifreg")),
  dineq = as.character(packageVersion("dineq")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(OUT, "decomp_R.json"))
cat("wrote decomp_R.json\n")
