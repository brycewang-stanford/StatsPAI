#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_decomp_R_parity.py
#
# Requires: R 4.5 + DasGuptR + ddecompose + cdgd + jsonlite. Run
# _generate_decomp_data.py first (it writes decomp_gap.csv / decomp_ye.csv),
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
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(DasGuptR); library(ddecompose); library(cdgd); library(jsonlite)
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

out$provenance <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  DasGuptR = as.character(packageVersion("DasGuptR")),
  ddecompose = as.character(packageVersion("ddecompose")),
  cdgd = as.character(packageVersion("cdgd")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(OUT, "decomp_R.json"))
cat("wrote decomp_R.json\n")
