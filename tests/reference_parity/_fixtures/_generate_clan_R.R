# Reference for sp.gate_test(clan=) vs R GenericML::CLAN (GPL >= 3; black
# box, outputs only); checked in
# tests/reference_parity/test_clan_genericml_parity.py.
# Run from this folder:  Rscript _generate_clan_R.R
# Groups: GenericML::quantile_group(proxy, cutoffs = c(.25, .5, .75)).
suppressMessages({library(GenericML); library(jsonlite)})
set.seed(3); n <- 300
d <- data.frame(proxy = round(rnorm(n), 6), z1 = round(rnorm(n), 6),
                z2 = round(runif(n), 6), z3 = rbinom(n, 1, .4),
                d = rep(0:1, n / 2), y = round(rnorm(n), 6))
write.csv(d, "clan_data.csv", row.names = FALSE); d <- read.csv("clan_data.csv")
m <- quantile_group(d$proxy, cutoffs = c(0.25, 0.5, 0.75))
cl <- CLAN(as.matrix(d[, c("z1", "z2", "z3")]), m)
out <- lapply(cl$generic_targets, function(g) list(estimate = unname(g[, "Estimate"]),
                                                    se = unname(g[, "Std. Error"])))
out$group <- unname(apply(m, 1, which))
out$versions <- list(R = R.version.string, GenericML = as.character(packageVersion("GenericML")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17)), "clan_R.json")
