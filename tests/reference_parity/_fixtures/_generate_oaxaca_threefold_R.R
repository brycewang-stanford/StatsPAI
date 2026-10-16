# Reference for tests/reference_parity/test_oaxaca_threefold_r_parity.py.
# R oaxaca package on tests/agent_bench/data/wage_gap.csv. Run from this directory.
suppressPackageStartupMessages({library(oaxaca); library(jsonlite)})
d <- read.csv("../../agent_bench/data/wage_gap.csv")
set.seed(1)
r <- oaxaca(log_wage ~ educ + exper | female, data = d, R = NULL)
tf <- r$threefold$overall
out <- list(oaxaca_version = as.character(packageVersion("oaxaca")),
            gap = unname(r$y$y.diff), endowments = unname(tf["coef(endowments)"]),
            coefficients = unname(tf["coef(coefficients)"]),
            interaction = unname(tf["coef(interaction)"]))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "oaxaca_threefold_R.json")
