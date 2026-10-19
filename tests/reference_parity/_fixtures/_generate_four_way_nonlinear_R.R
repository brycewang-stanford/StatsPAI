# CMAverse 0.1.0 (GPL-3, used as a black box: outputs only) reference for
# sp.four_way_decomposition(yreg=, mreg=, vce='ols') on four_way_nonlinear.csv.
#   cd tests/reference_parity/_fixtures && Rscript _generate_four_way_nonlinear_R.R
# cmest(model = "rb", estimation = "paramfunc", inference = "delta"),
# covariates at their means (CMAverse's default basecval).
suppressMessages(library(CMAverse))
df <- read.csv("four_way_nonlinear.csv")
cases <- list(
  "1" = list(y = "ybc", m = "mc", yreg = "logistic", mreg = "linear", mval = 0),
  "2" = list(y = "ybb", m = "mb", yreg = "logistic", mreg = "logistic", mval = 0),
  "3" = list(y = "ypc", m = "mc", yreg = "poisson", mreg = "linear", mval = 1),
  "4" = list(y = "ypb", m = "mb", yreg = "poisson", mreg = "logistic", mval = 1)
)
keep <- c("Rte", "ERcde", "ERintref", "ERintmed", "ERpnie")
out <- lapply(cases, function(k) {
  r <- suppressWarnings(suppressMessages(cmest(
    data = df, model = "rb", outcome = k$y, exposure = "a", mediator = k$m,
    basec = c("c1", "c2"), EMint = TRUE, mreg = list(k$mreg), yreg = k$yreg,
    astar = 0, a = 1, mval = list(k$mval), yval = 1,
    estimation = "paramfunc", inference = "delta")))
  list(b = as.list(r$effect.pe[keep]), se = as.list(r$effect.se[keep]))
})
writeLines(jsonlite::toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE),
           "four_way_nonlinear_R.json")
