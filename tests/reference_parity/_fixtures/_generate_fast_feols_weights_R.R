# Reference values for sp.fast.feols(weights=, vcov=) vs R fixest.
# Run from this directory: Rscript _generate_fast_feols_weights_R.R
# Writes fast_feols_weights_R.json. fixest defaults throughout (ssc()).
suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})
d <- read.csv("fast_feols_weights.csv")
out <- list()
add <- function(key, fit) {
  ct <- coeftable(fit)
  out[[key]] <<- list(
    b_x1 = unname(ct["x1", "Estimate"]), se_x1 = unname(ct["x1", "Std. Error"]),
    b_x2 = unname(ct["x2", "Estimate"]), se_x2 = unname(ct["x2", "Std. Error"]),
    nobs = nobs(fit)
  )
}
for (fe in c("one", "two")) {
  f <- if (fe == "one") y ~ x1 + x2 | firm else y ~ x1 + x2 | firm + year
  for (wt in c("none", "set")) {
    for (v in c("iid", "hetero", "cluster")) {
      args <- list(f, data = d)
      if (wt == "set") args$weights <- ~w
      if (v == "cluster") args$cluster <- ~g else args$vcov <- v
      add(paste(fe, wt, v, sep = "_"), do.call(feols, args))
    }
  }
}
# Clustered layouts: a key that nests neither effect (c2), one that is an
# absorbed dimension (year), one that nests the firm effect (g). fixest
# drops the nested dimensions from K; these pin that count.
d$c2 <- (d$firm * 7 + d$year) %% 20
for (fe in c("firm + year", "firm", "year")) {
  for (cl in c("c2", "year", "g")) {
    f <- as.formula(paste("y ~ x1 + x2 |", fe))
    add(paste("layout", gsub(" \\+ ", "_", fe), cl, sep = "_"),
        feols(f, data = d, cluster = as.formula(paste0("~", cl))))
  }
}
out[["_meta"]] <- list(
  fixest = as.character(packageVersion("fixest")),
  R = R.version.string,
  generated = format(Sys.time(), "%Y-%m-%d")
)
write_json(out, "fast_feols_weights_R.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("wrote fast_feols_weights_R.json with", length(out) - 1, "cells\n")
