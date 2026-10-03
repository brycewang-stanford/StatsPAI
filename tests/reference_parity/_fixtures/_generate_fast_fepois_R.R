# Reference values for sp.fast.fepois vs R fixest::fepois (default ssc()).
# Run from this directory: Rscript _generate_fast_fepois_R.R
# Writes fast_fepois_R.json.
suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})
d <- read.csv("fast_fepois.csv")
out <- list()
add <- function(key, fit) {
  ct <- coeftable(fit)
  out[[key]] <<- list(
    b_x1 = unname(ct["x1", "Estimate"]), se_x1 = unname(ct["x1", "Std. Error"]),
    b_x2 = unname(ct["x2", "Estimate"]), se_x2 = unname(ct["x2", "Std. Error"]),
    nobs = nobs(fit)
  )
}
for (fe in c("firm", "firm + year")) {
  f <- as.formula(paste("cnt ~ x1 + x2 |", fe))
  tag <- gsub(" \\+ ", "_", fe)
  for (wt in c("none", "set")) {
    for (v in c("iid", "hetero", "g", "c2", "year")) {
      args <- list(f, data = d)
      if (wt == "set") args$weights <- ~w
      if (v %in% c("g", "c2", "year")) {
        args$cluster <- as.formula(paste0("~", v))
      } else {
        args$vcov <- v
      }
      add(paste(tag, wt, v, sep = "__"), do.call(fepois, args))
    }
  }
}
out[["_meta"]] <- list(
  fixest = as.character(packageVersion("fixest")),
  R = R.version.string,
  generated = format(Sys.time(), "%Y-%m-%d")
)
write_json(out, "fast_fepois_R.json", auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("wrote fast_fepois_R.json with", length(out) - 1, "cells\n")
