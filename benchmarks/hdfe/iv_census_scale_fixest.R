#!/usr/bin/env Rscript
# fixest side of iv_census_scale_bench.py.
#   Rscript iv_census_scale_fixest.R <csv> <threads> <out.json>
# Loads the CSV with data.table::fread, runs a 20k-row warm-up fit, then
# times one full feols IV fit. fixef.rm = "singleton" drops singleton cells
# as ivreghdfe and sp.hdfe_ols do (fixest keeps them by default; they do
# not move the coefficients but change N in the small-sample factor).
# Reports the default clustered SE and, for the cross-engine check, the
# ivreghdfe-convention SE (ssc(adj = TRUE, fixef.K = "nested",
# cluster.adj = TRUE)), which is fixest's default for this model.
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
  library(jsonlite)
})
a <- commandArgs(trailingOnly = TRUE)
csv <- a[1]; threads <- as.integer(a[2]); out <- a[3]
setDTthreads(threads)
setFixest_nthreads(threads)
rss <- function() as.numeric(system(sprintf("ps -o rss= -p %d", Sys.getpid()),
                                    intern = TRUE)) * 1024

t0 <- proc.time()[["elapsed"]]
df <- fread(csv, colClasses = list(double = c("married", "tight", "z", "x"),
                                   integer = c("grp", "tract", "sex", "county")))
load_s <- proc.time()[["elapsed"]] - t0
rss_loaded <- rss()

fml <- married ~ x | grp^tract^sex | tight ~ z
invisible(feols(fml, data = df[1:20000], cluster = ~county,
                fixef.rm = "singleton"))
t0 <- proc.time()[["elapsed"]]
res <- feols(fml, data = df, cluster = ~county, fixef.rm = "singleton")
fit_s <- proc.time()[["elapsed"]] - t0

b <- coef(res); s <- se(res)
names(b) <- sub("^fit_", "", names(b)); names(s) <- names(b)
fs <- fitstat(res, "ivwald1")[[1]]
writeLines(toJSON(list(
  engine = "fixest",
  n_rows = nrow(df),
  load_s = load_s,
  fit_s = fit_s,
  rss_after_load_bytes = rss_loaded,
  coef = as.list(b),
  se = as.list(s),
  n_obs = nobs(res),
  n_fe = as.list(res$fixef_sizes),
  first_stage_F = unname(fs$stat),
  fixest_version = as.character(packageVersion("fixest")),
  r_version = R.version.string,
  threads = threads
), auto_unbox = TRUE, digits = NA, pretty = TRUE), out)
