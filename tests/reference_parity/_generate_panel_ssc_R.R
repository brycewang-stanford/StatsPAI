# R fixest reference for tests/reference_parity/test_panel_ssc_stata_parity.py
#
# Reads the same bytes Stata wrote (_fixtures/panel_ssc_data.csv) and fits the
# sp.panel designs with fixest defaults, i.e. ssc(adj = TRUE,
# fixef.K = "nested", cluster.adj = TRUE, t.df = "min").  Pins
# sp.panel(..., ssc = "fixest").
#
# Run from tests/reference_parity/:  Rscript _generate_panel_ssc_R.R
suppressMessages({
  library(fixest)
  library(jsonlite)
})

df <- read.csv("_fixtures/panel_ssc_data.csv")
df <- df[order(df$id, df$t), ]

specs <- list(
  fe     = y ~ x1 + x2 | id,
  twoway = y ~ x1 + x2 | id + t,
  pooled = y ~ x1 + x2
)
vcovs <- list(
  unadjusted = "iid",
  robust     = "hetero",
  cluster_id = ~id,
  cluster_st = ~st,
  cluster_t  = ~t
)

out <- list()
for (wt in c(FALSE, TRUE)) {
  for (m in names(specs)) {
    for (v in names(vcovs)) {
      fit <- feols(specs[[m]], data = df, vcov = vcovs[[v]], fixef.tol = 1e-11,
                   weights = if (wt) ~w else NULL)
      ct <- coeftable(fit)
      key <- paste0(if (wt) "w_" else "", m, "_", v)
      rec <- list()
      for (term in rownames(ct)) {
        nm <- if (term == "(Intercept)") "_cons" else term
        rec[[paste0("b_", nm)]] <- unname(ct[term, "Estimate"])
        rec[[paste0("se_", nm)]] <- unname(ct[term, "Std. Error"])
      }
      rec[["t_df"]] <- unname(degrees_freedom(fit, "t"))
      rec[["N"]] <- nobs(fit)
      out[[key]] <- rec
    }
  }
}
# First differences: fixest has no FD estimator, so fit the differenced
# rows (period t-1 observed) without an intercept -- regress D.y D.x, nocons.
d <- df
d$lag_t <- ave(d$t, d$id, FUN = function(z) c(NA, head(z, -1)))
for (v in c("x1", "x2", "y")) {
  d[[paste0("D.", v)]] <- ave(d[[v]], d$id, FUN = function(z) c(NA, diff(z)))
}
d <- d[!is.na(d$lag_t) & d$t == d$lag_t + 1, ]
for (v in names(vcovs)) {
  fit <- feols(D.y ~ D.x1 + D.x2 - 1, data = d, vcov = vcovs[[v]])
  ct <- coeftable(fit)
  rec <- list()
  for (term in rownames(ct)) {
    rec[[paste0("b_", term)]] <- unname(ct[term, "Estimate"])
    rec[[paste0("se_", term)]] <- unname(ct[term, "Std. Error"])
  }
  rec[["t_df"]] <- unname(degrees_freedom(fit, "t"))
  rec[["N"]] <- nobs(fit)
  out[[paste0("fd_", v)]] <- rec
}

out[["_meta"]] <- list(
  fixest_version = as.character(packageVersion("fixest")),
  r_version = R.version.string,
  generated = format(Sys.Date())
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE),
           "_fixtures/panel_ssc_fixest.json")
cat("wrote _fixtures/panel_ssc_fixest.json\n")
