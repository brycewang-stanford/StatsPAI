# Reference for sp.ltmle_survival: ltmle::ltmle(survivalOutcome = TRUE) on
# ltmle_survival.csv, one call per horizon t = 1, 2, 3 (the data cut after
# Y_t), regimes always-treat and never-treat, main-terms glm fits (the
# package default formulas), variance.method = "ic".
#
# Recorded per horizon and design (with and without censoring nodes): the
# risk under each regime, its IC standard error, and the risk difference
# with its standard error. ltmle is GPL-2; black box.
#
#   Rscript _generate_ltmle_survival_R.R
suppressMessages({
  library(ltmle)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
d0 <- read.csv(file.path(here, "ltmle_survival.csv"))

run <- function(d, t, censored) {
  keep <- "W"
  for (k in 1:t) {
    keep <- c(keep, paste0("L", k), paste0("A", k),
              if (censored) paste0("C", k), paste0("Y", k))
  }
  dd <- d[, keep]
  cn <- if (censored) paste0("C", 1:t) else NULL
  if (censored) for (v in cn) dd[[v]] <- BinaryToCensoring(is.uncensored = dd[[v]])
  fit <- suppressMessages(ltmle(
    dd, Anodes = paste0("A", 1:t), Cnodes = cn, Lnodes = paste0("L", 1:t),
    Ynodes = paste0("Y", 1:t), survivalOutcome = TRUE,
    abar = list(rep(1, t), rep(0, t)), variance.method = "ic",
    gbounds = c(0.01, 1), estimate.time = FALSE
  ))
  s <- summary(fit)
  list(
    risk_treated = unname(s$effect.measures$treatment$estimate),
    risk_treated_se = unname(s$effect.measures$treatment$std.dev),
    risk_control = unname(s$effect.measures$control$estimate),
    risk_control_se = unname(s$effect.measures$control$std.dev),
    rd = unname(s$effect.measures$ATE$estimate),
    rd_se = unname(s$effect.measures$ATE$std.dev),
    qform = unname(fit$formulas$Qform), gform = unname(fit$formulas$gform)
  )
}

# Design without censoring: the subjects never censored in the file.
never <- rowSums(sapply(1:3, function(k) !is.na(d0[[paste0("C", k)]]) &
                          d0[[paste0("C", k)]] == 0)) == 0
d_nc <- d0[never, ]
res <- list(
  versions = list(R = R.version.string,
                  ltmle = as.character(packageVersion("ltmle"))),
  n_uncensored = sum(never),
  censored = lapply(1:3, function(t) run(d0, t, TRUE)),
  uncensored = lapply(1:3, function(t) run(d_nc, t, FALSE))
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "ltmle_survival_R.json"))
str(res$censored[[3]])
