# Reference numbers for sp.panel_match from R PanelMatch 3.1.5.
#
#   python tests/reference_parity/_fixtures/_generate_panel_match_data.py
#   R_LIBS=<lib with PanelMatch 3.1.5> Rscript \
#       tests/reference_parity/_fixtures/_generate_panel_match_R.R
#
# PanelMatch is GPL-3; it is used here as a black box (its outputs only).
# Writes panel_match_R.json next to this script. Everything stored is
# deterministic (matched sets, refinement weights, point estimates, the
# conditional / unconditional analytic standard errors, covariate balance,
# placebo estimates) except the `boot` blocks, which are bootstrap summaries
# under set.seed() and serve as stochastic screens only.

suppressMessages({
  library(PanelMatch)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))

full <- read.csv(file.path(here, "panel_match_data.csv"))
gaps <- read.csv(file.path(here, "panel_match_gaps.csv"))
pd_full <- suppressWarnings(PanelData(full, unit.id = "id", time.id = "t",
                                      treatment = "d", outcome = "y"))
pd_gaps <- suppressWarnings(PanelData(gaps, unit.id = "id", time.id = "t",
                                      treatment = "d", outcome = "y"))

dump_sets <- function(s) {
  nm <- names(s)
  lapply(seq_along(s), function(i) {
    w <- attr(s[[i]], "weights")
    list(
      key = nm[i],
      controls = as.numeric(s[[i]]),
      weights = if (is.null(w)) numeric(0) else as.numeric(w)
    )
  })
}

estimate_block <- function(pm, pd, lead, qoi) {
  out <- list()
  if (qoi == "ate") {
    set.seed(1)
    pe <- PanelEstimate(pm, pd, se.method = "bootstrap", number.iterations = 500)
    out$estimate <- as.numeric(pe$estimate)
    out$boot_se <- as.numeric(pe$standard.error)
    return(out)
  }
  pc <- PanelEstimate(pm, pd, se.method = "conditional")
  pu <- PanelEstimate(pm, pd, se.method = "unconditional")
  out$estimate <- as.numeric(pc$estimate)
  out$se_conditional <- as.numeric(pc$standard.error)
  out$se_unconditional <- as.numeric(pu$standard.error)
  set.seed(1)
  pb <- PanelEstimate(pm, pd, se.method = "bootstrap", number.iterations = 500)
  out$boot_se <- as.numeric(pb$standard.error)
  out
}

balance_block <- function(pm, pd, covs) {
  b <- suppressWarnings(get_covariate_balance(pm, panel.data = pd,
                                              covariates = covs))
  q <- names(b[[1]])[1]
  ref <- b[[1]][[q]]
  unref <- attr(b, "unrefined.balance.results")[[1]][[q]]
  list(
    rows = rownames(ref),
    cols = colnames(ref),
    refined = unname(as.matrix(ref)),
    unrefined = unname(as.matrix(unref))
  )
}

cases <- list(
  none = list(data = "full", lag = 4, refinement.method = "none",
              qoi = "att", lead = 0:3),
  maha = list(data = "full", lag = 4, refinement.method = "mahalanobis",
              qoi = "att", lead = 0:3, size.match = 5,
              covs.formula = ~ x1 + x2),
  maha_lagged = list(data = "full", lag = 3, refinement.method = "mahalanobis",
                     qoi = "att", lead = 0:2, size.match = 4,
                     covs.formula = ~ I(lag(y, 1:3)) + I(lag(x1, 1:2)) + x2),
  maha_diag = list(data = "full", lag = 3, refinement.method = "mahalanobis",
                   qoi = "att", lead = 0:1, size.match = 3,
                   covs.formula = ~ x1 + x2,
                   use.diagonal.variance.matrix = TRUE),
  ps_weight = list(data = "full", lag = 4, refinement.method = "ps.weight",
                   qoi = "att", lead = 0:2,
                   covs.formula = ~ I(lag(y, 1:4)) + I(lag(x1, 1:4))),
  ps_match = list(data = "full", lag = 4, refinement.method = "ps.match",
                  qoi = "att", lead = 0:2, size.match = 5,
                  covs.formula = ~ I(lag(y, 1:2)) + x1 + x2),
  cbps_weight = list(data = "full", lag = 3, refinement.method = "CBPS.weight",
                     qoi = "att", lead = 0:1,
                     covs.formula = ~ I(lag(y, 1:2)) + x1),
  atc_maha = list(data = "full", lag = 3, refinement.method = "mahalanobis",
                  qoi = "atc", lead = 0:2, size.match = 5,
                  covs.formula = ~ x1 + x2),
  art_ps = list(data = "full", lag = 3, refinement.method = "ps.weight",
                qoi = "art", lead = 0:1,
                covs.formula = ~ I(lag(y, 1:2)) + x1),
  ate_maha = list(data = "full", lag = 3, refinement.method = "mahalanobis",
                  qoi = "ate", lead = 0:1, size.match = 5,
                  covs.formula = ~ x1 + x2),
  forbid = list(data = "full", lag = 4, refinement.method = "mahalanobis",
                qoi = "att", lead = 0:3, size.match = 5,
                covs.formula = ~ x1 + x2, forbid.treatment.reversal = TRUE),
  exact = list(data = "full", lag = 3, refinement.method = "mahalanobis",
               qoi = "att", lead = 0:1, size.match = 3,
               covs.formula = ~ x1 + x2, exact.match.variables = "region"),
  atc_ps = list(data = "full", lag = 3, refinement.method = "ps.weight",
                qoi = "atc", lead = 0:2,
                covs.formula = ~ I(lag(y, 1:2)) + x1),
  psm_ties = list(data = "full", lag = 3, refinement.method = "ps.match",
                  qoi = "att", lead = 0:1, size.match = 5,
                  covs.formula = ~ region),
  gaps_exact = list(data = "gaps", lag = 2, refinement.method = "mahalanobis",
                    qoi = "att", lead = 0, size.match = 4,
                    covs.formula = ~ x1, exact.match.variables = "region"),
  art_restrict = list(data = "full", lag = 4, refinement.method = "mahalanobis",
                      qoi = "art", lead = 0:1, size.match = 5,
                      covs.formula = ~ x1 + x2, restrict.control.period = 2),
  restrict = list(data = "full", lag = 4, refinement.method = "none",
                  qoi = "att", lead = 0:1, restrict.control.period = 2),
  gaps_missing = list(data = "gaps", lag = 3, refinement.method = "mahalanobis",
                      qoi = "att", lead = 0:2, size.match = 5,
                      covs.formula = ~ I(lag(y, 1:2)) + x1 + x2),
  gaps_strict = list(data = "gaps", lag = 3, refinement.method = "ps.weight",
                     qoi = "att", lead = 0:1, match.missing = FALSE,
                     covs.formula = ~ I(lag(y, 1:2)) + x1)
)

out <- list(meta = list(
  PanelMatch = as.character(packageVersion("PanelMatch")),
  CBPS = as.character(packageVersion("CBPS")),
  R = R.version.string
))

for (nm in names(cases)) {
  cfg <- cases[[nm]]
  pd <- if (cfg$data == "full") pd_full else pd_gaps
  args <- cfg[setdiff(names(cfg), "data")]
  args$panel.data <- pd
  pm <- suppressWarnings(do.call(PanelMatch, args))
  qoi <- cfg$qoi
  res <- list()
  if (qoi == "ate") {
    res$sets_att <- dump_sets(pm$att)
    res$sets_atc <- dump_sets(pm$atc)
  } else {
    res$sets <- dump_sets(pm[[qoi]])
  }
  res$est <- suppressWarnings(estimate_block(pm, pd, cfg$lead, qoi))
  if (qoi %in% c("att", "atc")) {
    res$balance <- balance_block(pm, pd, c("x1", "x2", "y"))
  }
  out$cases[[nm]] <- res
  cat(nm, "done\n")
}

# placebo test (pre-trend estimates)
pm_pl <- suppressWarnings(PanelMatch(pd_full, lag = 4,
                                     refinement.method = "mahalanobis",
                                     qoi = "att", lead = 0:1, size.match = 5,
                                     covs.formula = ~ x1 + x2,
                                     placebo.test = TRUE))
plc <- suppressWarnings(placebo_test(pm_pl, pd_full, se.method = "conditional"))
plu <- suppressWarnings(placebo_test(pm_pl, pd_full, se.method = "unconditional"))
out$placebo <- list(
  sets = dump_sets(pm_pl$att),
  names = names(plc$estimates),
  estimate = as.numeric(plc$estimates),
  se_conditional = as.numeric(plc$standard.errors),
  se_unconditional = as.numeric(plu$standard.errors)
)

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null",
                  na = "null"),
           file.path(here, "panel_match_R.json"))
cat("wrote panel_match_R.json\n")
