# Reference values for tests/reference_parity/test_did_synth_didvar_parity.py
#
#   sp.continuous_did(method="twfe")   vs  fixest::feols(y ~ dose:post | id + time)
#   sp.did_timevarying_covariates      vs  ptetools::pte_default(d_outcome = TRUE,
#                                            est_method = "reg", xformula = ~x1 + x2)
#                                      and did::att_gt(xformla = ~x1 + x2,
#                                            est_method = "reg")
#   sp.distributional_did              vs  didFF::distDD (configurations the
#                                            didFF extended fixture does not reach)
#
# Regenerate (from the repository root):
#   python  tests/reference_parity/_generate_did_synth_didvar_data.py   # panels
#   Rscript tests/reference_parity/_generate_did_synth_didvar_R.R
#
# Writes _fixtures/did_synth_didvar_R.json and _fixtures/did_synth_didvar_mpdta.csv
# (did::mpdta with lpop and the arithmetic weight used by the didFF fixture).

suppressMessages({
  library(fixest)
  library(ptetools)
  library(did)
  library(didFF)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")
out <- list()

# ---------------------------------------------------------------- fixest --
# sp.continuous_did(method = "twfe") regresses y on dose x post with unit and
# period fixed effects. fixest defaults: ssc(adj = TRUE, fixef.K = "nested",
# cluster.adj = TRUE); iid sigma^2 divides by n - K with K counting every
# absorbed fixed-effect level.
twfe_case <- function(file) {
  d <- read.csv(file.path(fixtures, file))
  d$post <- as.integer(d$time >= 4)
  d$dp <- d$dose * d$post
  res <- list()
  fit <- function(fml, vc) {
    m <- feols(fml, data = d, vcov = vc)
    list(coef = unname(coef(m)["dp"]), se = unname(se(m)["dp"]),
         nobs = nobs(m))
  }
  res$iid <- fit(y ~ dp | id + time, "iid")
  res$cl_id <- fit(y ~ dp | id + time, ~id)
  res$cl_region <- fit(y ~ dp | id + time, ~region)
  res$ctrl_iid <- fit(y ~ dp + x | id + time, "iid")
  res$ctrl_cl_id <- fit(y ~ dp + x | id + time, ~id)
  res
}
out$twfe <- list(
  balanced = twfe_case("did_synth_didvar_contdose.csv"),
  unbalanced = twfe_case("did_synth_didvar_contdose_unbal.csv")
)

# ------------------------------------------------------------- ptetools --
# Caetano, Callaway, Payne & Rodrigues time-varying covariates, the X_{g-1}
# version: long difference Y_t - Y_{g-1}, covariates frozen at the base period
# g - 1 for treated AND comparison units, outcome regression fitted on the
# comparison group only (DRDID::reg_did_panel), never-treated comparison.
tvc <- read.csv(file.path(fixtures, "did_synth_didvar_tvc.csv"))
# did 2.3.0 recodes g == 0 to Inf in place; on an integer column that
# assignment becomes NA, the never-treated units are dropped and the last
# cohort is silently coerced to "never treated". Store g as double.
tvc$g <- as.numeric(tvc$g)
# ptetools 1.0.1 subsets with `subset(data, G == g | ...)`, where `g` is the
# function argument -- but non-standard evaluation resolves it to a data
# column first. A panel whose cohort column is literally named `g` therefore
# keeps every cohort in every cell (the control_group argument is silently
# lost). Hand ptetools a copy without that column name.
tvc_pte <- tvc
tvc_pte$cohort <- tvc_pte$g
tvc_pte$g <- NULL
set.seed(1)
pte_fit <- pte_default(
  yname = "y", gname = "cohort", tname = "time", idname = "id", data = tvc_pte,
  xformula = ~ x1 + x2, d_outcome = TRUE, est_method = "reg",
  control_group = "nevertreated", base_period = "varying",
  cband = FALSE, biters = 50
)
agt <- pte_fit$att_gt
post_rows <- agt$t >= agt$group
out$tvc_pte <- list(
  group = agt$group[post_rows],
  time = agt$t[post_rows],
  att = agt$att[post_rows],
  overall_att = pte_fit$overall_att$overall.att
)

# The same ATT(g, t) from the `did` package: panel 2x2 cells take covariates
# from the earlier (base) period, i.e. X_{g-1} for every post cell.
cs <- suppressWarnings(att_gt(
  yname = "y", gname = "g", tname = "time", idname = "id", data = tvc,
  xformla = ~ x1 + x2, est_method = "reg", control_group = "nevertreated",
  base_period = "universal", bstrap = FALSE, cband = FALSE
))
keep <- cs$t >= cs$group
agg_s <- aggte(cs, type = "simple", bstrap = FALSE, cband = FALSE)
agg_g <- aggte(cs, type = "group", bstrap = FALSE, cband = FALSE)
out$tvc_did <- list(
  group = cs$group[keep], time = cs$t[keep], att = cs$att[keep],
  simple_att = agg_s$overall.att, group_att = agg_g$overall.att
)
# One covariate, to pin the list handling.
cs1 <- suppressWarnings(att_gt(
  yname = "y", gname = "g", tname = "time", idname = "id", data = tvc,
  xformla = ~ x1, est_method = "reg", control_group = "nevertreated",
  base_period = "universal", bstrap = FALSE, cband = FALSE
))
keep1 <- cs1$t >= cs1$group
out$tvc_did_x1 <- list(
  group = cs1$group[keep1], time = cs1$t[keep1], att = cs1$att[keep1],
  simple_att = aggte(cs1, type = "simple", bstrap = FALSE, cband = FALSE)$overall.att,
  group_att = aggte(cs1, type = "group", bstrap = FALSE, cband = FALSE)$overall.att
)

# --------------------------------------------------------------- distDD --
data(mpdta, package = "did")
mp <- as.data.frame(mpdta)[, c("countyreal", "year", "lemp", "lpop", "first.treat")]
names(mp)[names(mp) == "first.treat"] <- "first_treat"
mp$w <- 1 + ((mp$countyreal * 7919) %% 97) / 50
write.csv(mp, file.path(fixtures, "did_synth_didvar_mpdta.csv"), row.names = FALSE)
mp <- read.csv(file.path(fixtures, "did_synth_didvar_mpdta.csv"))

pull_dist <- function(res) {
  list(level = as.character(res$table$level),
       estimates = res$table$test.estimates,
       se = res$table$test.se)
}
# didFF 0.1.0 dies ("arguments imply differing number of rows") whenever a
# bin's influence function is dropped as degenerate, because it builds the
# output table from all bin levels but only the retained estimates. Such cases
# are recorded as errors, not silently skipped.
dd <- function(...) {
  tryCatch(
    pull_dist(suppressWarnings(distDD(data = mp, yname = "lemp", tname = "year",
                                      idname = "countyreal", gname = "first_treat",
                                      nbins = 6, seed = 0, ...))),
    error = function(e) list(error = conditionMessage(e))
  )
}
out$distdd <- list(
  x_dr = dd(xformla = ~ lpop),
  x_reg = dd(xformla = ~ lpop, est_method = "reg"),
  x_ipw = dd(xformla = ~ lpop, est_method = "ipw"),
  notyet = dd(control_group = "notyettreated"),
  dynamic = dd(aggte_type = "dynamic"),
  dynamic_window = dd(aggte_type = "dynamic", min_e = -2, max_e = 2),
  dynamic_balance = dd(aggte_type = "dynamic", balance_e = 1),
  calendar = dd(aggte_type = "calendar"),
  weighted_x_notyet = dd(xformla = ~ lpop, weightsname = "w",
                         control_group = "notyettreated")
)
out$distdd$binpoints <- tryCatch(
  pull_dist(suppressWarnings(distDD(data = mp, yname = "lemp", tname = "year",
                                    idname = "countyreal", gname = "first_treat",
                                    binpoints = c(4, 5, 6, 7, 8, 9), seed = 0))),
  error = function(e) list(error = conditionMessage(e))
)

# distDD dies on the balance_e = 1 case (one bin's influence function is
# degenerate). Recompute that case through distDD's own recipe -- whole-panel
# cut(nbins = 6), 1{Y in b} outcome, att_gt(base_period = "universal"),
# aggte(type = "dynamic", balance_e = 1), SE = sqrt(diag(crossprod(IF) /
# n_eff) / n_eff) over units with a non-zero IF -- so the retained bins still
# have a reference. `NA` marks the degenerate bin.
manual_dd <- function(aggte_type, balance_e) {
  mm <- mp
  mm$first_treat <- as.numeric(mm$first_treat)
  bins <- as.numeric(cut(mm$lemp, breaks = 6, include.lowest = TRUE,
                         dig.lab = 21))
  est <- c(); infs <- list()
  for (b in sort(unique(bins))) {
    mm$yb <- 1 * (bins == b)
    cs <- suppressMessages(suppressWarnings(att_gt(
      yname = "yb", gname = "first_treat", tname = "year",
      idname = "countyreal", data = mm, control_group = "nevertreated",
      est_method = "dr", bstrap = FALSE, cband = FALSE,
      base_period = "universal")))
    ag <- suppressMessages(suppressWarnings(aggte(
      cs, type = aggte_type, balance_e = balance_e, na.rm = TRUE,
      bstrap = FALSE, cband = FALSE)))
    est <- c(est, ag$overall.att)
    infs[[length(infs) + 1]] <- ag$inf.function$dynamic.inf.func
  }
  inf <- do.call(cbind, infs)
  keep <- colSums(inf^2) > 1e-6
  inf_k <- inf[, keep, drop = FALSE]
  n_eff <- sum(rowSums(abs(inf_k)) > 1e-6)
  se <- rep(NA_real_, length(est))
  se[keep] <- sqrt(diag(crossprod(inf_k) / n_eff) / n_eff)
  list(estimates = est, se = se, used = keep)
}
out$distdd_manual <- list(dynamic_balance = manual_dd("dynamic", 1),
                          dynamic = manual_dd("dynamic", NULL))

# ----------------------------------------------------------------- meta --
out$meta <- list(
  r_version = R.version.string,
  fixest_version = as.character(packageVersion("fixest")),
  ptetools_version = as.character(packageVersion("ptetools")),
  did_version = as.character(packageVersion("did")),
  DRDID_version = as.character(packageVersion("DRDID")),
  didFF_version = as.character(packageVersion("didFF")),
  generated_by = "_generate_did_synth_didvar_R.R"
)

write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(fixtures, "did_synth_didvar_R.json"))
cat("wrote", file.path(fixtures, "did_synth_didvar_R.json"), "\n")
