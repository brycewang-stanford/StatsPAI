#!/usr/bin/env Rscript
# R reference values for StatsPAI's dynamic-panel GMM family.
#
# Second, independent anchor alongside Stata (`_generate_dynpanel_stata.do`).
# Where R and Stata disagree the disagreement is itself a fact worth pinning:
# `plm::pgmm` and `xtabond2` use different small-sample conventions for the
# one-step robust VCE, and knowing *which* convention StatsPAI reproduces is
# the point of having two references.
#
# Reads:   dynpanel_abdata.csv  (written by the .do at full double precision)
# Writes:  dynpanel_R.json
#
# Requires: plm (>= 2.6), pdynmc (>= 0.9), jsonlite
# Run:      Rscript _generate_dynpanel_R.R   (from this directory)

suppressMessages({
  library(plm)
  library(jsonlite)
})

here <- tryCatch({
  a <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f[1]))) else getwd()
}, error = function(e) getwd())

df <- read.csv(file.path(here, "dynpanel_abdata.csv"))
pdf <- pdata.frame(df, index = c("id", "year"))

# ---------------------------------------------------------------------------
# pack(): coefficient / SE vectors of a pgmm fit, plus its diagnostics.
# `robust=TRUE` selects plm's Windmeijer-corrected (two-step) or
# cluster-robust (one-step) VCE, which is the analogue of Stata's vce(robust).
# ---------------------------------------------------------------------------
pack <- function(fit, robust = TRUE) {
  s <- summary(fit, robust = robust)
  co <- coef(s)
  sg <- tryCatch(s$sargan, error = function(e) NULL)
  m1 <- tryCatch(s$m1, error = function(e) NULL)
  m2 <- tryCatch(s$m2, error = function(e) NULL)
  list(
    coef = as.list(setNames(unname(co[, "Estimate"]), rownames(co))),
    se = as.list(setNames(unname(co[, "Std. Error"]), rownames(co))),
    diag = list(
      n_obs = unname(nobs(fit)),
      n_groups = unname(pdim(fit)$nT$n),
      sargan_stat = if (is.null(sg)) NA else unname(sg$statistic),
      sargan_df = if (is.null(sg)) NA else unname(sg$parameter),
      m1_z = if (is.null(m1)) NA else unname(m1$statistic),
      m2_z = if (is.null(m2)) NA else unname(m2$statistic)
    )
  )
}

out <- list(
  `_meta` = list(
    R_version = R.version.string,
    plm_version = as.character(packageVersion("plm")),
    data = "abdata (Arellano-Bond 1991 UK employment panel), via dynpanel_abdata.csv",
    note = paste(
      "plm::pgmm instrument lists use lag(y, 2:99) to mean 'all available",
      "deeper level lags', which is Stata xtabond's default GMM-style set."
    )
  )
)

# --- R1: pure AR(1) difference GMM, one-step robust --------------------------
f1 <- pgmm(n ~ lag(n, 1) | lag(n, 2:99),
           data = pdf, effect = "individual",
           model = "onestep", transformation = "d")
out$R1_ar1_diff_1step <- pack(f1, robust = TRUE)
out$R1_ar1_diff_1step_classic <- pack(f1, robust = FALSE)

# --- R2: pure AR(1) difference GMM, two-step (Windmeijer) --------------------
f2 <- pgmm(n ~ lag(n, 1) | lag(n, 2:99),
           data = pdf, effect = "individual",
           model = "twosteps", transformation = "d")
out$R2_ar1_diff_2step_wc <- pack(f2, robust = TRUE)
out$R2_ar1_diff_2step_conv <- pack(f2, robust = FALSE)

# --- R3: Arellano-Bond (1991) Table 4 shape, one-step robust ----------------
f3 <- pgmm(n ~ lag(n, 1:2) + lag(w, 0:1) + lag(k, 0:2) |
             lag(n, 2:99),
           data = pdf, effect = "individual",
           model = "onestep", transformation = "d")
out$R3_ab1991_diff_1step <- pack(f3, robust = TRUE)

# --- R4: same, two-step Windmeijer ------------------------------------------
f4 <- pgmm(n ~ lag(n, 1:2) + lag(w, 0:1) + lag(k, 0:2) |
             lag(n, 2:99),
           data = pdf, effect = "individual",
           model = "twosteps", transformation = "d")
out$R4_ab1991_diff_2step_wc <- pack(f4, robust = TRUE)

# --- R5: SYSTEM GMM (transformation = "ld"), one-step robust ----------------
f5 <- pgmm(n ~ lag(n, 1) + w + k | lag(n, 2:99),
           data = pdf, effect = "individual",
           model = "onestep", transformation = "ld")
out$R5_sys_1step <- pack(f5, robust = TRUE)

# --- R6: system GMM, two-step Windmeijer ------------------------------------
f6 <- pgmm(n ~ lag(n, 1) + w + k | lag(n, 2:99),
           data = pdf, effect = "individual",
           model = "twosteps", transformation = "ld")
out$R6_sys_2step_wc <- pack(f6, robust = TRUE)

# --- R7: collapsed instruments (plm supports collapse= since 2.6) -----------
f7 <- tryCatch(
  pgmm(n ~ lag(n, 1) | lag(n, 2:99),
       data = pdf, effect = "individual",
       model = "onestep", transformation = "d", collapse = TRUE),
  error = function(e) NULL
)
if (!is.null(f7)) out$R7_ar1_diff_collapse <- pack(f7, robust = TRUE)

# --- R8: capped instrument depth (lags 2..4 only) ---------------------------
f8 <- pgmm(n ~ lag(n, 1) | lag(n, 2:4),
           data = pdf, effect = "individual",
           model = "onestep", transformation = "d")
out$R8_ar1_diff_lag2to4 <- pack(f8, robust = TRUE)

# --- R9: pdynmc cross-check (independent implementation) --------------------
if (requireNamespace("pdynmc", quietly = TRUE)) {
  suppressMessages(library(pdynmc))
  d9 <- df[order(df$id, df$year), c("id", "year", "n", "w", "k")]
  f9 <- tryCatch(
    pdynmc::pdynmc(
      dat = d9, varname.i = "id", varname.t = "year",
      use.mc.diff = TRUE, use.mc.lev = FALSE, use.mc.nonlin = FALSE,
      include.y = TRUE, varname.y = "n", lagTerms.y = 1,
      fur.con = FALSE, include.dum = FALSE,
      w.mat = "iid.err", std.err = "corrected",
      estimation = "onestep", opt.meth = "none"
    ),
    error = function(e) NULL
  )
  if (!is.null(f9)) {
    # pdynmc keeps coefficients and (corrected = robust) SEs as named lists.
    cf <- unlist(f9$coefficients)
    sd <- unlist(f9$stderr)
    out$R9_pdynmc_ar1_diff_1step <- list(
      coef = as.list(cf),
      se = as.list(setNames(unname(sd), names(cf))),
      diag = list(pdynmc_version = as.character(packageVersion("pdynmc")))
    )
  }
}

write_json(out, file.path(here, "dynpanel_R.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "null")
cat(sprintf("wrote dynpanel_R.json: %d specs\n", length(out) - 1L))
