# Reference for sp.etwfe, option by option, on _fixtures/etwfe_options_panel.csv
# (written by _generate_etwfe_options_data.py): estimates AND standard errors
# of every emfx aggregation (simple / event / group / calendar).
#
# R 4.5, etwfe 0.6.2, fixest 0.14.0, marginaleffects 0.32.0, numDeriv.
# marginaleffects differentiates numerically, by forward differences unless
# told otherwise, which leaves its delta-method SEs ~1e-7 to 1e-5 relative
# from the exact linear map; numDeriv's Richardson extrapolation is used here
# so that the SEs can be compared at 1e-6.
#
# Specifications (key -> call):
#   notyet / never            etwfe(y ~ 0, cgroup), vcov = ~id
#   notyet_hetero             etwfe(y ~ 0), vcov = "hetero"  (sp.etwfe(panel=False))
#   w_notyet / w_never        etwfe(..., weights = ~w), vcov = ~id;
#                             emfx() (every treated row weight one: agg_weights='unit')
#                             and emfx(wts = "w") (estimation weights: agg_weights='estimation')
#   xv_notyet / xv_never      etwfe(..., xvar = xv), emfx(by_xvar = FALSE)
#   ctrl_notyet / ctrl_never  the additive-control design sp.etwfe(controls=) fits
#                             (Stata jwdid's exovar(); R etwfe interacts controls
#                             with the cells instead): fixest::feols on etwfe's own
#                             .Dtreat cell basis plus x1, cohort + year effects,
#                             aggregated by marginaleffects::slopes exactly as
#                             emfx does (wts = 1 per treated row)
#   xcat_notyet / xcat_never  categorical moderator xcat (levels a / b / c):
#                             the per-level design sp.etwfe fits -- each level
#                             dummy demeaned within cohort x period cells,
#                             interacted with the cells and, one slope per
#                             level, with the periods -- by fixest::feols and
#                             aggregated as emfx does.  etwfe 0.6.2 itself
#                             writes the period term of a factor as
#                             i(year, (xcatb_dm + xcatc_dm)), one summed
#                             slope; xcat_etwfe_notyet records that fit, whose
#                             estimates coincide (the terms are orthogonal to
#                             the cells) and whose SEs do not
#   pois_* / logi_*           etwfe(yc | yb ~ 0, family = "poisson" | "logit"),
#                             vcov = ~id; emfx on both scales
#   pois_xv_notyet            the same Poisson fit with xvar = xv
#
# The nonlinear fits are run to convergence twice over.  fixest's default
# glm.tol = 1e-8 stops IRLS with the bread ~1e-6 relative from the converged
# one, so glm.tol = 1e-12; and fixest's vcov uses the IRLS weights of the
# *penultimate* iterate, which near the optimum is ~sqrt(glm.tol) away
# (8e-7 relative in the logit SEs here, with the estimates equal to 1e-14),
# so each model is refitted from its own solution (start = coef(fit)): one
# IRLS pass whose weights are those of the converged coefficients.  Together
# these remove the whole of the 1e-6 to 1e-5 SE gap earlier comparisons
# against etwfe's GLM fits showed.
#
# xv_* also record each cohort's moderator slope: the cell slopes
# .Dtreat:g:year:xv_dm averaged over the cohort's post-treatment cells by
# their observation counts, with the SE from vcov(fit).
#
# Values are written with 17 significant digits; a reference period (SE NA
# in emfx) is written as null.
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_etwfe_options_R.R
suppressPackageStartupMessages({
  library(etwfe)
  library(fixest)
  library(marginaleffects)
  library(jsonlite)
})
options(marginaleffects_numDeriv = list(method = "Richardson"))

d <- read.csv("tests/reference_parity/_fixtures/etwfe_options_panel.csv")
d$g <- as.numeric(d$g)
d$year <- as.integer(d$year)

key_of <- function(type) {
  switch(type, simple = NULL, group = "g", calendar = "year", event = "event")
}

grab <- function(e, type) {
  e <- as.data.frame(e)
  out <- list(estimate = unname(e$estimate), std.error = unname(e$std.error))
  k <- key_of(type)
  if (!is.null(k)) out$key <- unname(as.numeric(e[[k]]))
  out
}

all_types <- function(fit, ...) {
  res <- list()
  for (type in c("simple", "group", "calendar", "event")) {
    res[[type]] <- grab(emfx(fit, type = type, ...), type)
  }
  res
}

ssc_of <- function(fit) {
  list(n = nobs(fit), K = unname(fit$nparams))
}

out <- list()

# --- linear, unweighted ----------------------------------------------------
for (cg in c("notyet", "never")) {
  fit <- etwfe(fml = y ~ 0, tvar = year, gvar = g, data = d, vcov = ~id,
               cgroup = cg)
  out[[cg]] <- c(all_types(fit), list(ssc = ssc_of(fit)))
}
fit <- etwfe(fml = y ~ 0, tvar = year, gvar = g, data = d, vcov = "hetero")
out$notyet_hetero <- c(all_types(fit), list(ssc = ssc_of(fit)))

# --- linear, weighted ------------------------------------------------------
for (cg in c("notyet", "never")) {
  fit <- etwfe(fml = y ~ 0, tvar = year, gvar = g, data = d, vcov = ~id,
               cgroup = cg, weights = ~w)
  out[[paste0("w_", cg)]] <- list(
    unit = all_types(fit),
    estimation = all_types(fit, wts = "w"),
    ssc = ssc_of(fit)
  )
}

# --- linear, continuous moderator -----------------------------------------
xv_slopes <- function(fit) {
  b <- coef(fit)
  V <- vcov(fit)
  nm <- grep("^\\.Dtreat:g::[0-9]+:year::[0-9]+:xv_dm$", names(b), value = TRUE)
  g <- as.numeric(sub("^\\.Dtreat:g::([0-9]+):.*$", "\\1", nm))
  tt <- as.numeric(sub("^.*:year::([0-9]+):xv_dm$", "\\1", nm))
  n <- mapply(function(gg, t0) sum(d$g == gg & d$year == t0), g, tt)
  res <- list(cohort = numeric(0), estimate = numeric(0), std.error = numeric(0))
  for (gg in sort(unique(g))) {
    on <- tt >= g & g == gg
    w <- rep(0, length(nm))
    w[on] <- n[on] / sum(n[on])
    res$cohort <- c(res$cohort, gg)
    res$estimate <- c(res$estimate, sum(w * b[nm]))
    res$std.error <- c(res$std.error,
                       sqrt(drop(t(w) %*% V[nm, nm] %*% w)))
  }
  res
}
for (cg in c("notyet", "never")) {
  fit <- etwfe(fml = y ~ 0, tvar = year, gvar = g, data = d, vcov = ~id,
               cgroup = cg, xvar = xv)
  out[[paste0("xv_", cg)]] <- c(all_types(fit, by_xvar = FALSE),
                                list(slopes = xv_slopes(fit), ssc = ssc_of(fit)))
}

# --- linear, additive control (exovar design) ------------------------------
# etwfe's .Dtreat / i(g, i.year) cell basis, built exactly as etwfe() does
# for a panel with never-treated units (gref = 0 < min(year)).
ctrl_fit <- function(cg) {
  dd <- d
  if (cg == "notyet") {
    dd$.Dtreat <- dd$year >= dd$g & dd$g != 0
    m <- feols(y ~ .Dtreat:i(g, i.year, ref = 0, ref2 = 2001) + x1 | g + year,
               data = dd, vcov = ~id, notes = FALSE)
    nd <- dd[dd$.Dtreat, ]
  } else {
    dd$.Dtreat <- dd$year != dd$g - 1L
    m <- feols(y ~ .Dtreat:i(g, i.year, ref = 0) + x1 | g + year,
               data = dd, vcov = ~id, notes = FALSE)
    nd <- dd[dd$g != 0, ]
  }
  res <- emfx_like(m, nd, cg)
  res$x1 <- list(estimate = unname(coef(m)["x1"]), std.error = unname(se(m)["x1"]))
  res
}

# emfx's own aggregation, on a model fitted outside etwfe(): slopes of
# .Dtreat over the treated rows (never-treated controls: the post-treatment
# rows for simple / group / calendar, every cohort row for event), each row
# weight one.
emfx_like <- function(m, nd, cg) {
  nd$N <- 1L
  nd$event <- nd$year - nd$g
  res <- list()
  for (type in c("simple", "group", "calendar", "event")) {
    sub <- nd
    if (cg == "never" && type != "event") {
      sub <- sub[sub$.Dtreat & sub$year >= sub$g, ]
    }
    by <- switch(type, simple = ".Dtreat", group = "g", calendar = "year",
                 event = "event")
    s <- slopes(m, newdata = sub, wts = "N", variables = ".Dtreat", by = by)
    res[[type]] <- grab(s, type)
  }
  res$ssc <- ssc_of(m)
  res
}
out$ctrl_notyet <- ctrl_fit("notyet")
out$ctrl_never <- ctrl_fit("never")

# --- linear, categorical moderator (per-level design) ----------------------
xcat_fit <- function(cg) {
  dd <- d
  ct <- interaction(dd$g, dd$year, drop = TRUE)
  for (lev in c("b", "c")) {
    x <- as.numeric(dd$xcat == lev)
    dd[[paste0("xcat", lev, "_dm")]] <- x - ave(x, ct)
  }
  if (cg == "notyet") {
    dd$.Dtreat <- dd$year >= dd$g & dd$g != 0
    m <- feols(y ~ .Dtreat:i(g, i.year, ref = 0, ref2 = 2001) /
                 (xcatb_dm + xcatc_dm) + i(year, xcatb_dm, ref = 2001) +
                 i(year, xcatc_dm, ref = 2001) | g + year,
               data = dd, vcov = ~id, notes = FALSE)
    nd <- dd[dd$.Dtreat, ]
  } else {
    dd$.Dtreat <- dd$year != dd$g - 1L
    m <- feols(y ~ .Dtreat:i(g, i.year, ref = 0) / (xcatb_dm + xcatc_dm) +
                 i(year, xcatb_dm, ref = 2001) + i(year, xcatc_dm, ref = 2001) |
                 g + year,
               data = dd, vcov = ~id, notes = FALSE)
    nd <- dd[dd$g != 0, ]
  }
  emfx_like(m, nd, cg)
}
out$xcat_notyet <- xcat_fit("notyet")
out$xcat_never <- xcat_fit("never")
dfac <- d
dfac$xcat <- factor(dfac$xcat)
fit <- etwfe(fml = y ~ 0, tvar = year, gvar = g, data = dfac, vcov = ~id,
             xvar = xcat)
out$xcat_etwfe_notyet <- list(
  simple = grab(emfx(fit, type = "simple", by_xvar = FALSE), "simple")
)

# --- nonlinear families (fe = "none": explicit cohort and year dummies) ----
for (fam in c("poisson", "logit")) {
  yv <- if (fam == "poisson") "yc" else "yb"
  for (cg in c("notyet", "never")) {
    fml <- as.formula(paste(yv, "~ 0"))
    fit0 <- etwfe(fml = fml, tvar = year, gvar = g, data = d, vcov = ~id,
                  cgroup = cg, family = fam, glm.tol = 1e-12)
    fit <- etwfe(fml = fml, tvar = year, gvar = g, data = d, vcov = ~id,
                 cgroup = cg, family = fam, glm.tol = 1e-12,
                 start = coef(fit0))
    out[[paste0(substr(fam, 1, 4), "_", cg)]] <- list(
      response = all_types(fit),
      link = all_types(fit, predict = "link"),
      ssc = ssc_of(fit)
    )
  }
}
fit0 <- etwfe(fml = yc ~ 0, tvar = year, gvar = g, data = d, vcov = ~id,
              family = "poisson", xvar = xv, glm.tol = 1e-12)
fit <- etwfe(fml = yc ~ 0, tvar = year, gvar = g, data = d, vcov = ~id,
             family = "poisson", xvar = xv, glm.tol = 1e-12,
             start = coef(fit0))
out$pois_xv_notyet <- list(
  response = all_types(fit, by_xvar = FALSE),
  link = all_types(fit, by_xvar = FALSE, predict = "link"),
  ssc = ssc_of(fit)
)

out$versions <- list(
  r = R.version.string,
  etwfe = as.character(packageVersion("etwfe")),
  fixest = as.character(packageVersion("fixest")),
  marginaleffects = as.character(packageVersion("marginaleffects")),
  numDeriv = as.character(packageVersion("numDeriv"))
)
write_json(out, "tests/reference_parity/_fixtures/etwfe_options_R.json",
           digits = I(17), na = "null", auto_unbox = TRUE,
           pretty = TRUE)
cat("wrote etwfe_options_R.json\n")
