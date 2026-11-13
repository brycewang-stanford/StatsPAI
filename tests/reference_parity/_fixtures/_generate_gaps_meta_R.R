# metafor reference for sp.meta_analysis(method = 'PM' | 'ML'),
# sp.meta_regression and sp.trim_and_fill.
#
#   Rscript tests/reference_parity/_fixtures/_generate_gaps_meta_R.R
#
# Reads gaps_meta.csv (_generate_gaps_meta_data.py) and probe_epi_meta.csv
# (_generate_probe_epi_data.py); writes gaps_meta_R.json.
# R 4.5.2, metafor 5.0.1 (GPL; run as a black box).
#
# metafor stops its iterative tau^2 estimators at a relative change of 1e-5
# by default. StatsPAI iterates to machine precision, so the iterative fits
# below tighten metafor's control (`ct`). trimfill() refits the model with
# the object's own settings, so the tightened control carries through.
suppressMessages({library(metafor); library(jsonlite)})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))
d <- read.csv(file.path(here, "gaps_meta.csv"))
e <- read.csv(file.path(here, "probe_epi_meta.csv"))
ct <- list(threshold = 1e-14, tol = 1e-14, maxiter = 10000)
ctl <- function(m) if (m %in% c("REML", "ML", "PM")) ct else list()

pooled <- function(r) list(
  tau2 = r$tau2, estimate = as.numeric(r$beta), se = r$se, zval = r$zval,
  pval = r$pval, ci_lb = r$ci.lb, ci_ub = r$ci.ub, QE = r$QE, QEp = r$QEp
)

out <- list()

# --- tau^2 estimators, intercept only -------------------------------------
est <- list()
for (nm in c("gaps", "probe")) {
  dat <- if (nm == "gaps") d else e
  for (m in c("PM", "ML")) {
    est[[paste(nm, m, sep = "_")]] <-
      pooled(rma(yi, sei = sei, data = dat, method = m, control = ct))
    est[[paste(nm, m, "knha", sep = "_")]] <-
      pooled(rma(yi, sei = sei, data = dat, method = m, test = "knha", control = ct))
  }
}
out$estimators <- est

# --- meta-regression -------------------------------------------------------
reg <- function(r) list(
  beta = as.numeric(r$beta), se = r$se, zval = r$zval, pval = r$pval,
  ci_lb = r$ci.lb, ci_ub = r$ci.ub, vb = as.numeric(r$vb),
  tau2 = r$tau2, I2 = r$I2, H2 = r$H2, R2 = r$R2,
  QE = r$QE, QEp = r$QEp, QM = r$QM, QMp = r$QMp,
  QMdf = as.numeric(r$QMdf), k = r$k
)
mr <- list()
for (m in c("REML", "DL", "ML", "PM", "FE")) {
  key <- if (m == "FE") "fixed" else m
  mr[[paste(key, "two", sep = "_")]] <-
    reg(rma(yi, sei = sei, mods = ~ dose + blind, data = d, method = m, control = ctl(m)))
  if (m != "FE") mr[[paste(key, "two_knha", sep = "_")]] <-
    reg(rma(yi, sei = sei, mods = ~ dose + blind, data = d, method = m,
            test = "knha", control = ctl(m)))
  mr[[paste(key, "one", sep = "_")]] <-
    reg(rma(yi, sei = sei, mods = ~ dose, data = d, method = m, control = ctl(m)))
}
# Without an intercept metafor centres yi at its unweighted mean before
# computing QE and tau^2 (QE is then not the weighted residual sum of
# squares of the fit; compare lm(yi ~ dose + blind - 1, weights = 1 / vi)).
# Stored to document that, with the lm() number next to it; the parity
# reference for this case is Stata (see _generate_gaps_meta_Stata.do).
r0 <- rma(yi, sei = sei, mods = ~ dose + blind - 1, data = d, method = "DL")
f0 <- lm(yi ~ dose + blind - 1, data = d, weights = 1 / sei^2)
mr[["DL_noint"]] <- c(reg(r0), list(
  wrss_lm = sum(resid(f0)^2 / d$sei^2), ybar = mean(d$yi)))
r <- rma(yi, sei = sei, mods = ~ dose + blind, data = d, method = "REML", control = ct)
pr <- predict(r, newmods = cbind(c(2, 5, 9), c(0, 1, 1)))
mr[["REML_two_predict"]] <- list(pred = as.numeric(pr$pred), se = as.numeric(pr$se),
                                 ci_lb = as.numeric(pr$ci.lb), ci_ub = as.numeric(pr$ci.ub))
out$meta_regression <- mr

# --- trim and fill ---------------------------------------------------------
tf <- list()
fill <- function(r, ...) {
  # Q0 is undefined (square root of a negative number) when every centred
  # effect is positive; metafor stops with an error there, recorded as such.
  t <- tryCatch(suppressWarnings(trimfill(r, ...)), error = function(err) NULL)
  if (is.null(t)) return(list(error = TRUE))
  # trimfill() refits with metafor's default stopping rule whatever the
  # control of `r`, so for the iterative estimators its pooled effect
  # carries a relative error of about 1e-5 to 1e-4. `tight` is the same
  # model on the same augmented studies run to convergence.
  m <- r$method
  tight <- rma(as.numeric(t$yi), as.numeric(t$vi), method = m, control = ctl(m))
  list(k0 = t$k0, side = t$side, estimate = as.numeric(t$beta), se = t$se,
       ci_lb = t$ci.lb, ci_ub = t$ci.ub, pval = t$pval, tau2 = t$tau2,
       k = t$k, yi_fill = as.numeric(t$yi[t$fill]), sei_fill = sqrt(as.numeric(t$vi[t$fill])),
       estimate_tight = as.numeric(tight$beta), se_tight = tight$se,
       tau2_tight = tight$tau2)
}
for (nm in c("gaps", "probe")) {
  dat <- if (nm == "gaps") d else e
  for (m in c("DL", "REML", "PM", "ML", "FE")) {
    key <- if (m == "FE") "fixed" else m
    r <- rma(yi, sei = sei, data = dat, method = m, control = ctl(m))
    for (est in c("L0", "R0", "Q0")) {
      tf[[paste(nm, key, est, "auto", sep = "_")]] <- fill(r, estimator = est)
      for (sd in c("left", "right")) {
        tf[[paste(nm, key, est, sd, sep = "_")]] <- fill(r, estimator = est, side = sd)
      }
    }
  }
}
out$trimfill <- tf
out[["_meta"]] <- list(R = R.version.string,
                       metafor = as.character(packageVersion("metafor")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null"),
           file.path(here, "gaps_meta_R.json"))
cat("wrote gaps_meta_R.json\n")
