# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_prodest_parity.py.
#
# prodest (Rovigatti & Mollisi) on prodest_panel.csv, written by
# _generate_prodest_data.py. Run from this directory:
#
#   Rscript _generate_prodest_R.R
#
# Writes prodest_R.json. prodest's optimisers start from the first-stage
# coefficients plus rnorm(0, 0.01) noise and stop at optim()'s default
# BFGS tolerance, so the seed is fixed and, next to each estimate, the
# script records the value of prodest's own criterion there (gOPLP / gACF
# evaluated on prodest's own data objects). The test uses those values to
# show where the R optimiser stopped relative to the minimiser.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(prodest))
library(jsonlite)

d <- read.csv("prodest_panel.csv")
set.seed(20260913)

crit_oplp <- function(fit, d, theta) {
  # prodest:::finalOPLP's stage-2 data, rebuilt with prodest's own helpers.
  sX <- as.matrix(d$k); fX <- as.matrix(d$l)
  fs <- fit@Model$FSbetas
  regvars <- cbind(fX, model.matrix(~.^2 - 1, data = data.frame(sX, pX = fit@Data$proxy)),
                   sX^2, fit@Data$proxy^2)
  phi <- as.vector(cbind(1, regvars) %*% ifelse(is.na(fs), 0, fs)) - as.vector(fX * fs[2])
  lag.phi <- prodest:::lagPanel(d$id, d$year, phi)
  lag.sX <- prodest:::lagPanel(d$id, d$year, d$k)
  res <- d$y - as.vector(fX * fs[2])
  ok <- !is.na(lag.phi) & !is.na(lag.sX)
  as.numeric(prodest:::gOPLP(theta, as.matrix(d$k[ok]), as.matrix(lag.sX[ok]), phi[ok],
                             lag.phi[ok], res[ok], 1e-100, 0, FALSE))
}

out <- list()

op <- prodestOP(d$y, fX = d$l, sX = d$k, pX = d$lninv, idvar = d$id, timevar = d$year, R = 2)
lp <- prodestLP(d$y, fX = d$l, sX = d$k, pX = d$m, idvar = d$id, timevar = d$year, R = 2)
for (nm in c("op", "lp")) {
  fit <- get(nm)
  out[[nm]] <- list(
    pars = unname(fit@Estimates$pars),
    fs_betas = unname(fit@Model$FSbetas),
    opt_value = fit@Model$opt.outcome$value,
    crit_at_pars = crit_oplp(fit, d, fit@Estimates$pars[2]),
    convergence = fit@Model$opt.outcome$convergence
  )
}

acf <- prodestACF(d$y, fX = d$l, sX = d$k, pX = d$m, idvar = d$id, timevar = d$year, R = 2)
out$acf <- list(
  pars = unname(acf@Estimates$pars),
  opt_value = acf@Model$opt.outcome$value,
  convergence = acf@Model$opt.outcome$convergence
)

wrdg <- prodestWRDG(d$y, fX = d$l, sX = d$k, pX = d$m, idvar = d$id, timevar = d$year)
out$wrdg <- list(
  pars = unname(wrdg@Estimates$pars),
  se = unname(wrdg@Estimates$std.errors)
)

out$`_meta` <- list(
  generator = "_generate_prodest_R.R",
  prodest = as.character(packageVersion("prodest")),
  R = R.version.string,
  data = "prodest_panel.csv",
  seed = 20260913
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "prodest_R.json")
cat("wrote prodest_R.json\n")
