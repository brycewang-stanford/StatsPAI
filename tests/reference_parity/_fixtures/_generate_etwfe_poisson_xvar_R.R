# Reference for sp.etwfe(family='poisson', fe='cohort', xvar=...) -- R etwfe's
# own design -- on _fixtures/etwfe_poisson_jwdid_hettype.csv (the panel of
# test_etwfe_poisson_jwdid_parity.py): a categorical covariate xcat (factor)
# and a time-varying continuous covariate xc (etwfe takes a single xvar).
#
# R 4.5, etwfe 0.6.2, fixest 0.14.0, marginaleffects 0.32.0. Run from the
# repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_etwfe_poisson_xvar_R.R
suppressPackageStartupMessages({
  library(etwfe); library(jsonlite)
})
d <- read.csv("tests/reference_parity/_fixtures/etwfe_poisson_jwdid_hettype.csv")
d$xcat <- factor(d$xcat)

grab <- function(e, by = NULL) {
  e <- as.data.frame(e)
  out <- list(estimate = e$estimate, std.error = e$std.error)
  if (!is.null(by) && by %in% names(e)) out$by <- as.character(e[[by]])
  if ("event" %in% names(e)) out$event <- e$event
  out
}
specs <- list(
  xcat    = list(xvar = "xcat", cgroup = "notyet"),
  xc      = list(xvar = "xc",   cgroup = "notyet"),
  xcat_never = list(xvar = "xcat", cgroup = "never")
)
res <- list()
for (nm in names(specs)) {
  s <- specs[[nm]]
  m <- etwfe(fml = y ~ 1, tvar = year, gvar = g, data = d, xvar = s$xvar,
             cgroup = s$cgroup, vcov = ~id, family = "poisson")
  message(nm, ": collinear = ", paste(m$collin.var, collapse = ", "))
  r <- list(
    simple_response = grab(emfx(m, type = "simple", by_xvar = FALSE)),
    simple_link = grab(emfx(m, type = "simple", by_xvar = FALSE, predict = "link")),
    event_response = grab(emfx(m, type = "event", by_xvar = FALSE)),
    event_link = grab(emfx(m, type = "event", by_xvar = FALSE, predict = "link"))
  )
  if (identical(s$xvar, "xcat")) {
    r$by_response <- grab(emfx(m, type = "simple", by_xvar = TRUE), "xcat")
    r$by_link <- grab(emfx(m, type = "simple", by_xvar = TRUE, predict = "link"), "xcat")
  }
  res[[nm]] <- r
}
res$versions <- paste0("R ", getRversion(), "; etwfe ", packageVersion("etwfe"),
                       "; fixest ", packageVersion("fixest"),
                       "; marginaleffects ", packageVersion("marginaleffects"))
write_json(res, "tests/reference_parity/_fixtures/etwfe_poisson_xvar_R.json",
           digits = NA, auto_unbox = TRUE, pretty = TRUE)

# ---------------------------------------------------------------------------
# etwfe 0.6.2 with a factor xvar writes i(year, (xcat2_dm + xcat3_dm)): fixest
# reads the parenthesised sum as ONE covariate, so every level shares a
# single period slope.  The cell interactions are per level
# (.Dtreat:...:xcat2_dm, :xcat3_dm).  The per-level design StatsPAI fits is
# etwfe's with i(year, xcat2_dm) + i(year, xcat3_dm); here it is fitted with
# fixest directly and aggregated from counterfactual predictions
# (.Dtreat = FALSE), the quantity emfx computes.
# ---------------------------------------------------------------------------
perlevel <- function(cgroup) {
  dd <- d
  dd$.ct <- interaction(dd$g, dd$year, drop = TRUE)
  X <- model.matrix(~ xcat, dd)[, -1, drop = FALSE]
  for (j in seq_len(ncol(X))) {
    nm <- paste0("xcat", j + 1, "_dm")
    dd[[nm]] <- X[, j] - ave(X[, j], dd$.ct)
  }
  if (cgroup == "notyet") {
    dd$.Dtreat <- dd$year >= dd$g & dd$g != 0
    fml <- y ~ .Dtreat:i(g, i.year, ref = 0, ref2 = 2001) / (xcat2_dm + xcat3_dm) +
      i(year, xcat2_dm, ref = 2001) + i(year, xcat3_dm, ref = 2001) | g + year
    treated <- dd$.Dtreat
  } else {
    dd$.Dtreat <- dd$year != dd$g - 1L
    fml <- y ~ .Dtreat:i(g, i.year, ref = 0) / (xcat2_dm + xcat3_dm) +
      i(year, xcat2_dm, ref = 2001) + i(year, xcat3_dm, ref = 2001) | g + year
    treated <- dd$g != 0 & dd$year >= dd$g
  }
  m <- fixest::feglm(fml, data = dd, family = "poisson", vcov = ~id, notes = FALSE)
  d0 <- dd; d0$.Dtreat <- FALSE
  eta1 <- predict(m, newdata = dd, type = "link")
  eta0 <- predict(m, newdata = d0, type = "link")
  mu1 <- exp(eta1); mu0 <- exp(eta0)
  lev <- levels(dd$xcat)
  list(
    simple_link = mean((eta1 - eta0)[treated]),
    simple_response = mean((mu1 - mu0)[treated]),
    by = lev,
    by_link = sapply(lev, function(l) mean((eta1 - eta0)[treated & dd$xcat == l])),
    by_response = sapply(lev, function(l) mean((mu1 - mu0)[treated & dd$xcat == l]))
  )
}
res$xcat_perlevel <- perlevel("notyet")
res$xcat_never_perlevel <- perlevel("never")
write_json(res, "tests/reference_parity/_fixtures/etwfe_poisson_xvar_R.json",
           digits = NA, auto_unbox = TRUE, pretty = TRUE)
