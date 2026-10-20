# Black-box fixture for the clean-room DR-DiD engines (docs/dev/cleanroom/did_drdid.md).
#
# R DRDID is GPL-3: only its outputs are used. For every estimator, design and
# weighting this records ATT, se and five projections mean(IF * v) of the
# influence function att.inf.func, which pin the whole vector in five
# directions without committing it.
#
#   Rscript _generate_drdid_blackbox_R.R    (from this directory; DRDID 1.3.0)
suppressMessages(library(DRDID))
rows <- list()
put <- function(design, weighting, fn, r, v) {
  inf <- as.numeric(r$att.inf.func)
  rows[[length(rows) + 1]] <<- data.frame(
    design = design, weighting = weighting, fn = fn, att = r$ATT, se = r$se,
    m_one = mean(inf), m_x1 = mean(inf * v$x1), m_x2 = mean(inf * v$x2),
    m_d = mean(inf * v$D), m_y = mean(inf * v$y))
}
for (s in 1:3) {
  p <- read.csv(sprintf("drdid_blackbox_panel_%d.csv", s))
  X <- cbind(1, p$x1, p$x2)
  v <- list(x1 = p$x1, x2 = p$x2, D = p$D, y = p$y1 - p$y0)
  for (wt in c("none", "w")) {
    W <- if (wt == "w") p$w else NULL
    dsg <- sprintf("panel_%d", s)
    put(dsg, wt, "drdid_panel", drdid_panel(p$y1, p$y0, p$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "drdid_imp_panel", drdid_imp_panel(p$y1, p$y0, p$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "std_ipw_did_panel", std_ipw_did_panel(p$y1, p$y0, p$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "ipw_did_panel", ipw_did_panel(p$y1, p$y0, p$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "reg_did_panel", reg_did_panel(p$y1, p$y0, p$D, X, i.weights = W, inffunc = TRUE), v)
  }
  r <- read.csv(sprintf("drdid_blackbox_rc_%d.csv", s))
  X <- cbind(1, r$x1, r$x2)
  v <- list(x1 = r$x1, x2 = r$x2, D = r$D, y = r$y)
  for (wt in c("none", "w")) {
    W <- if (wt == "w") r$w else NULL
    dsg <- sprintf("rc_%d", s)
    put(dsg, wt, "drdid_rc", drdid_rc(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "drdid_rc1", drdid_rc1(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "drdid_imp_rc", drdid_imp_rc(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "drdid_imp_rc1", drdid_imp_rc1(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "std_ipw_did_rc", std_ipw_did_rc(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "ipw_did_rc", ipw_did_rc(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "reg_did_rc", reg_did_rc(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
    put(dsg, wt, "twfe_did_rc", twfe_did_rc(r$y, r$post, r$D, cbind(r$x1, r$x2), i.weights = W, inffunc = TRUE), v)
  }
}
# Design 4: x2 is constant within two cells. Only the improved estimators
# accept it (they drop the column there); the others stop with an error.
r <- read.csv("drdid_blackbox_rc_4.csv")
X <- cbind(1, r$x1, r$x2)
v <- list(x1 = r$x1, x2 = r$x2, D = r$D, y = r$y)
for (wt in c("none", "w")) {
  W <- if (wt == "w") r$w else NULL
  put("rc_4", wt, "drdid_imp_rc", drdid_imp_rc(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
  put("rc_4", wt, "drdid_imp_rc1", drdid_imp_rc1(r$y, r$post, r$D, X, i.weights = W, inffunc = TRUE), v)
  for (fn in c("drdid_rc", "reg_did_rc")) {
    err <- tryCatch({ get(fn)(r$y, r$post, r$D, X, i.weights = W); "" },
                    error = function(e) "error")
    stopifnot(err == "error")
  }
}
out <- do.call(rbind, rows)
out$r_version <- paste(R.version$major, R.version$minor, sep = ".")
out$drdid_version <- as.character(packageVersion("DRDID"))
write.csv(out, "drdid_blackbox_R.csv", row.names = FALSE)
