#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for sp.panel_qtet — Callaway & Li (2019) panel QTT.
#
#  Writes qte_panel_qtet_nocov_R.json.
#
#  Two things this script does that matter:
#
#  1. It calls qte::panel.qtet with NO covariates, which forces the
#     package's `method = "pscore"` branch — the clean copula construction
#     the Python port implements. (With xformla supplied, `x` is non-NULL
#     and a different, covariate-adjusted path runs.)
#
#  2. It ALSO re-implements the five algorithm steps by hand and asserts
#     the two agree. That is what licenses the Python port to claim an
#     exact algorithmic match rather than a tolerance: if the hand
#     replication drifts from the package, this script fails loudly here
#     rather than silently shipping a mismatched fixture.
#
#  TRAP: t / tmin1 / tmin2 are VALUES of the tname column, not indices.
#  The panel is 1974 / 1975 / 1978, so t=1978, tmin1=1975, tmin2=1974.
#  Swapping them does not error — it returns a different contrast.
#
#  Environment: R 4.5.2 / qte 1.3.1
# =====================================================================
suppressPackageStartupMessages({library(qte); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

data(lalonde)
PROBS <- seq(0.05, 0.95, 0.05)
d <- lalonde.psid.panel

r <- panel.qtet(re ~ treat, t = 1978, tmin1 = 1975, tmin2 = 1974,
                tname = "year", data = d, idname = "id",
                probs = PROBS, se = FALSE)

# ---- hand replication of the no-covariate branch --------------------
d <- d[order(d$id, d$year), ]
tr <- d[d$treat == 1, ]; un <- d[d$treat == 0, ]
t_t  <- tr[tr$year == 1978, ]; t_m1 <- tr[tr$year == 1975, ]
t_m2 <- tr[tr$year == 1974, ]
u_t  <- un[un$year == 1978, ]; u_m1 <- un[un$year == 1975, ]
t_t <- t_t[order(t_t$id), ]; t_m1 <- t_m1[order(t_m1$id), ]
t_m2 <- t_m2[order(t_m2$id), ]
u_t <- u_t[order(u_t$id), ]; u_m1 <- u_m1[order(u_m1$id), ]

F_t_m1 <- ecdf(t_m1$re); F_t_m2 <- ecdf(t_m2$re)
F_u_dt <- ecdf(u_t$re - u_m1$re)       # untreated change, period t
F_t_dm1 <- ecdf(t_m1$re - t_m2$re)     # treated change, period t-1

q1 <- quantile(F_t_m1, probs = F_t_m2(t_m2$re))
q2 <- quantile(F_u_dt, probs = F_t_dm1(t_m1$re - t_m2$re))
cf <- q1 + q2
manual_qte <- quantile(t_t$re, PROBS) - quantile(cf, PROBS)

gap <- max(abs(as.numeric(r$qte) - as.numeric(manual_qte)))
cat("[panel-qtet-fixture] max |pkg - manual| =", gap, "\n")
if (gap > 1e-9) stop("hand replication no longer matches qte::panel.qtet")

out <- list(
  probs = PROBS,
  pkg_qte = as.numeric(r$qte),
  pkg_ate = as.numeric(r$ate),
  manual_qte = as.numeric(manual_qte),
  cf_sample = as.numeric(sort(cf)),
  treated_t_y = as.numeric(sort(t_t$re)),
  n_treated = nrow(t_t),
  n_untreated = nrow(u_t),
  r_version = R.version.string,
  qte_version = as.character(packageVersion("qte"))
)
write(toJSON(out, digits = 12, auto_unbox = TRUE),
      file.path(OUT, "qte_panel_qtet_nocov_R.json"))
cat("[panel-qtet-fixture] wrote qte_panel_qtet_nocov_R.json\n")
