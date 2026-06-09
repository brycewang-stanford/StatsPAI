#!/usr/bin/env Rscript
# R reference values for sp count / quantile / Tobit parity.
#
#   * sp.poisson -> stats::glm(family = poisson)        [model-based SE]
#   * sp.nbreg   -> MASS::glm.nb  (NB2; sp alpha = 1/theta)
#   * sp.qreg    -> quantreg::rq  (tau = 0.5, 0.75; unique solutions)
#   * sp.tobit   -> AER::tobit(left = 0)  (Gaussian survreg; scale = sigma)
#
# zip / zinb (pscl) are added by the companion *_pscl.R generator only when
# pscl is installed; their absence does not block this fixture.
suppressMessages({
  library(MASS)
  library(quantreg)
  library(AER)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/count_quantile_data.csv")

pack <- function(co, se, r_names, py_names) {
  out <- list()
  for (i in seq_along(r_names)) {
    out[[py_names[i]]] <- list(coef = unname(co[r_names[i]]),
                               se = if (is.null(se)) NULL else unname(se[r_names[i]]))
  }
  out
}
r_nm <- c("(Intercept)", "x1", "x2")
py_nm <- c("_cons", "x1", "x2")

pm <- glm(yc ~ x1 + x2, family = poisson, data = df)
nb <- glm.nb(yc ~ x1 + x2, data = df)
q50 <- rq(yl ~ x1 + x2, tau = 0.50, data = df)
q75 <- rq(yl ~ x1 + x2, tau = 0.75, data = df)
tb <- tobit(yt ~ x1 + x2, left = 0, data = df)

out <- list(
  meta = list(
    R_version = R.version.string,
    MASS_version = as.character(packageVersion("MASS")),
    quantreg_version = as.character(packageVersion("quantreg")),
    AER_version = as.character(packageVersion("AER")),
    n_obs = nrow(df)
  ),
  poisson = pack(coef(pm), sqrt(diag(vcov(pm))), r_nm, py_nm),
  nbreg = c(
    pack(coef(nb), sqrt(diag(vcov(nb))), r_nm, py_nm),
    list(theta = unname(nb$theta), alpha = unname(1 / nb$theta))
  ),
  qreg_tau50 = pack(coef(q50), NULL, c("(Intercept)", "x1", "x2"),
                    c("const", "x1", "x2")),
  qreg_tau75 = pack(coef(q75), NULL, c("(Intercept)", "x1", "x2"),
                    c("const", "x1", "x2")),
  tobit = c(
    pack(coef(tb), sqrt(diag(vcov(tb)))[1:3], r_nm, c("const", "x1", "x2")),
    list(sigma = unname(tb$scale))
  )
)
write_json(out, "tests/reference_parity/_fixtures/count_quantile_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("poisson x1: %.6f (se %.6f)\n", coef(pm)["x1"], sqrt(vcov(pm)["x1","x1"])))
cat(sprintf("nbreg   x1: %.6f  theta=%.6f alpha=%.6f\n",
            coef(nb)["x1"], nb$theta, 1 / nb$theta))
cat(sprintf("qreg50  x1: %.6f   qreg75 x1: %.6f\n", coef(q50)["x1"], coef(q75)["x1"]))
cat(sprintf("tobit   x1: %.6f  sigma=%.6f\n", coef(tb)["x1"], tb$scale))
