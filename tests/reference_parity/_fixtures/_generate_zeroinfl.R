#!/usr/bin/env Rscript
# R pscl reference values for sp zero-inflated count parity.
#
#   * sp.zip_model -> pscl::zeroinfl(dist = "poisson")
#   * sp.zinb      -> pscl::zeroinfl(dist = "negbin")  (sp ln_alpha;
#                     theta = 1 / exp(ln_alpha))
#
# Requires pscl.  The frozen JSON it writes is what the test consumes, so
# pscl is NOT needed at test time.
suppressMessages({
  library(pscl)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/zeroinfl_data.csv")

# pscl term order: count_(Intercept), count_x1, count_x2, zero_(Intercept),
# zero_z. Map to sp labels: const, x1, x2, inflate_const, inflate_z.
r_terms <- c("count_(Intercept)", "count_x1", "count_x2",
             "zero_(Intercept)", "zero_z")
sp_terms <- c("const", "x1", "x2", "inflate_const", "inflate_z")

pack <- function(co, se) {
  out <- list()
  for (i in seq_along(r_terms)) {
    out[[sp_terms[i]]] <- list(coef = unname(co[r_terms[i]]),
                               se = unname(se[r_terms[i]]))
  }
  out
}

zp <- zeroinfl(y ~ x1 + x2 | z, data = df, dist = "poisson")
zn <- zeroinfl(y ~ x1 + x2 | z, data = df, dist = "negbin")

out <- list(
  meta = list(
    R_version = R.version.string,
    pscl_version = as.character(packageVersion("pscl")),
    n_obs = nrow(df)
  ),
  zip = pack(coef(zp), sqrt(diag(vcov(zp)))),
  zinb = c(
    pack(coef(zn), sqrt(diag(vcov(zn)))),
    list(theta = unname(zn$theta), alpha = unname(1 / zn$theta))
  )
)
write_json(out, "tests/reference_parity/_fixtures/zeroinfl_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("ZIP  x1: %.6f (se %.6f)\n", coef(zp)["count_x1"],
            sqrt(vcov(zp)["count_x1", "count_x1"])))
cat(sprintf("ZINB x1: %.6f  theta=%.6f\n", coef(zn)["count_x1"], zn$theta))
