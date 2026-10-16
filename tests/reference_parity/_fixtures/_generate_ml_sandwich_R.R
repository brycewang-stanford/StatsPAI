# Reference SEs for tests/reference_parity/test_ml_sandwich_r_parity.py.
# R 4.5, sandwich. Run from this directory:
#   Rscript _generate_ml_sandwich_R.R
suppressPackageStartupMessages({library(sandwich); library(jsonlite)})
d <- read.csv("ml_sandwich_data.csv")
ctl <- glm.control(epsilon = 1e-14, maxit = 200)
out <- list(r_version = R.version.string,
            sandwich_version = as.character(packageVersion("sandwich")))
fits <- list(
  logit = list(f = yb ~ x1 + x2, fam = binomial("logit")),
  probit = list(f = yb ~ x1 + x2, fam = binomial("probit")),
  poisson = list(f = c ~ x1 + x2, fam = poisson("log"))
)
for (nm in names(fits)) {
  for (wt in c("none", "w")) {
    spec <- fits[[nm]]
    m <- if (wt == "none") glm(spec$f, family = spec$fam, data = d, control = ctl)
         else suppressWarnings(glm(spec$f, family = spec$fam, data = d,
                                   weights = w, control = ctl))
    entry <- list(coef = unname(coef(m)), se_model = unname(sqrt(diag(vcov(m)))))
    for (type in c("HC0", "HC1", "HC2", "HC3")) {
      entry[[type]] <- unname(sqrt(diag(vcovHC(m, type = type))))
    }
    out[[paste(nm, wt, sep = "_")]] <- entry
  }
}
m <- glm(c ~ x1 + x2 + offset(log(expo)), family = poisson, data = d, control = ctl)
out$poisson_exposure <- list(coef = unname(coef(m)),
                             se_model = unname(sqrt(diag(vcov(m)))),
                             HC0 = unname(sqrt(diag(vcovHC(m, type = "HC0")))))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "ml_sandwich_R.json")
