# R paths 0.1.2 (Zhou & Yamamoto; GPL (>= 2), used as a black box: outputs
# only) reference for sp.mediate_paths on mediate_paths.csv. Install paths
# into a private library, e.g.
#   R_LIBS_USER=/tmp/rlib Rscript -e 'install.packages("paths", lib="/tmp/rlib")'
#   cd tests/reference_parity/_fixtures && R_LIBS_USER=/tmp/rlib Rscript _generate_mediate_paths_R.R
# Only the point estimates are compared (the bootstrap is stochastic);
# nboot = 2 keeps the run short.
suppressMessages(library(paths))
d <- read.csv("mediate_paths.csv")
ps <- glm(a ~ x1 + x2, family = binomial, data = d)
run <- function(forms, m, family = NULL) {
  # do.call writes the formula itself into each model's call, which paths
  # re-evaluates when it bootstraps
  mods <- lapply(forms, function(f) {
    if (is.null(family)) do.call("lm", list(formula = f, data = quote(d)))
    else do.call("glm", list(formula = f, family = family, data = quote(d)))
  })
  set.seed(1)
  p <- suppressWarnings(paths(a = "a", y = all.vars(forms[[1]])[1], m = m,
                              models = mods, ps_model = ps, data = d, nboot = 2))
  out <- list()
  for (est in c("pure", "hybrid")) {
    tab <- p[[est]]
    for (dec in c("Type I", "Type II")) {
      sub <- tab[tab$decomposition == dec, ]
      key <- paste(est, if (dec == "Type I") "type1" else "type2", sep = "_")
      out[[key]] <- as.list(setNames(sub$estimate, gsub(" ", "_", sub$estimand)))
    }
  }
  out
}
res <- list(
  inter = run(list(y ~ a + x1 + x2, y ~ a * m1 + x1 + x2,
                   y ~ a * m1 + a * m2 + x1 + x2), list("m1", "m2")),
  additive = run(list(y ~ a + x1 + x2, y ~ a + m1 + x1 + x2,
                      y ~ a + m1 + m2 + x1 + x2), list("m1", "m2")),
  blocks = run(list(y ~ a + x1 + x2, y ~ a * m1 + a * m1b + x1 + x2,
                    y ~ a * m1 + a * m1b + a * m2 + x1 + x2),
               list(c("m1", "m1b"), "m2")),
  single = run(list(y ~ a + x1 + x2, y ~ a * m2 + x1 + x2), list("m2")),
  binary = run(list(yb ~ a + x1 + x2, yb ~ a * m1 + x1 + x2,
                    yb ~ a * m1 + a * m2 + x1 + x2), list("m1", "m2"),
               family = binomial)
)
writeLines(jsonlite::toJSON(res, digits = I(17), auto_unbox = TRUE, pretty = TRUE),
           "mediate_paths_R.json")
