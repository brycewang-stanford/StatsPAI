#!/usr/bin/env Rscript
# Reference for sp.dag(...).implied_independencies() / .test_implications()
# / .adjustment_sets() with named latents, from dagitty.
#
#   Rscript tests/reference_parity/_generate_dag_implications.R
#
# Writes tests/reference_parity/_fixtures/dag_implications.csv (simulated
# linear SEM on the eight-node graph) and dag_implications_R.json.

suppressMessages({
  library(dagitty)
  library(jsonlite)
})
FIX <- "tests/reference_parity/_fixtures"
spec <- "Z1 -> X1; X1 -> D; Z1 -> X2; Z2 -> X3; X3 -> Y; Z2 -> X2; X2 -> Y; X2 -> D; M -> Y; D -> M"
g <- dagitty(paste0("dag{", spec, "}"))
set.seed(20261004)
n <- 2000
Z1 <- rnorm(n); Z2 <- rnorm(n)
X1 <- Z1 + rnorm(n); X2 <- Z1 + Z2 + rnorm(n); X3 <- Z2 + rnorm(n)
D <- X1 + X2 + rnorm(n); M <- D + rnorm(n); Y <- M + X2 + X3 + rnorm(n)
dat <- data.frame(Z1, Z2, X1, X2, X3, D, M, Y)
write.csv(dat, file.path(FIX, "dag_implications.csv"), row.names = FALSE)
dat <- read.csv(file.path(FIX, "dag_implications.csv"))

ci <- impliedConditionalIndependencies(g)
lt <- localTests(g, dat, type = "cis")
tests <- lapply(seq_along(ci), function(i) list(
  x = ci[[i]]$X, y = ci[[i]]$Y, given = as.list(sort(as.character(unlist(ci[[i]]$Z)))),
  estimate = lt$estimate[i], p_value = lt$p.value[i]))

sets <- function(a) lapply(a, function(z) as.list(sort(as.character(unlist(z)))))
latent_cases <- list(
  list(spec = "D -> Y; X -> D; F -> X; F -> D; X -> Y", latent = list("F")),
  list(spec = "D -> Y; X -> D; F -> D; A -> F; A -> X; A -> D; X -> Y", latent = list("F")),
  list(spec = "D -> Y; X -> D; F -> D; A -> F; A -> X; A -> D; F -> Y; X -> Y", latent = list("F", "A")),
  list(spec = "D -> Y; X -> D; F -> D; A -> F; A -> X; A -> D; D -> M; M -> Y; X -> M; X -> Y", latent = list("F", "A", "M")),
  list(spec = "D -> Y; X -> D; F -> D; A -> F; A -> X; A -> D; D -> M; F -> M; M -> Y; X -> M; X -> Y", latent = list("F", "A", "M")))
for (i in seq_along(latent_cases)) {
  cs <- latent_cases[[i]]
  gl <- dagitty(paste0("dag{", cs$spec, "; ", paste(unlist(cs$latent), "[latent]", collapse = "; "), "}"))
  latent_cases[[i]]$minimal <- sets(adjustmentSets(gl, "D", "Y"))
  latent_cases[[i]]$n_implied <- length(impliedConditionalIndependencies(gl))
}
out <- list(dagitty_version = as.character(packageVersion("dagitty")), spec = spec,
            minimal_adjustment = sets(adjustmentSets(g, "D", "Y")),
            all_adjustment = sets(adjustmentSets(g, "D", "Y", type = "all")),
            tests = tests, latent_cases = latent_cases)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(FIX, "dag_implications_R.json"))
cat("wrote dag_implications fixtures\n")
