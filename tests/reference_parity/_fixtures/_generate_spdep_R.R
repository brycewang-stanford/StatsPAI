#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_spdep_parity.py
#
# Requires: R 4.5 + spdep + spatialreg.
# Run:      Rscript _generate_spdep_R.R    (from this directory)
#
# Weights conventions, which decide every number below
# ----------------------------------------------------
# `sp.W` is BINARY until `w.transform = "R"`. spdep's `nb2listw` defaults to
# style = "W" (row-standardised). Comparing a binary W against a
# row-standardised listw is not a parity test -- the two sides are computing
# different statistics -- so both styles are emitted here and the Python side
# selects the matching one per statistic:
#
#   moran / geary / localmoran / lm.RStests / lm.morantest / lmSLX / sacsarlm
#       -> row-standardised (style = "W")
#   globalG.test / localG / joincount.multi
#       -> binary (style = "B"), which is what spdep itself recommends for
#          the Getis-Ord family
#
# localG is emitted twice: Gi (neighbours only) and Gi* (self included via
# include.self), because StatsPAI's `star=` switches between them and the two
# use different standardisations.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(spdep); library(spatialreg); library(jsonlite)
})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

d <- read.csv(file.path(OUT, "spdep_data.csv"))
n <- nrow(d)
A <- (outer(d$grid_row, d$grid_row, function(a, b) abs(a - b)) +
      outer(d$grid_col, d$grid_col, function(a, b) abs(a - b))) == 1
A <- A * 1
nb <- mat2listw(A, style = "B")$neighbours
lw_W <- nb2listw(nb, style = "W")
lw_B <- nb2listw(nb, style = "B")

out <- list()

mr <- moran.test(d$y, lw_W, randomisation = TRUE)
out$moran <- list(I = unname(mr$estimate[1]), E = unname(mr$estimate[2]),
                  V = unname(mr$estimate[3]), z = unname(mr$statistic),
                  p = unname(mr$p.value))
gr <- geary.test(d$y, lw_W, randomisation = TRUE)
gn <- geary.test(d$y, lw_W, randomisation = FALSE)
out$geary_rand <- list(C = unname(gr$estimate[1]), E = unname(gr$estimate[2]),
                       V = unname(gr$estimate[3]), z = unname(gr$statistic),
                       p = unname(gr$p.value))
out$geary_norm <- list(C = unname(gn$estimate[1]), E = unname(gn$estimate[2]),
                       V = unname(gn$estimate[3]), z = unname(gn$statistic),
                       p = unname(gn$p.value))

gg <- globalG.test(d$pos, lw_B)
out$getis_g <- list(G = unname(gg$estimate[1]), E = unname(gg$estimate[2]),
                    V = unname(gg$estimate[3]), z = unname(gg$statistic))

fx <- factor(d$bin, levels = c(0, 1))
jc <- joincount.multi(fx, lw_B)
out$joincount <- list(WW = unname(jc["0:0", 1]), BB = unname(jc["1:1", 1]),
                      BW = unname(jc["1:0", 1]), S0_half = sum(A) / 2)

lm_ <- localmoran(d$y, lw_W)
out$localmoran <- list(Ii = as.numeric(lm_[, 1]), E = as.numeric(lm_[, 2]),
                       V = as.numeric(lm_[, 3]), Z = as.numeric(lm_[, 4]))
out$localG <- as.numeric(localG(d$pos, lw_B))
out$localGstar <- as.numeric(
  localG(d$pos, nb2listw(include.self(nb), style = "B")))

ols <- lm(y ~ x1 + x2, data = d)
lmt <- lm.RStests(ols, lw_W, test = "all")
out$lm_tests <- lapply(lmt, function(z) list(stat = unname(z$statistic),
                                             df = unname(z$parameter),
                                             p = unname(z$p.value)))
# lm.morantest defaults to alternative = "greater" (one-sided). StatsPAI
# reports a two-sided p, as `moran` does, so both are recorded rather than
# leaving the comparison to guess which convention the number is in.
mt  <- lm.morantest(ols, lw_W)
mt2 <- lm.morantest(ols, lw_W, alternative = "two.sided")
out$moran_resid <- list(I = unname(mt$estimate[1]), E = unname(mt$estimate[2]),
                        V = unname(mt$estimate[3]),
                        z = as.numeric(mt$statistic),
                        p_greater = as.numeric(mt$p.value),
                        p = as.numeric(mt2$p.value))

sx <- lmSLX(y ~ x1 + x2, data = d, listw = lw_W)
out$slx <- list(coef = as.list(coef(sx)))
sc <- sacsarlm(y ~ x1 + x2, data = d, listw = lw_W)
out$sac <- list(rho = unname(sc$rho), lambda = unname(sc$lambda),
                coef = as.list(coef(sc)))
sar <- lagsarlm(y ~ x1 + x2, data = d, listw = lw_W)
imp <- impacts(sar, listw = lw_W)
out$sar <- list(rho = unname(sar$rho), coef = as.list(coef(sar)),
                direct = as.numeric(imp$direct),
                indirect = as.numeric(imp$indirect),
                total = as.numeric(imp$total))

# ---- point-pattern weights ------------------------------------------------
# knn and distance-band constructors are pinned on a separate random point
# set: the lattice above has exact distance ties, which make k-nearest
# neighbour sets non-unique and would test the tie-break rather than the
# constructor.
pts_df <- read.csv(file.path(OUT, "spdep_points.csv"))
pts <- as.matrix(pts_df[, c("cx", "cy")])
kn <- knn2nb(knearneigh(pts, k = 4))
out$knn4 <- lapply(seq_along(kn), function(i) as.integer(kn[[i]]) - 1L)
db <- dnearneigh(pts, 0, 0.25)
out$dband025 <- lapply(seq_along(db), function(i) {
  v <- db[[i]]
  if (length(v) == 1 && v == 0) integer(0) else as.integer(v) - 1L
})

out$provenance <- list(r = R.version.string,
                       spdep = as.character(packageVersion("spdep")),
                       spatialreg = as.character(packageVersion("spatialreg")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           file.path(OUT, "spdep_R.json"))
cat("wrote spdep_R.json\n")
