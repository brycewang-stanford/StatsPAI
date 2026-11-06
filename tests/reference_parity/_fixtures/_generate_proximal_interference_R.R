#!/usr/bin/env Rscript
# R reference values for the proximal and interference parity tests.
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_proximal_interference_R.R
# Writes proximal_interference_R.json next to this script.
#
# ivreg, sandwich and lmtest are GPL and are run as black boxes; nothing in
# StatsPAI was written from their source. Every reference quantity below is
# computed by the package from the committed CSVs; nothing is rebuilt by hand
# except the network matrices W y, W x, W^2 x, W^3 x of the peer-effects
# model, which are formed here from the edge list (row-normalised adjacency,
# rows of isolates left at zero).

suppressPackageStartupMessages({
  library(ivreg)
  library(sandwich)
  library(jsonlite)
})

fx <- "tests/reference_parity/_fixtures"
rd <- function(f) read.csv(file.path(fx, f))

coefse <- function(fit, V, term) {
  list(estimate = unname(coef(fit)[term]), se = unname(sqrt(diag(V))[term]))
}

out <- list()

# ------------------------------------------------------------- proximal
P <- rd("proximal_ref_data.csv")

# sp.proximal: 2SLS of y on (1, d, X, W) with instruments (1, d, X, Z);
# classical covariance sigma^2 (X_hat'X_hat)^-1, sigma^2 = RSS / (n - k).
f1 <- ivreg(y ~ d + x1 + x2 + w1 | d + x1 + x2 + z1, data = P)
f2 <- ivreg(y ~ d + x1 + w1 | d + x1 + z1 + z2, data = P)
f3 <- ivreg(y ~ d + x1 + x2 + w1 + w2 | d + x1 + x2 + z1 + z2, data = P)
f4 <- ivreg(y ~ d + w1 | d + z1, data = P)
out$proximal <- list(
  just_identified = coefse(f1, vcov(f1), "d"),
  over_identified = coefse(f2, vcov(f2), "d"),
  two_w = coefse(f3, vcov(f3), "d"),
  no_covariates = coefse(f4, vcov(f4), "d")
)

# sp.negative_control_outcome / _exposure: OLS with HC1.
o1 <- lm(nco ~ d + x1 + x2, data = P)
o2 <- lm(nco ~ d, data = P)
e1 <- lm(y ~ nce + x1, data = P)
e2 <- lm(y ~ nce, data = P)
out$nco <- list(
  covariates = coefse(o1, vcovHC(o1, type = "HC1"), "d"),
  bare = coefse(o2, vcovHC(o2, type = "HC1"), "d")
)
out$nce <- list(
  covariates = coefse(e1, vcovHC(e1, type = "HC1"), "nce"),
  bare = coefse(e2, vcovHC(e2, type = "HC1"), "nce")
)

# sp.double_negative_control: 2SLS of y on (d, nco, X) with (d, nce, X),
# HC1 sandwich.
g1 <- ivreg(y ~ d + nco + x1 + x2 | d + nce + x1 + x2, data = P)
g2 <- ivreg(y ~ d + nco | d + nce, data = P)
out$dnc <- list(
  covariates = coefse(g1, vcovHC(g1, type = "HC1"), "d"),
  bare = coefse(g2, vcovHC(g2, type = "HC1"), "d")
)

# --------------------------------------------------------- interference
U <- rd("interference_unit_data.csv")
# sp.inward_outward_spillover: OLS, classical covariance.
i1 <- lm(y ~ d + e_in + e_out + x, data = U)
i2 <- lm(y ~ d + e_in + e_out, data = U)
out$inward_outward <- list(
  covariates = list(inward = coefse(i1, vcov(i1), "e_in"),
                    outward = coefse(i1, vcov(i1), "e_out")),
  bare = list(inward = coefse(i2, vcov(i2), "e_in"),
              outward = coefse(i2, vcov(i2), "e_out"))
)

C <- rd("interference_cluster_data.csv")
# sp.cluster_cross_interference: OLS of cluster means on the cluster's
# treatment and neighbour share, classical covariance.
cm <- aggregate(cbind(y, treat, nshare, pair) ~ cluster, data = C, FUN = mean)
c1 <- lm(y ~ treat + nshare, data = cm)
out$cluster_cross <- list(
  direct = coefse(c1, vcov(c1), "treat"),
  spillover = coefse(c1, vcov(c1), "nshare"),
  n_clusters = nrow(cm)
)

# sp.cluster_matched_pair: paired t-test on the cluster means.
cm <- cm[order(cm$pair, -cm$treat), ]
yt <- cm$y[cm$treat == 1][order(cm$pair[cm$treat == 1])]
yc <- cm$y[cm$treat == 0][order(cm$pair[cm$treat == 0])]
tt <- t.test(yt, yc, paired = TRUE)
out$matched_pair <- list(
  estimate = unname(tt$estimate), se = unname(tt$stderr),
  n_pairs = length(yt)
)

# sp.peer_effects: 2SLS with instruments (1, X, WX, W^2X, W^3X), HC0.
N <- rd("peer_network_data.csv")
E <- rd("peer_network_edges.csv")
n <- nrow(N)
A <- matrix(0, n, n)
A[cbind(E$from, E$to)] <- 1
A[cbind(E$to, E$from)] <- 1
rs <- rowSums(A)
W <- A / ifelse(rs == 0, 1, rs)
X <- as.matrix(N[, c("xa", "xb")])
WX <- W %*% X
W2X <- W %*% WX
W3X <- W %*% W2X
N$wy <- drop(W %*% N$y)
N$wxa <- WX[, 1]; N$wxb <- WX[, 2]
N$w2xa <- W2X[, 1]; N$w2xb <- W2X[, 2]
N$w3xa <- W3X[, 1]; N$w3xb <- W3X[, 2]
p1 <- ivreg(y ~ wy + xa + xb + wxa + wxb |
              xa + xb + wxa + wxb + w2xa + w2xb + w3xa + w3xb, data = N)
p2 <- ivreg(y ~ wy + xa + xb |
              xa + xb + wxa + wxb + w2xa + w2xb + w3xa + w3xb, data = N)
V1 <- vcovHC(p1, type = "HC0")
V2 <- vcovHC(p2, type = "HC0")
out$peer <- list(
  contextual = list(coef = as.list(coef(p1)), se = as.list(sqrt(diag(V1)))),
  no_contextual = list(coef = as.list(coef(p2)), se = as.list(sqrt(diag(V2))))
)

out$versions <- list(
  R = R.version.string,
  ivreg = as.character(packageVersion("ivreg")),
  sandwich = as.character(packageVersion("sandwich")),
  jsonlite = as.character(packageVersion("jsonlite"))
)

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
  file.path(fx, "proximal_interference_R.json")
)
