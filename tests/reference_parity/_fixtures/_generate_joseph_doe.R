# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/test_joseph_doe_parity.py
#
# The computations of Joseph, "Experimental Design for Data Science and
# Engineering" (Chapman and Hall/CRC, 2025), run with the R packages the
# book uses, on public test functions and on data simulated here. The data
# are written into the JSON, so the Python side reads the same numbers.
#
# Requires: SFDesign (0.1.5), MaxPro (4.1-2), twinning (1.1), SPlit (1.3),
#           support (0.1.7, archived on CRAN), sensitivity (1.31.0),
#           FrF2 (2.3-5), DoE.base (1.2-5), unrepx (1.0-2), AlgDesign
#           (1.2.1.2), rkriging (1.0.2), MASS, jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_joseph_doe.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(SFDesign); library(MaxPro); library(twinning); library(SPlit)
  library(support); library(sensitivity); library(FrF2); library(DoE.base)
  library(unrepx); library(AlgDesign); library(rkriging); library(jsonlite)
})
out <- list()

# --- space-filling criteria and augmentation (deterministic) ---------------
set.seed(11); X <- matrix(runif(24), 8, 3)
C <- matrix(runif(600), 200, 3)
S <- matrix(rnorm(900), 300, 3); S[, 2] <- S[, 2] * 3 + S[, 1]
out$criteria <- list(
  X = X, maxpro = maxpro.crit(X), maxpro_delta = maxpro.crit(X, 1e-3),
  maximin = maximin.crit(X), reciprocal = maximin.crit(X, surrogate = TRUE),
  reciprocal_r4 = maximin.crit(X, r = 4, surrogate = TRUE), wraparound = uniform.crit(X),
  C = C, augmented = MaxProAugment(X, C, nNew = 6)$Design,
  S = S, energy = energy(S, S[1:25, ]))

# --- quality of searched designs (stochastic: a screen) --------------------
best <- function(n, p, f, crit, k = 5) {
  v <- sapply(1:k, function(s) { set.seed(s); crit(f(n, p)) }); c(min(v), median(v), max(v)) }
out$quality <- list(
  maxpro_20_2 = best(20, 2, function(n, p) maxpro.optim(maxproLHD(n, p)$design)$design, maxpro.crit),
  maxpro_50_5 = best(50, 5, function(n, p) maxpro.optim(maxproLHD(n, p)$design)$design, maxpro.crit),
  maximin_20_2 = best(20, 2, function(n, p) maximinLHD(n, p)$design, maximin.crit),
  maximin_50_5 = best(50, 5, function(n, p) maximinLHD(n, p)$design, maximin.crit),
  uniform_20_2 = best(20, 2, function(n, p) uniformLHD(n, p)$design, uniform.crit),
  uniform_50_5 = best(50, 5, function(n, p) uniformLHD(n, p)$design, uniform.crit))

# --- fractional factorials --------------------------------------------------
wl <- list()
for (cfg in list(c(8,4), c(8,5), c(8,6), c(8,7), c(16,5), c(16,6), c(16,7), c(16,8), c(16,9),
                 c(16,10), c(32,6), c(32,7), c(32,8), c(32,9), c(64,7), c(64,8), c(64,9))) {
  d <- FrF2(cfg[1], cfg[2], randomize = FALSE)
  wl[[paste(cfg, collapse = "_")]] <- list(gwlp = as.numeric(GWLP(d))[-1],
                                           resolution = design.info(d)$catlg.entry[[1]]$res)
}
out$frf2 <- wl
L9 <- data.matrix(undesign(oa.design(nfactors = 4, nlevels = 3, randomize = FALSE)))
L18 <- data.matrix(undesign(oa.design(ID = L18, randomize = FALSE)))
set.seed(3); R <- cbind(sample(rep(1:2, 9)), sample(rep(1:3, 6)), sample(rep(1:3, 6)), sample(rep(1:6, 3)))
out$gwlp <- list(L9 = L9, L9_twice = as.numeric(GWLP(rbind(L9, L9)))[-1],
                 L18 = L18, L18_cols_2_5 = as.numeric(GWLP(L18[, 2:5]))[-1],
                 L18_cols_3_6 = as.numeric(GWLP(L18[, 3:6]))[-1], L18_all_k4 = as.numeric(GWLP(L18, k = 4))[-1],
                 mixed = R, mixed_gwlp = as.numeric(GWLP(R))[-1])

# --- unreplicated factorial: Lenth ------------------------------------------
set.seed(5); e15 <- c(rnorm(12), 6, -8, 11)
g <- expand.grid(A = c(-1, 1), B = c(-1, 1), C = c(-1, 1), D = c(-1, 1))
set.seed(8); g$y <- 10 + 3 * g$A - 2 * g$C + 1.5 * g$A * g$C + rnorm(16, sd = .5)
fit <- lm(y ~ A * B * C * D, data = g); eff <- 2 * coef(fit)[-1]
set.seed(1)
out$lenth <- list(e15 = e15, pse15 = unname(PSE(e15, method = "Lenth")), me15 = unname(ME(e15, method = "Lenth")),
                  data = g, effects = unname(eff), names = names(eff), pse = unname(PSE(eff, method = "Lenth")),
                  me = unname(ME(eff, method = "Lenth")))
half <- g[g$D == g$A * g$B * g$C, ]
hf <- lm(y ~ (A + B + C)^3, data = half)
out$lenth$half <- list(rows = as.integer(rownames(half)), coef = unname(coef(hf)))

# --- support points, SPlit, twinning ----------------------------------------
set.seed(123); N <- 400
D <- MASS::mvrnorm(N, c(0, 0), matrix(c(1, .5, .5, 1), 2)); D[, 2] <- 3 * D[, 2] + D[, 1]^2
sink(tempfile()); sp_en <- sapply(1:5, function(s) { set.seed(s); energy(D, sp(40, 2, dist.samp = D)$sp) })
set.seed(1); SP <- sp(40, 2, dist.samp = D)$sp; sink()
split_en <- sapply(1:5, function(s) { set.seed(s); energy(D, D[SPlit(D, splitRatio = .2, tolerance = 1e-8), ]) })
set.seed(9); F <- data.frame(x = rnorm(240), g = factor(sample(c("a", "b", "c"), 240, TRUE)), y = runif(240))
set.seed(2); G <- matrix(rnorm(3 * 501), 501, 3)
out$support <- list(D = D, sp_energy = sp_en, SP = SP, subsample = subsample(D, SP),
                    split_energy = split_en,
                    twin_r5_u1 = twin(D, r = 5, u1 = 1), twin_r4_u77 = twin(D, r = 4, u1 = 77),
                    twin_r10_u300 = twin(D, r = 10, u1 = 300),
                    F = F, twin_F = twin(F, r = 4, u1 = 10), G = G, twin_G = twin(G, r = 5, u1 = 3))

# --- sensitivity analysis (borehole function on the unit cube) --------------
f <- function(x) {
  lower <- c(0.05, 100, 63070, 990, 63.1, 700, 1120, 9855); upper <- c(0.15, 50000, 115600, 1110, 116, 820, 1680, 12045)
  x <- lower + x * (upper - lower)
  2 * pi * x[3] * (x[4] - x[6]) / (log(x[2] / x[1]) * (1 + 2 * x[7] * x[3] / (log(x[2] / x[1]) * x[1]^2 * x[8]) + x[3] / x[5]))
}
borehole <- function(X) apply(X, 1, f)
set.seed(1); p <- 8; m <- 200
A <- matrix(runif(m * p), nrow = m); B <- matrix(runif(m * p), nrow = m)
a <- soboljansen(model = borehole, X1 = data.frame(A), X2 = data.frame(B))
set.seed(1)
mo <- morris(model = borehole, factors = 8, r = 6, design = list(type = "oat", levels = 4, grid.jump = 2))
out$sensitivity <- list(A = A, B = B, first = a$S[, 1], total = a$T[, 1],
                        morris_X = mo$X, morris_y = as.numeric(mo$y), mu = unname(apply(mo$ee, 2, mean)),
                        mu_star = unname(apply(mo$ee, 2, function(x) mean(abs(x)))),
                        sigma = unname(apply(mo$ee, 2, sd)))

# --- optimal designs ---------------------------------------------------------
cand <- data.frame(x = seq(-1, 1, length = 301))
set.seed(1)
a <- optFederov(~ 1 + x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8) + I(x^9),
                nTrials = 10, data = cand, approximate = FALSE)
cand2 <- expand.grid(a = seq(-1, 1, length = 5), b = seq(-1, 1, length = 5), c = seq(-1, 1, length = 5))
set.seed(2); q <- optFederov(~ quad(.), data = cand2, nTrials = 14, nRepeats = 20)
out$optimal <- list(poly9_x = sort(a$design[, 1]), poly9_D = a$D,
                    quad3_D = q$D, quad3_A = q$A, quad3_I = q$I, quad3_design = q$design)

# --- kriging -----------------------------------------------------------------
fn <- function(x) sin(10 * pi * x) / (1 + 64 * (x - .25)^2) + x^2
test <- seq(0, 1, length = 301)
n <- 8; D1 <- ((1:n) - .5) / n; y1 <- fn(D1)
k0 <- Fit.Kriging(D1, y1, fit = FALSE, kernel = Get.Kernel(.08 / sqrt(2), type = "Gaussian"), model = "OK")
p0 <- Get.Kriging.Parameters(k0); q0 <- Predict.Kriging(k0, test)
k1 <- Fit.Kriging(D1, y1, kernel.parameters = list(type = "Gaussian"))
p1 <- Get.Kriging.Parameters(k1); q1 <- Predict.Kriging(k1, test)
EI <- function(x, hmin, obj) { pr <- Predict.Kriging(obj, cbind(x)); s <- max(pr$sd, 1e-10); u <- (hmin - pr$mean) / s; s * (u * pnorm(u) + dnorm(u)) }
ei <- apply(cbind(test), 1, EI, hmin = min(y1), obj = k1)
set.seed(5); Dn <- rep(((1:10) - 1) / 9, 2); yn <- fn(Dn) + rnorm(20, sd = .1)
k2 <- Fit.Kriging(Dn, yn, interpolation = FALSE, kernel.parameters = list(type = "Gaussian"))
p2 <- Get.Kriging.Parameters(k2); q2 <- Predict.Kriging(k2, test)
out$kriging <- list(x = D1, y = y1, test = test,
                    fixed = list(lengthscale = .08 / sqrt(2), nu2 = p0$nu2, mu = p0$mu, mean = q0$mean, sd = q0$sd),
                    fitted = list(lengthscale = p1$lengthscale, nu2 = p1$nu2, mu = p1$mu, mean = q1$mean, sd = q1$sd, ei = ei),
                    noisy = list(x = Dn, y = yn, lengthscale = p2$lengthscale, nu2 = p2$nu2, sigma2 = p2$sigma2, mu = p2$mu,
                                 mean = q2$mean, sd = q2$sd))

out$versions <- sapply(c("SFDesign", "MaxPro", "twinning", "SPlit", "support", "sensitivity", "FrF2",
                         "DoE.base", "unrepx", "AlgDesign", "rkriging"), function(p) as.character(packageVersion(p)))
out$R <- R.version.string
write_json(out, "tests/reference_parity/_fixtures/joseph_doe_R.json", digits = 17, auto_unbox = TRUE,
           dataframe = "columns", pretty = FALSE)
cat("written\n")
