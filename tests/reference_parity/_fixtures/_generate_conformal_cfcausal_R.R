# Reference values for tests/reference_parity/test_conformal_cfcausal_r_parity.py
#
# cfcausal (Lei and Candes; MIT licence, GitHub lihualei71/cfcausal) is the
# canonical implementation of weighted split conformal inference for
# counterfactuals and individual treatment effects. conformalInference
# (Tibshirani et al.; GPL-2, GitHub ryantibs/conformal) is run as a black box
# for a second opinion on plain and weighted split conformal prediction.
#
# Every split is supplied (conformal_cfcausal_spec.json, written by
# _generate_conformal_cfcausal_data.py), so nothing random happens here. The
# outcome learner is lm, the propensity is the known e(x) of the data, and the
# weights are evaluated in closed form, so both sides fit the same models.
# predict.conformalSplit is called with wthigh = Inf, wtlow = 0 (no weight
# truncation; its defaults 20 / 0.05 truncate normalised weights).
#
# Install both into the private library first:
#   L=tests/reference_parity/_fixtures/_rlib_parity-families
#   R CMD INSTALL -l $L <cfcausal source dir>          # commit e4ca670
#   R CMD INSTALL -l $L <conformal>/conformalInference  # commit 15c51c6
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_conformal_cfcausal_R.R

fx <- "tests/reference_parity/_fixtures"
.libPaths(c(file.path(fx, "_rlib_parity-families"), .libPaths()))
suppressPackageStartupMessages({
  library(cfcausal)
  library(conformalInference)
  library(jsonlite)
})

dat <- read.csv(file.path(fx, "conformal_cfcausal_data.csv"))
spec <- fromJSON(file.path(fx, "conformal_cfcausal_spec.json"), simplifyVector = TRUE)

lmfun <- function(Y, X, Xtest) {
  X <- as.data.frame(X); Xtest <- as.data.frame(Xtest)
  colnames(Xtest) <- colnames(X)
  fit <- lm(Y ~ ., data = cbind(Y = Y, X))
  unname(predict(fit, newdata = Xtest))
}
ps <- function(X) plogis(spec$ps_a + spec$ps_b * X[, 1])
mat <- function(d) as.matrix(d[, c("x1", "x2")])

band <- function(X, Y, trainid, Xtest, alpha, wtfun = NULL, useInf = TRUE) {
  obj <- conformal(X, Y, type = "mean", side = "two", outfun = lmfun,
                   wtfun = wtfun, trainid = trainid)
  out <- predict(obj, Xtest, alpha = alpha, wthigh = Inf, wtlow = 0,
                 useInf = useInf)
  list(lower = out$lower, upper = out$upper, point = obj$Ymodel(Xtest))
}
# JSON cannot hold Inf: encode as strings and decode in the test
enc <- function(v) ifelse(is.finite(v), format(v, digits = 17), ifelse(v > 0, "inf", "-inf"))
encl <- function(b) lapply(b, enc)

out <- list()

# ---- weighted_conformal_prediction -------------------------------------
w <- dat[dat$block == "wcp", ]
tr <- w[w$role == "train", ]; ca <- w[w$role == "calib", ]; te <- w[w$role == "test", ]
X <- rbind(mat(tr), mat(ca)); Y <- c(tr$y, ca$y); tid <- seq_len(nrow(tr))
wt <- function(X) exp(spec$shift * X[, 1])
out$wcp <- list(
  plain_a10 = encl(band(X, Y, tid, mat(te), 0.10)),
  plain_a20 = encl(band(X, Y, tid, mat(te), 0.20)),
  weighted_a10 = encl(band(X, Y, tid, mat(te), 0.10, wt)),
  weighted_a05 = encl(band(X, Y, tid, mat(te), 0.05, wt))
)
# conformalInference: train on the first n_tr rows, same lm
trf <- function(x, y) lm.fit(cbind(1, x), y)$coefficients
prf <- function(out, newx) cbind(1, newx) %*% out
ci_plain <- conformal.pred.split(X, Y, mat(te), trf, prf, alpha = 0.10, split = tid)
ci_wt <- conformal.pred.split(X, Y, mat(te), trf, prf, alpha = 0.10, split = tid,
                              w = c(wt(X), wt(mat(te))))
out$wcp_conformalInference <- list(
  plain_a10 = list(lower = enc(ci_plain$lo[, 1]), upper = enc(ci_plain$up[, 1])),
  weighted_a10 = list(lower = enc(ci_wt$lo[, 1]), upper = enc(ci_wt$up[, 1]))
)

# ---- counterfactual and ITE intervals ----------------------------------
cf <- dat[dat$block == "cf" & dat$role == "data", ]
cft <- mat(dat[dat$block == "cf" & dat$role == "test", ])
out$cf <- list()
for (key in names(spec$cf)) {
  s <- spec$cf[[key]]
  arm <- function(a, alpha) {
    d <- cf[cf$t == a, ]
    wf <- if (a == 1) function(X) 1 / ps(X) else function(X) 1 / (1 - ps(X))
    band(mat(d), d$y, s[[as.character(a)]]$train, cft, alpha, wf)
  }
  y1 <- arm(1, 0.10); y0 <- arm(0, 0.10)
  # ITE: each counterfactual band at alpha / 2, combined as in cfcausal's
  # naive rule (conformalIteNaive): [L1 - U0, U1 - L0]
  i1 <- arm(1, 0.05); i0 <- arm(0, 0.05)
  out$cf[[key]] <- list(
    Y1 = encl(y1), Y0 = encl(y0),
    ite = list(lower = enc(i1$lower - i0$upper), upper = enc(i1$upper - i0$lower),
               point = enc(i1$point - i0$point))
  )
}

# ---- conformal_regression(method = "split", newdata = test) ------------
out$reg <- list()
for (key in names(spec$reg)) {
  b <- band(X, Y, spec$reg[[key]], mat(te), 0.10)
  out$reg[[key]] <- encl(b)
}

# ---- conformal_cate (randomized; unweighted per-arm bands) -------------
rct <- dat[dat$block == "rct", ]
out$cate <- list()
for (key in names(spec$cate)) {
  s <- spec$cate[[key]]
  arm <- function(a) {
    r <- s[[as.character(a)]]
    Xa <- rbind(mat(rct[r$train, ]), mat(rct[r$calib, ]))
    Ya <- c(rct$y[r$train], rct$y[r$calib])
    # alpha / 2 per arm; max score when the rank exceeds the calibration
    # size (useInf = FALSE), as conformal_cate does
    band(Xa, Ya, seq_along(r$train), mat(rct), s$alpha / 2, useInf = FALSE)
  }
  b1 <- arm(1); b0 <- arm(0)
  out$cate[[key]] <- list(
    cate = enc(b1$point - b0$point),
    lower = enc(b1$lower - b0$upper), upper = enc(b1$upper - b0$lower)
  )
}

# ---- conformal_continuous (plain split conformal on (dose, x1)) --------
ds <- dat[dat$block == "dose" & dat$role == "data", ]
dt <- dat[dat$block == "dose" & dat$role == "test", ]
dm <- function(d) as.matrix(d[, c("t", "x1")])
out$dose <- list()
for (key in names(spec$dose)) {
  s <- spec$dose[[key]]
  Xd <- rbind(dm(ds[s$train, ]), dm(ds[s$calib, ]))
  Yd <- c(ds$y[s$train], ds$y[s$calib])
  out$dose[[key]] <- encl(band(Xd, Yd, seq_along(s$train), dm(dt), s$alpha,
                               useInf = FALSE))
}

out$versions <- list(
  R = R.version.string,
  cfcausal = as.character(packageVersion("cfcausal")),
  cfcausal_commit = "e4ca6706599873505e1cce2d8cf0078ceff5d2e6",
  conformalInference = as.character(packageVersion("conformalInference")),
  conformalInference_commit = "15c51c66e3e5cab578a5c4ebd7494685efa83788"
)
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = NA),
           file.path(fx, "conformal_cfcausal_R.json"))
cat("wrote conformal_cfcausal_R.json\n")
