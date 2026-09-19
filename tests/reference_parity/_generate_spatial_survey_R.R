#!/usr/bin/env Rscript
# Frozen R reference for the spatial half of the "spatial_survey" parity
# family (tests/reference_parity/test_spatial_survey_R_parity.py).
#
# Every section reads the CSVs written by _prepare_spatial_survey_data.R, so
# R and Python see identical bytes.
#
#   weights   spdep::poly2nb (queen / rook) on Columbus polygons, nb2listw
#             styles W / B / S / U, nb2blocknb (block weights),
#             spdep::nb2listwdist(type = "dpd", alpha = 2) and
#             GWmodel::gw.weight (kernel weights) on Georgia points.
#   gwr       GWmodel::gwr.basic for bisquare / gaussian / exponential x
#             adaptive / fixed; gwr.aic / gwr.cv criterion values; bw.gwr.
#   mgwr      GWmodel::gwr.multiscale at FIXED bandwidths
#             (bw.seled = TRUE, force.armadillo = TRUE -- the default C++
#             path ignores bw.seled -- predictor.centered = FALSE).
#   sarar     spatialreg::gstsls on Columbus (default, robust, sig2n_k).
#   spiv      sphet::spreg(model = "lag", het = TRUE) with an endogenous
#             regressor and an excluded instrument.
#   panel     splm::spml(model = "within") SAR / SEM, individual and
#             two-way effects; SDM as SAR with W-lagged regressors built per
#             period from the raw X.
#
# Regenerate: Rscript tests/reference_parity/_generate_spatial_survey_R.R
suppressPackageStartupMessages({
  library(jsonlite); library(sf); library(spdep); library(spatialreg)
  library(GWmodel); library(sphet); library(splm); library(plm)
})
args_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(args_file)) dirname(normalizePath(args_file)) else "tests/reference_parity"
fx <- file.path(here, "_fixtures")
rd <- function(f) read.csv(file.path(fx, f), stringsAsFactors = FALSE)
nb0 <- function(nb) lapply(seq_along(nb), function(i) {
  v <- as.integer(nb[[i]]); if (length(v) == 1L && v == 0L) integer(0) else sort(v) - 1L
})
# weights of a listw in the neighbour order of nb0() (0-based, sorted)
lw_rows <- function(lw) lapply(seq_along(lw$neighbours), function(i) {
  v <- as.integer(lw$neighbours[[i]])
  if (length(v) == 1L && v == 0L) return(numeric(0))
  as.numeric(lw$weights[[i]])[order(v)]
})
out <- list()

# ------------------------------------------------------------- weights --
columbus <- rd("spatial_survey_columbus.csv")
geom <- st_as_sfc(columbus$wkt)
nbq <- poly2nb(geom, queen = TRUE)
nbr <- poly2nb(geom, queen = FALSE)
out$contiguity <- list(
  queen = nb0(nbq), rook = nb0(nbr),
  queen_W = lw_rows(nb2listw(nbq, style = "W")),
  queen_S = lw_rows(nb2listw(nbq, style = "S")),
  queen_U = lw_rows(nb2listw(nbq, style = "U")),
  rook_W = lw_rows(nb2listw(nbr, style = "W"))
)
nbb <- nb2blocknb(NULL, columbus$regime)
out$block <- list(neighbours = nb0(nbb),
                  W = lw_rows(nb2listw(nbb, style = "W", zero.policy = TRUE)))

georgia <- rd("spatial_survey_georgia.csv")
pts <- st_as_sf(georgia, coords = c("X", "Y"))
loc <- as.matrix(georgia[, c("X", "Y")])
dm <- gw.dist(loc, loc)
H <- 100000
nbd <- dnearneigh(pts, 0, H)
lw_dpd <- nb2listwdist(nbd, pts, type = "dpd", style = "raw", alpha = 2, dmax = H)
lw_dpd_W <- nb2listwdist(nbd, pts, type = "dpd", style = "W", alpha = 2, dmax = H)
K <- 8L
out$kernel <- list(
  h = H, k = K,
  dpd_neighbours = nb0(nbd), dpd_raw = lw_rows(lw_dpd), dpd_W = lw_rows(lw_dpd_W),
  # gw.weight matrices: column i holds the weights around point i
  gw_gaussian_fixed = unname(gw.weight(dm, H, "gaussian", FALSE)),
  gw_bisquare_fixed = unname(gw.weight(dm, H, "bisquare", FALSE)),
  # adaptive: GWmodel counts the point itself, so K others -> bw = K + 1
  gw_bisquare_adaptive = unname(gw.weight(dm, K + 1, "bisquare", TRUE)),
  gw_gaussian_adaptive = unname(gw.weight(dm, K + 1, "gaussian", TRUE))
)

# ----------------------------------------------------------------- gwr --
fm <- PctBach ~ PctRural + PctPov + PctBlack
cfgs <- list(
  list(id = "bisquare_adaptive", kernel = "bisquare", adaptive = TRUE, bw = 40),
  list(id = "gaussian_adaptive", kernel = "gaussian", adaptive = TRUE, bw = 40),
  list(id = "exponential_adaptive", kernel = "exponential", adaptive = TRUE, bw = 40),
  list(id = "bisquare_fixed", kernel = "bisquare", adaptive = FALSE, bw = 150000),
  list(id = "gaussian_fixed", kernel = "gaussian", adaptive = FALSE, bw = 60000),
  list(id = "exponential_fixed", kernel = "exponential", adaptive = FALSE, bw = 60000),
  list(id = "bisquare_adaptive_frac", kernel = "bisquare", adaptive = TRUE, bw = 40.7),
  list(id = "bisquare_adaptive_over_n", kernel = "bisquare", adaptive = TRUE, bw = 200)
)
gw <- list()
for (cf in cfgs) {
  m <- gwr.basic(fm, data = pts, bw = cf$bw, kernel = cf$kernel, adaptive = cf$adaptive)
  sdf <- as.data.frame(m$SDF)
  vars <- c("Intercept", "PctRural", "PctPov", "PctBlack")
  gw[[cf$id]] <- list(
    kernel = cf$kernel, adaptive = cf$adaptive, bw = cf$bw,
    betas = unname(as.matrix(sdf[, vars])),
    se = unname(as.matrix(sdf[, paste0(vars, "_SE")])),
    local_R2 = sdf$Local_R2, yhat = sdf$yhat,
    diag = m$GW.diagnostic
  )
}
x <- model.matrix(fm, georgia); y <- georgia$PctBach
crit <- list()
for (b in list(list("bisquare", TRUE, 30), list("bisquare", TRUE, 60),
               list("gaussian", TRUE, 45), list("bisquare", FALSE, 120000),
               list("gaussian", FALSE, 80000), list("exponential", FALSE, 80000))) {
  crit[[length(crit) + 1]] <- list(
    kernel = b[[1]], adaptive = b[[2]], bw = b[[3]],
    aicc = GWmodel:::gwr.aic(b[[3]], x, y, b[[1]], b[[2]], loc, dMat = dm, verbose = FALSE),
    cv = c(GWmodel:::gwr.cv(b[[3]], x, y, b[[1]], b[[2]], loc, dMat = dm, verbose = FALSE)))
}
sel <- list()
for (k in c("bisquare", "gaussian")) for (ad in c(TRUE, FALSE)) for (ap in c("AICc", "CV")) {
  invisible(capture.output(bw <- bw.gwr(fm, data = pts, approach = ap, kernel = k, adaptive = ad)))
  sel[[length(sel) + 1]] <- list(kernel = k, adaptive = ad, approach = ap, bw = bw)
}
out$gwr <- list(fits = gw, criteria = crit, bw_select = sel)

# ---------------------------------------------------------------- mgwr --
bws_mg <- c(40, 80, 120, 60)
invisible(capture.output(mg <- gwr.multiscale(
  fm, data = pts, kernel = "bisquare", adaptive = TRUE, bws0 = bws_mg,
  bw.seled = rep(TRUE, 4), predictor.centered = rep(FALSE, 3),
  threshold = 1e-12, max.iterations = 20000, criterion = "dCVR",
  force.armadillo = TRUE, hatmatrix = FALSE)))
mgs <- as.data.frame(mg$SDF)
out$mgwr <- list(bws = bws_mg, betas = unname(as.matrix(mgs[, c("Intercept", "PctRural", "PctPov", "PctBlack")])),
                 yhat = mgs$yhat, threshold = 1e-12)

# --------------------------------------------------------------- sarar --
lwq <- nb2listw(nbq, style = "W")
sar_out <- list()
for (cf in list(list(id = "default", robust = FALSE, sig2n_k = FALSE),
                list(id = "robust", robust = TRUE, sig2n_k = FALSE),
                list(id = "sig2n_k", robust = FALSE, sig2n_k = TRUE))) {
  g <- gstsls(CRIME ~ INC + HOVAL, data = columbus, listw = lwq,
              robust = cf$robust, sig2n_k = cf$sig2n_k)
  sar_out[[cf$id]] <- list(
    coef = as.list(g$coefficients), se = as.list(sqrt(diag(g$secstep_var))),
    lambda = unname(g$lambda), gm_sigma2 = unname(g$GMs2),
    gm_objective = unname(g$optres$objective), start = unname(g$pars))
}
out$sarar <- sar_out

# ---------------------------------------------------------------- spiv --
siv <- spreg(CRIME ~ INC, data = columbus, listw = lwq, endog = ~HOVAL,
             instruments = ~DISCBD, model = "lag", het = TRUE)
cf <- as.numeric(siv$coefficients); names(cf) <- rownames(siv$coefficients)
out$spatial_iv <- list(coef = as.list(cf),
                       se = as.list(setNames(sqrt(diag(as.matrix(siv$var))), names(cf))))

# --------------------------------------------------------------- panel --
pr <- rd("spatial_survey_produc.csv")
ww <- as.matrix(read.csv(file.path(fx, "spatial_survey_usaww.csv"), row.names = 1))
lwp <- mat2listw(ww, style = "W")
Wr <- ww / rowSums(ww)
pr <- pr[order(pr$year, pr$state), ]
for (v in c("lpcap", "lpc", "lemp", "unemp"))
  pr[[paste0("W_", v)]] <- unlist(lapply(split(pr[[v]], pr$year), function(z) as.vector(Wr %*% z)))
fmp <- lgsp ~ lpcap + lpc + lemp + unemp
fmd <- lgsp ~ lpcap + lpc + lemp + unemp + W_lpcap + W_lpc + W_lemp + W_unemp
pan <- list()
for (eff in c("individual", "twoways")) {
  for (spec in list(list(id = "sar", f = fmp, lag = TRUE, err = "none"),
                    list(id = "sem", f = fmp, lag = FALSE, err = "b"),
                    list(id = "sdm", f = fmd, lag = TRUE, err = "none"))) {
    m <- spml(spec$f, data = pr, index = c("state", "year"), listw = lwp,
              model = "within", effect = eff, lag = spec$lag, spatial.error = spec$err)
    pan[[paste(spec$id, eff, sep = "_")]] <- list(
      coef = as.list(coef(m)),
      # vcov(spml) carries no dimnames; its order is coef(m)'s
      se = as.list(setNames(sqrt(diag(as.matrix(vcov(m)))),
                            if (is.null(rownames(vcov(m)))) names(coef(m)) else rownames(vcov(m)))),
      logLik = if (spec$id == "sem") NULL else as.numeric(m$logLik),
      sigma2 = as.numeric(m$sigma2))
  }
}
out$panel <- pan

out$versions <- list(
  R = R.version.string,
  spdep = as.character(packageVersion("spdep")),
  spatialreg = as.character(packageVersion("spatialreg")),
  GWmodel = as.character(packageVersion("GWmodel")),
  sphet = as.character(packageVersion("sphet")),
  splm = as.character(packageVersion("splm")),
  plm = as.character(packageVersion("plm")),
  sf = as.character(packageVersion("sf")),
  spData = as.character(packageVersion("spData"))
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = FALSE, null = "null"),
           file.path(fx, "spatial_survey_R.json"))
cat("wrote spatial_survey_R.json\n")
