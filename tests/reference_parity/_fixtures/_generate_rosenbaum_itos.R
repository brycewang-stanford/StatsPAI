# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/test_rosenbaum_itos_parity.py
#
# The analyses of Rosenbaum, "An Introduction to the Theory of Observational
# Studies" (Springer, 2025), run with the three packages the book uses.
# Both sides read the CSV files written by _generate_rosenbaum_itos_data.R:
# simulated data in the committed fixtures, or the book's own data when
# STATSPAI_ITOS_FIXTURES points at a directory holding them (see that script).
#
# Requires: iTOS (1.0.3), weightedRank (0.7.0), tightenBlock (0.1.7),
#           senstrat, sensitivitymv, DOS2, rcbalance, rlemon, jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_rosenbaum_itos.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(weightedRank); library(iTOS); library(senstrat); library(sensitivitymv)
  library(DOS2); library(rcbalance); library(jsonlite)
})
# DOS2 and rcbalance export functions of the same names; the book's are iTOS's.
for (fn in c("addcaliper", "addMahal", "addNearExact", "addinteger", "addquantile",
             "startcost", "amplify", "noether", "evalBal"))
  assign(fn, getExportedValue("iTOS", fn))
here <- Sys.getenv("STATSPAI_ITOS_FIXTURES", "tests/reference_parity/_fixtures")
rd <- function(f) read.csv(file.path(here, f))
out <- list()

y <- as.matrix(rd("rosenbaum_itos_hdl.csv"))                 # 406 x 4
bp <- rd("rosenbaum_itos_bp.csv")
yb <- as.matrix(bp[, c("b", "n", "p")])                      # 207 x 3
peri <- rd("rosenbaum_itos_peri.csv")
pp <- peri[peri$pair == 1, ]; perD <- t(matrix(pp$pd, 4, 606))
zp <- t(matrix(pp$z, 4, 606))
sets <- peri[peri$pair == 0, ]; sperD <- t(matrix(sets$pd, 5, 213))
zs <- t(matrix(sets$z, 5, 213))
f <- function(r) c(list(pval = r$pval), as.list(r$detail))

## wgtRank: one treated individual per block ---------------------------------
w <- list()
for (phi in c("wilc", "quade", "u868", "u878", "u888", "u858", "mixed"))
  for (g in c(1, 2, 3.5, 5)) w[[paste(phi, g)]] <- f(wgtRank(y, gamma = g, phi = phi))
out$wgtRank <- w
out$wgtRank_pairs <- f(wgtRank(yb[, 1:2], gamma = 2.3, phi = "u878"))

## wgtRankCI: point estimates and confidence intervals ------------------------
ci <- list()
for (phi in c("wilc", "quade", "u868", "u878")) for (g in c(1, 2))
  for (alt in c("greater", "twosided", "less"))
    ci[[paste(phi, g, alt)]] <- wgtRankCI(y, phi = phi, gamma = g, alternative = alt)
out$wgtRankCI <- ci

## wgtRanktt: adaptive choice between two weight functions --------------------
tt <- list()
for (g in c(3, 5)) {
  r <- wgtRanktt(y, gamma = g)
  tt[[paste("u868 u878", g)]] <- list(jointP = r$jointP, cor12 = r$cor12, detail = r$detail)
}
r <- wgtRanktt(y, phi1 = "wilc", phi2 = "mixed", gamma = 4)
tt[["wilc mixed 4"]] <- list(jointP = r$jointP, cor12 = r$cor12, detail = r$detail)
out$wgtRanktt <- tt

## dwgtRank: scores, the gap, lower-tail tests --------------------------------
d <- list()
d$second_factor <- f(dwgtRank(yb[, 3:1], gamma = 1.45, alternative = "less",
                              scores = c(1, 2, 5), range = FALSE, m = 8, m1 = 8, m2 = 8))
d$stratified_wilcoxon <- f(dwgtRank(yb[, 3:1], gamma = 1.45, alternative = "less",
                                    scores = c(1, 2, 3), m = 1, m1 = 1, m2 = 1))
d$pairs <- f(dwgtRank(yb[, 1:2], gamma = 2.3, m = 8, m1 = 7, m2 = 8))
d$gap <- f(dwgtRank(yb, gamma = 1.5, m = 8, m1 = 6, m2 = 8, range = FALSE))
d$less <- f(dwgtRank(yb, gamma = 1.5, m = 8, m1 = 6, m2 = 8, alternative = "less"))
out$dwgtRank <- d

## gwgtRank: blocks with two treated individuals ------------------------------
G <- list()
for (phi in c("wilc", "quade", "u868", "u878")) for (g in c(1, 2, 4)) {
  r <- gwgtRank(perD, zp, phi = phi, gamma = g)
  G[[paste(phi, g)]] <- list(pval = r$pval, detail = as.list(r$detail))
}
r <- gwgtRank(perD, zp, phi = "u868", gamma = 3, detail = TRUE)
G$taylor <- list(lin = as.list(r$LinearBoundResult), sep = as.list(r$Separable))
out$gwgtRank <- G

## gwgtRankC and wgtRankC: conditioning on the extremes -----------------------
k <- function(r) list(pval = r$pval, detail = as.list(r$detail), counts = as.list(r$block.counts))
C <- list()
C$m20_g11 <- k(gwgtRankC(perD, zp, gamma = 11, m1 = 19, m2 = 20, m = 20))
C$u878_g7 <- k(gwgtRankC(perD, zp, gamma = 7, m1 = 7, m2 = 8, m = 8))
C$u222_g45 <- k(gwgtRankC(perD, zp, gamma = 4.5, m1 = 2, m2 = 2, m = 2))
C$mixed_g3 <- k(gwgtRankC(perD, zp, gamma = 3, m = NULL))
C$less_g2 <- k(gwgtRankC(perD, zp, gamma = 2, m1 = 7, m2 = 8, m = 8, alternative = "less"))
C$sets_g89 <- k(gwgtRankC(sperD, zs, gamma = 8.9, m1 = 7, m2 = 8, m = 8))
C$hdl_g6 <- k(gwgtRankC(y, 1, gamma = 6, m1 = 7, m2 = 8, m = 8))
out$gwgtRankC <- C
# With continuous outcomes there are no ties, and wgtRankC is the same test.
r <- wgtRankC(yb, phi = "u878", gamma = 1.8)
out$wgtRankC_untied <- list(pval = r$pval, deviate = unname(r$detail["Deviate"]),
                            info = as.list(r$block.information))
r <- suppressWarnings(wgtRankC(y, phi = "u878", gamma = 6))
out$wgtRankC_tied <- list(pval = r$pval)

## ef2C and truncatedP --------------------------------------------------------
g <- function(r) list(pvals = as.list(r$pvals), detail = r$detail)
e <- list()
e$g23_u145 <- g(ef2C(yb, gamma = 2.3, upsilon = 1.45))
e$g26_u17 <- g(ef2C(yb, gamma = 2.6, upsilon = 1.7))
e$fisher <- g(ef2C(yb, gamma = 2.6, upsilon = 1.7, trunc = 1))
e$range_123 <- g(ef2C(yb, gamma = 2, upsilon = 2, m1 = c(6, 6), range = TRUE, scores = 1:3))
e$less <- g(ef2C(yb, gamma = 1.5, upsilon = 1.2, alternative = "less"))
out$ef2C <- e
out$truncatedP <- list(
  a = truncatedP(c(.01, .3)), b = truncatedP(c(.01, .3, .15, .04), trunc = .1),
  c = truncatedP(c(.5, .3)), d = truncatedP(c(.01, .3, .6), trunc = 1),
  e = truncatedP(c(.19, .02, .2, .7, .0004)))
out$amplify <- list(a = as.numeric(amplify(4, 7)), b = as.numeric(amplify(4, 5:19)),
                    c = as.numeric(amplify(1.45, 2.5)))

## noether --------------------------------------------------------------------
dn <- rd("rosenbaum_itos_pairs.csv")$d
out$noether <- list(sign = noether(dn, f = 0, gamma = 3), top_third = noether(dn, gamma = 3),
                    two_sided = noether(dn, f = 1 / 3, gamma = 1, alternative = "two.sided"),
                    less = noether(-dn, gamma = 2, alternative = "less"))

## estPower -------------------------------------------------------------------
p <- list(); gam <- c(1, 2, 3, 4, 5, 7, 9)
for (phi in c("wilc", "quade", "u868", "u878", "u888", "mixed")) {
  r <- estPower(y, gam, phi = phi)
  p[[phi]] <- list(power = as.numeric(r$power), jackm = as.numeric(r$jackm), jackv = as.numeric(r$jackv))
}
r <- estPower(y, gam, phi = "u868", ssratio = 1000 / 406)
p$u868_ratio <- list(power = as.numeric(r$power))
out$estPower <- p

## senstrat / sen2sample: strata of any size ----------------------------------
sc <- rank(peri$pd); s <- list()
for (gm in c(1, 1.5, 3)) for (alt in c("greater", "less")) {
  r <- senstrat(sc, peri$z, peri$block, gamma = gm, alternative = alt, method = "RK", detail = TRUE)
  s[[paste(gm, alt)]] <- list(lin = as.list(r$LinearBoundResult), sep = as.list(r$Separable),
                              desc = as.list(r$Description))
}
out$senstrat_blocks <- s
binge <- rd("rosenbaum_itos_binge.csv")
bn <- binge[binge$AlcGroup != "P", ]; zz <- 1 * (bn$AlcGroup == "B")
st <- as.integer(interaction(bn$ageC, bn$female))
# Eight large strata. method = "RK" is exact but cubic in the stratum size,
# so it is run on the first 1200 rows; "BU" takes the hypergeometric moments
# from BiasedUrn at its default precision of 1e-7.
s2 <- list()
r <- senstrat(rank(bn$bpCombined[1:1200]), zz[1:1200], st[1:1200], gamma = 1.3, method = "RK", detail = TRUE)
s2[["rank 1.3 RK"]] <- list(lin = as.list(r$LinearBoundResult), sep = as.list(r$Separable))
for (gm in c(1.1, 1.3)) {
  r <- senstrat(rank(bn$bpCombined), zz, st, gamma = gm, detail = TRUE)
  s2[[paste("rank", gm)]] <- list(lin = as.list(r$LinearBoundResult), sep = as.list(r$Separable))
  r <- senstrat(hodgeslehmann(bn$bpCombined, zz, st, align = "mean"), zz, st, gamma = gm, detail = TRUE)
  s2[[paste("aligned", gm)]] <- list(lin = as.list(r$LinearBoundResult), sep = as.list(r$Separable))
  r <- senstrat(bn$bpCombined, zz, st, gamma = gm, detail = TRUE)
  s2[[paste("raw", gm)]] <- list(lin = as.list(r$LinearBoundResult), sep = as.list(r$Separable))
}
out$senstrat_strata <- s2
# One stratum: the first 300 rows of the same comparison.
b3 <- bn[1:300, ]; z3 <- zz[1:300]
two <- list()
for (gm in c(1, 1.2, 2)) for (alt in c("greater", "less")) {
  r <- sen2sample(rank(b3$bpCombined), z3, gamma = gm, alternative = alt, method = "RK")
  two[[paste(gm, alt)]] <- list(pval = r$pval, detail = as.list(r$detail))
}
out$sen2sample <- two
out$ev <- list(RK = ev(1:5, c(0, 1, 0, 1, 0), 3, 2, "RK"))

## Matched pairs: the existing sp.rosenbaum_bounds ----------------------------
dp <- yb[, 1] - yb[, 2]; pr <- list()
for (gm in c(1, 1.5, 2, 2.5)) {
  a <- senWilcox(dp, gamma = gm, conf.int = TRUE, alternative = "twosided")
  pr[[paste(gm)]] <- list(pval_greater = senWilcox(dp, gamma = gm)$pval, two = a,
                          wgt_wilc = wgtRank(yb[, 1:2], gamma = gm, phi = "wilc")$pval,
                          senU = senU(dp, gamma = gm, m = 8, m1 = 7, m2 = 8)$pval)
}
out$pairs <- pr

## Two-criteria matching -------------------------------------------------------
run <- function(left, right, ncontrols = 1, controlcosts = NULL, treatedcosts = NULL, tighten = FALSE) {
  if (is.null(controlcosts)) controlcosts <- rep(0, ncol(left))
  net <- if (tighten) tightenBlock::makenetwork(left, right, ncontrols = ncontrols,
                                                controlcosts = controlcosts, treatedcosts = treatedcosts)$net
         else iTOS::makenetwork(left, right, ncontrols = ncontrols, controlcosts = controlcosts)$net
  res <- callrelax(net, solver = "rlemon")
  nT <- nrow(left); nC <- ncol(left); x <- res$x
  P <- matrix(x[1:(nT * nC)], nT, nC); U <- x[(nT * nC + 1):(nT * nC + nC)]
  W <- matrix(x[(nT * nC + nC + 1):(2 * nT * nC + nC)], nT, nC)
  byp <- if (tighten) x[(2 * nT * nC + nC + 1):(2 * nT * nC + nC + nT)] else rep(0, nT)
  list(controls = sort(as.numeric(colnames(left)[colSums(P) == 1])),
       left_true = sum(left * P), right_true = sum(right * W), cc = sum(controlcosts * U),
       objective_truncated = sum(trunc(left) * P) + sum(trunc(right) * W) + sum(trunc(controlcosts) * U),
       n_skipped = sum(byp))
}
prep <- function(grp) {
  z <- rep(NA, nrow(binge)); z[binge$AlcGroup == "B"] <- 1; z[binge$AlcGroup == grp] <- 0
  dt <- binge[!is.na(z), ]; z <- z[!is.na(z)]; o <- order(1 - z, dt$SEQN); dt <- dt[o, ]; z <- z[o]
  rownames(dt) <- dt$SEQN; names(z) <- dt$SEQN
  ctl <- glm.control(epsilon = 1e-14, maxit = 100)
  p <- glm(z ~ age + female + education + smokenow + smokeQuit + bpRX + bmi + vigor + waisthip,
           family = binomial, data = dt, control = ctl)$fitted.values
  list(dt = dt, z = z, p = p)
}
M <- list()
# Section 4.3 of the book: a caliper and fine balance on the propensity score
q <- prep("N"); z <- q$z; p <- q$p
left <- addcaliper(startcost(z), z, p, penalty = 10)
right <- addinteger(startcost(z), z, (p > 0.05) + (p > .1) + (p > .15) + (p > .2))
M$propensity <- run(left, right); M$propensity$p_head <- as.numeric(p[1:5]); M$propensity$sd_p <- sd(p)
M$propensity_1to2 <- run(left, right, ncontrols = 2)
# The matches that produce the book's matched sample bingeM
mk <- function(q, cc = FALSE) {
  z <- q$z; p <- q$p; dt <- q$dt
  left <- startcost(z)
  left <- addinteger(left, z, dt$ageC, penalty = 100); left <- addNearExact(left, z, dt$female, penalty = 10000)
  left <- addNearExact(left, z, dt$bpRX, penalty = 10000); left <- addNearExact(left, z, dt$vigor, penalty = 10)
  left <- addinteger(left, z, dt$smokenow, penalty = 10)
  left <- addMahal(left, z, cbind(dt$age, dt$bpRX, dt$female, dt$education, dt$smokenow,
                                  dt$smokeQuit, dt$bmi, dt$vigor, dt$waisthip))
  right <- addMahal(startcost(z), z, cbind(dt$female, dt$age, p))
  right <- addinteger(right, z, dt$education, penalty = 10); right <- addinteger(right, z, dt$smokenow, penalty = 1000)
  right <- addinteger(right, z, dt$smokeQuit, penalty = 10)
  right <- addcaliper(right, z, p, caliper = c(-1, .03), penalty = 10)
  controlcosts <- if (cc) ((p[z == 0] < .4) & (dt$age[z == 0] > 42)) * 1000 else NULL
  r <- run(left, right, controlcosts = controlcosts)
  r$left_sum <- sum(left); r$right_sum <- sum(right); r
}
M$never <- mk(prep("N")); M$past <- mk(prep("P"), cc = TRUE)
# The matched sample: the treated group and the controls chosen from each
# reservoir, as in the book's bingeM.
bm <- rbind(data.frame(SEQN = binge$SEQN[binge$AlcGroup == "B"], AlcGroup = "B", z = 1),
            data.frame(SEQN = M$never$controls, AlcGroup = "N", z = 0),
            data.frame(SEQN = M$past$controls, AlcGroup = "P", z = 0))
write.csv(bm, file.path(here, "rosenbaum_itos_matched.csv"), row.names = FALSE)
bm <- rd("rosenbaum_itos_matched.csv")
# The cost terms on six individuals
six <- rbind(head(binge[binge$AlcGroup == "B", ], 2), head(binge[binge$AlcGroup == "N", ], 4))
six <- six[order(six$SEQN), ]
z <- 1 * (six$AlcGroup == "B"); names(z) <- six$SEQN
M$terms <- list(
  seqn = six$SEQN, mahalanobis = addMahal(startcost(z), z, cbind(six$age, six$female)),
  near_exact = addNearExact(startcost(z), z, six$female),
  caliper_one_step = addcaliper(startcost(z), z, six$age, caliper = 10, twostep = FALSE),
  caliper = addcaliper(startcost(z), z, six$age, caliper = 10),
  caliper_asymmetric = addcaliper(startcost(z), z, six$age, caliper = c(-2, 10)),
  caliper_default = addcaliper(startcost(z), z, six$age),
  integer = addinteger(startcost(z), z, six$education, penalty = 3))
bpn <- binge[binge$AlcGroup != "N", ]; zz2 <- 1 * (bpn$AlcGroup == "B"); names(zz2) <- bpn$SEQN
M$terms$quantile_sum <- sum(addquantile(startcost(zz2), zz2, bpn$age, pct = c(1/4, 1/2, 3/4), penalty = 5))
out$matching <- M

## Tightening a block design ---------------------------------------------------
hb <- rd("rosenbaum_itos_blocks.csv"); rownames(hb) <- hb$SEQN
tight <- function(dat, x = NULL, f = NULL, ncontrols = 1, subset = NULL, pspace = 10) {
  z <- dat$z; names(z) <- rownames(dat); block <- dat$block
  left <- tightenBlock::startcost(z); right <- tightenBlock::startcost(z)
  if (!is.null(x)) left <- tightenBlock::addMahal(left, z, x)
  penalty <- (10 + ceiling(max(as.vector(left)))) * pspace
  if (!is.null(f)) {
    if (is.vector(f)) f <- matrix(f, length(f), 1)
    for (j in 1:ncol(f)) right <- tightenBlock::addNearExact(right, z, f[, j], penalty = penalty)
    penalty <- (penalty + ceiling(max(right))) * pspace
  }
  left <- tightenBlock::addNearExact(left, z, block, penalty = penalty)
  tc <- if (is.null(subset)) NULL else rep(subset, sum(z))
  r <- run(left, right, ncontrols = ncontrols, treatedcosts = tc, tighten = TRUE)
  r$block_penalty <- penalty; r
}
hb$bmicat <- (hb$bmi > 22.5) + (hb$bmi > 27.5) + (hb$bmi > 32.5)
Tt <- list()
Tt$bmi_1to2 <- tight(hb, x = cbind(hb$age, hb$education), f = cbind(hb$ibmi, hb$bmicat), ncontrols = 2)
dif1 <- tapply(((hb$z == 1) & (hb$dentist1Y == 0)), hb$block, sum) == 1
dif0 <- tapply(((hb$z == 0) & (hb$dentist1Y == 1)), hb$block, sum) >= 1
keep <- is.element(hb$block, as.numeric(names(dif1))[dif1 & dif0]) &
  (((hb$z == 1) & (hb$dentist1Y == 0)) | ((hb$z == 0) & (hb$dentist1Y == 1)))
he <- hb[keep, ]
write.csv(he[, c("SEQN", "z", "block", "age", "education")],
          file.path(here, "rosenbaum_itos_dentist.csv"), row.names = FALSE)
xe <- cbind(he$age, he$education)
Tt$dentist_all <- tight(he, x = xe, f = he$education)
Tt$dentist_150 <- tight(he, x = xe, f = he$education, subset = 150)
Tt$dentist_50 <- tight(he, x = xe, f = he$education, subset = 50)
Tt$dentist_n <- c(nrow(he), sum(he$z))
Tt$n_blocks <- sum(hb$z)
out$tighten <- Tt

## evalBal: balance in the matched sample --------------------------------------
bmf <- merge(bm, binge, by = "SEQN", sort = FALSE, suffixes = c("", ".y"))
vars <- c("age", "female", "education", "bmi", "waisthip", "vigor", "smokenow", "bpRX", "smokeQuit")
xBP <- bmf[bmf$AlcGroup != "N", vars]; zBP <- bmf$z[bmf$AlcGroup != "N"]
ee <- function(e) list(actual = as.list(e$actual), share_better = as.list(e$simBetter / nrow(e$sim)),
                       sim_median = as.list(apply(e$sim, 2, median)))
set.seed(5); B <- list()
B$auto <- ee(evalBal(zBP, xBP, reps = 2000))
B$fisher_all <- ee(evalBal(bmf$z, bmf[, vars], reps = 500, trunc = 1))
B$t <- ee(evalBal(zBP, xBP, reps = 200, statistic = "t"))
B$w <- ee(evalBal(zBP, xBP, reps = 200, statistic = "w"))
B$five_levels <- ee(evalBal(zBP, xBP, reps = 200, nunique = 5))
out$evalBal <- B

write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, force = TRUE),
      file.path(here, "rosenbaum_itos_R.json"))
