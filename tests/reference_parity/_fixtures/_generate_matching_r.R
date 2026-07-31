#!/usr/bin/env Rscript
# Reference values for the StatsPAI matching / weighting module.
#
# Data: MatchIt::lalonde (614 obs, 185 treated), the canonical matching
# benchmark, with the `race` factor expanded to explicit black / hispan
# dummies so the design matrix is unambiguous across languages. The frame
# is committed as `matching_lalonde.csv`; regenerate both together with
#
#   Rscript tests/reference_parity/_fixtures/_generate_matching_r.R
#
# Reference packages (see R_PACKAGE_VERSIONS.md):
#   CBPS     — Imai & Ratkovic (2014) covariate balancing propensity score
#   ebal     — Hainmueller (2012) entropy balancing
#   MatchIt  — Ho, Imai, King & Stuart nearest / optimal matching
#   optmatch — Hansen & Klopfer optimal full / pair matching
#
# Every number written here is a package output, not a hand computation,
# except the Hajek weighted contrasts, which are one-liners over the
# package-returned weights and are spelled out below.

suppressPackageStartupMessages({
  library(MatchIt)
  library(CBPS)
  library(ebal)
  library(optmatch)
  library(jsonlite)
})

FIX <- "tests/reference_parity/_fixtures"
CSV <- file.path(FIX, "matching_lalonde.csv")
OUT <- file.path(FIX, "matching_R.json")

if (file.exists(CSV)) {
  df <- read.csv(CSV)
} else {
  data(lalonde, package = "MatchIt")
  df <- lalonde
  df$black <- as.integer(df$race == "black")
  df$hispan <- as.integer(df$race == "hispan")
  df$race <- NULL
  df <- df[, c("treat", "age", "educ", "married", "nodegree",
               "re74", "re75", "black", "hispan", "re78")]
  write.csv(df, CSV, row.names = FALSE)
}

COV <- c("age", "educ", "married", "nodegree", "re74", "re75", "black", "hispan")
FML <- as.formula(paste("treat ~", paste(COV, collapse = " + ")))
X <- as.matrix(df[, COV])
Tr <- df$treat
Y <- df$re78

res <- list()

# Hajek (self-normalised) weighted contrast: the estimand every weighting
# method below feeds into, and the one sp.cbps / sp.ebalance report.
hajek <- function(w) {
  t1 <- Tr == 1
  t0 <- Tr == 0
  sum(Y[t1] * w[t1]) / sum(w[t1]) - sum(Y[t0] * w[t0]) / sum(w[t0])
}

## --- 1. CBPS ----------------------------------------------------------
# standardize = FALSE keeps the raw (unnormalised) weights so the Hajek
# contrast below is the only normalisation applied.
for (est in c("ATT", "ATE")) {
  for (meth in c("over", "exact")) {
    key <- paste0("cbps_", tolower(est), "_", meth)
    fit <- CBPS::CBPS(FML, data = df, ATT = if (est == "ATT") 1 else 0,
                      method = meth, standardize = FALSE)
    res[[key]] <- list(
      coefficients = as.numeric(coef(fit)),
      att = hajek(fit$weights)
    )
  }
}

## --- 2. Entropy balancing --------------------------------------------
eb <- ebal::ebalance(Treatment = Tr, X = X, print.level = -1)
w_eb <- numeric(length(Tr))
w_eb[Tr == 1] <- 1
w_eb[Tr == 0] <- eb$w
res$ebal_att <- list(
  att = hajek(w_eb),
  # ebal's own achieved balance, for the "we are at least as exact" check
  bal_treated = as.numeric(colMeans(X[Tr == 1, , drop = FALSE])),
  bal_control_weighted = as.numeric(
    colSums(X[Tr == 0, , drop = FALSE] * eb$w) / sum(eb$w)
  ),
  w_control_normalised = as.numeric(eb$w / sum(eb$w))
)

## --- 3. MatchIt nearest-neighbour configurations ----------------------
matchit_att <- function(m) {
  md <- MatchIt::match.data(m, data = df, drop.unmatched = TRUE)
  w <- md$weights
  sum(md$re78[md$treat == 1] * w[md$treat == 1]) / sum(w[md$treat == 1]) -
    sum(md$re78[md$treat == 0] * w[md$treat == 0]) / sum(w[md$treat == 0])
}
ps_cfgs <- list(
  nn_ps_noreplace_1 = list(replace = FALSE, ratio = 1L),
  nn_ps_noreplace_2 = list(replace = FALSE, ratio = 2L),
  nn_ps_noreplace_3 = list(replace = FALSE, ratio = 3L)
)
for (nm in names(ps_cfgs)) {
  m <- do.call(MatchIt::matchit, c(
    list(formula = FML, data = df, method = "nearest", distance = "glm",
         link = "logit", estimand = "ATT"), ps_cfgs[[nm]]
  ))
  md <- MatchIt::match.data(m, data = df, drop.unmatched = TRUE)
  res[[paste0("matchit_", nm)]] <- list(
    att = matchit_att(m),
    n_matched_treated = sum(md$treat == 1),
    n_matched_control = sum(md$treat == 0)
  )
}

## --- 4. Mahalanobis metric -------------------------------------------
# The treated x control Mahalanobis distance matrix MatchIt itself builds.
# Pinning the metric (rather than a downstream ATT) isolates the distance
# definition from MatchIt's internal assignment heuristics.
M <- as.matrix(getFromNamespace("mahalanobis_dist", "MatchIt")(FML, data = df))
res$mahalanobis_dist <- list(
  d_00 = M[1, 1], d_01 = M[1, 2], d_10 = M[2, 1],
  frobenius = sqrt(sum(M^2)),
  row0_min = min(M[1, ]), row0_argmin = which.min(M[1, ]) - 1L
)

# Greedy nearest-neighbour without replacement, run by MatchIt on a
# *supplied* distance matrix so the comparison is purely about the
# assignment rule. Row/col order is treated-then-control as in `M`.
Dm <- M
rownames(Dm) <- which(Tr == 1)
colnames(Dm) <- which(Tr == 0)
for (mo in c("data", "closest", "farthest")) {
  m <- MatchIt::matchit(treat ~ age, data = df, method = "nearest",
                        distance = Dm, replace = FALSE, ratio = 1L,
                        estimand = "ATT", m.order = mo)
  res[[paste0("greedy_supplied_D_", mo)]] <- list(att = matchit_att(m))
}

## --- 5. Optimal matching ---------------------------------------------
df$ps <- as.numeric(fitted(glm(FML, data = df, family = binomial(link = "logit"))))
pm <- optmatch::pairmatch(optmatch::match_on(treat ~ ps, data = df), data = df)
ok <- !is.na(pm)
sub <- df[ok, ]
g <- droplevels(pm[ok])
tot <- 0
diffs <- c()
for (lv in levels(g)) {
  ii <- which(g == lv)
  tot <- tot + abs(sub$ps[ii][sub$treat[ii] == 1] - sub$ps[ii][sub$treat[ii] == 0])
  diffs <- c(diffs, sub$re78[ii][sub$treat[ii] == 1] - sub$re78[ii][sub$treat[ii] == 0])
}
res$optmatch_pair_ps <- list(
  att = mean(diffs),
  total_distance = tot,
  n_pairs = length(levels(g))
)

## --- 6. Logistic propensity score (shared anchor) ---------------------
fit_ps <- glm(FML, data = df, family = binomial(link = "logit"))
res$logit_ps <- list(
  coefficients = as.numeric(coef(fit_ps)),
  coef_names = names(coef(fit_ps)),
  sd_ps = sd(fitted(fit_ps))
)

res$`_meta` <- list(
  generated_by = "_generate_matching_r.R",
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  packages = list(
    MatchIt = as.character(packageVersion("MatchIt")),
    CBPS = as.character(packageVersion("CBPS")),
    ebal = as.character(packageVersion("ebal")),
    optmatch = as.character(packageVersion("optmatch"))
  )
)

cat(jsonlite::toJSON(res, digits = 17, auto_unbox = TRUE, pretty = TRUE),
    file = OUT)
cat("wrote", OUT, "\n")
