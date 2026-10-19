#!/usr/bin/env Rscript
# Frozen reference for design= on the DiD estimators (sp.callaway_santanna
# and sp.did_2x2) vs R did 2.x + survey 4.x.
#
# Regenerate (from the repository root):
#   Rscript tests/reference_parity/_generate_survey_did_R.R
#
# _fixtures/survey_did_data.csv is written once (fixed seed, 6-decimal
# rounding) and never overwritten afterwards.
#
# Design: 4 strata with 4-6 PSUs, 5-9 units per PSU, 5 periods; the unit is
# the sampled element (its weight, stratum and PSU do not change over time).
# Cohorts g in {0, 3, 4} (0 = never treated); time-invariant covariate x.
#
# What is frozen
# --------------
# * did::att_gt(est_method = "dr" and "reg", base_period = "varying",
#   weightsname = "w", bstrap = FALSE): its unit-level influence functions
#   (inffunc, n x #cells, with Var = IF'IF / n^2), and svyrecvar of IF / n
#   on the unit-level design -- the ATT(g, t) design SEs; the same for
#   aggte(type = "simple") and aggte(type = "dynamic") via their
#   inf.function.
# * 2x2 DID on periods 2 / 3, cohorts 0 and 3: svyglm(y ~ treat * post) on
#   the long design (the unit's two rows in its PSU).
# * MSM (sp.msm): _fixtures/survey_msm_data.csv, 4 periods per unit with a
#   time-varying confounder L. Stabilised weights by hand (pooled logits
#   A ~ A_lag + V and A ~ A_lag + V + L, cumulative product within unit),
#   then svyglm(y ~ cumA + V) and svyglm(yb ~ cumA + V, quasibinomial) on
#   the design with weights = survey weight x stabilised weight -- the
#   ipw::ipwtm + svyglm recipe.
suppressPackageStartupMessages({
  library(survey); library(jsonlite); library(did)
})

csv <- "tests/reference_parity/_fixtures/survey_did_data.csv"
if (!file.exists(csv)) {
  set.seed(20261020)
  n_psu <- c(4, 5, 6, 4)
  pop_mult <- c(5, 8, 3, 6)
  units <- list(); uid <- 0
  for (h in seq_along(n_psu)) {
    for (j in seq_len(n_psu[h])) {
      m <- sample(5:9, 1)
      a_psu <- rnorm(1, sd = 0.7)
      for (k in seq_len(m)) {
        uid <- uid + 1
        x <- round(rnorm(1, mean = 0.2 * h + 0.4 * a_psu), 6)
        pr <- plogis(-0.4 + 0.6 * x)
        gg <- sample(c(0, 3, 4), 1, prob = c(1 - pr, pr * 0.5, pr * 0.5))
        units[[uid]] <- data.frame(
          id = uid, stratum = h, psu = j, x = x, g = gg,
          a = rnorm(1) + a_psu,
          w = round(pop_mult[h] * runif(1, 5, 15) * (1 + 0.3 * (gg > 0)), 4),
          fpc_N = n_psu[h] * pop_mult[h] * 2)
      }
    }
  }
  U <- do.call(rbind, units)
  rows <- list()
  for (tt in 1:5) {
    eff <- ifelse(U$g > 0 & tt >= U$g, 1 + 0.5 * (tt - U$g) + 0.3 * U$x, 0)
    y <- U$a + 0.3 * tt + 0.4 * U$x * tt / 5 + eff + rnorm(nrow(U))
    rows[[tt]] <- data.frame(U[, c("id", "stratum", "psu", "x", "g", "w",
                                   "fpc_N")], t = tt, y = round(y, 6))
  }
  df <- do.call(rbind, rows)
  df <- df[order(df$id, df$t), ]
  write.csv(df, csv, row.names = FALSE)
}
df <- read.csv(csv)
dig <- function(x) unname(as.numeric(x))

units_all <- df[df$t == 1, ]
unit_designs <- function(ids) {
  # the unit-level design, rows in the order of the influence-function rows
  units <- units_all[match(ids, units_all$id), ]
  list(
    full = svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                     fpc = ~fpc_N, nest = TRUE, data = units),
    nofpc = svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                      nest = TRUE, data = units))
}
recvar <- function(u, des) {
  unname(svyrecvar(as.matrix(u), des$cluster, des$strata, des$fpc,
                   postStrata = des$postStrata))
}

cs_block <- function(est_method) {
  a <- att_gt(yname = "y", tname = "t", idname = "id", gname = "g",
              xformla = ~x, data = df, weightsname = "w", panel = TRUE,
              bstrap = FALSE, est_method = est_method,
              base_period = "varying", control_group = "nevertreated")
  # att_gt sorts the units (by cohort, then id): its processed data gives
  # the order of the influence-function rows
  dp <- as.data.frame(a$DIDparams$data)
  ids <- dp$id[dp$t == min(dp$t)]
  udes <- unit_designs(ids)
  IF <- as.matrix(a$inffunc)
  n <- nrow(IF)
  stopifnot(n == length(ids), n == nrow(units_all))
  s <- aggte(a, type = "simple", bstrap = FALSE)
  dy <- aggte(a, type = "dynamic", bstrap = FALSE)
  out <- list(group = dig(a$group), time = dig(a$t), att = dig(a$att),
              se_did = dig(a$se), simple = s$overall.att,
              dynamic_e = dig(dy$egt), dynamic_att = dig(dy$att.egt))
  for (nm in names(udes)) {
    des <- udes[[nm]]
    out[[nm]] <- list(
      se_cells = sqrt(diag(recvar(IF / n, des))),
      se_simple = sqrt(recvar(cbind(s$inf.function$simple.att) / n, des)[1, 1]),
      se_dynamic = sqrt(diag(recvar(dy$inf.function$dynamic.inf.func.e / n,
                                    des))),
      degf = degf(des))
  }
  out
}

res <- list(cs_dr = cs_block("dr"), cs_reg = cs_block("reg"))

sub <- df[df$t %in% c(2, 3) & df$g %in% c(0, 3), ]
sub$treat <- as.integer(sub$g == 3); sub$post <- as.integer(sub$t == 3)
res$did2x2 <- list()
for (nm in c("full", "nofpc")) {
  des <- if (nm == "full")
    svydesign(ids = ~psu, strata = ~stratum, weights = ~w, fpc = ~fpc_N,
              nest = TRUE, data = sub)
  else svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE,
                 data = sub)
  f <- svyglm(y ~ treat * post, design = des)
  res$did2x2[[nm]] <- list(att = unname(coef(f)["treat:post"]),
                           se = unname(SE(f)["treat:post"]),
                           degf = degf(des))
}

msm_csv <- "tests/reference_parity/_fixtures/survey_msm_data.csv"
if (!file.exists(msm_csv)) {
  set.seed(20261021)
  n_psu <- c(4, 5, 4, 5)
  rows <- list(); uid <- 0
  for (h in seq_along(n_psu)) {
    for (j in seq_len(n_psu[h])) {
      a_psu <- rnorm(1, sd = 0.5)
      for (k in seq_len(sample(6:10, 1))) {
        uid <- uid + 1
        a <- rnorm(1) + a_psu
        V <- round(rnorm(1), 6)
        w <- round((3 + h) * runif(1, 5, 15), 4)
        A_prev <- 0; cumA <- 0
        for (tt in 1:4) {
          L <- round(0.5 * A_prev + 0.5 * a + rnorm(1), 6)
          A <- rbinom(1, 1, plogis(-0.3 + 0.7 * L + 0.5 * A_prev + 0.2 * V))
          cumA <- cumA + A
          y <- round(1 + 0.8 * cumA + 0.5 * L + 0.3 * V + a + rnorm(1), 6)
          yb <- rbinom(1, 1, plogis(-1 + 0.4 * cumA + 0.3 * L))
          rows[[length(rows) + 1]] <- data.frame(
            id = uid, stratum = h, psu = j, w = w, t = tt, A = A, L = L,
            V = V, y = y, yb = yb, fpc_N = n_psu[h] * 6)
          A_prev <- A
        }
      }
    }
  }
  write.csv(do.call(rbind, rows), msm_csv, row.names = FALSE)
}
m <- read.csv(msm_csv)
m <- m[order(m$id, m$t), ]
m$A_lag <- ave(m$A, m$id, FUN = function(a) c(0, head(a, -1)))
pn <- predict(glm(A ~ A_lag + V, binomial, data = m,
                  control = glm.control(epsilon = 1e-14, maxit = 100)),
              type = "response")
pd <- predict(glm(A ~ A_lag + V + L, binomial, data = m,
                  control = glm.control(epsilon = 1e-14, maxit = 100)),
              type = "response")
ratio <- ifelse(m$A == 1, pn / pd, (1 - pn) / (1 - pd))
m$sw <- ave(ratio, m$id, FUN = cumprod)
m$cumA <- ave(m$A, m$id, FUN = cumsum)
m$wt <- m$w * m$sw
res$msm <- list()
for (nm in c("full", "nofpc")) {
  des <- if (nm == "full")
    svydesign(ids = ~psu, strata = ~stratum, weights = ~wt, fpc = ~fpc_N,
              nest = TRUE, data = m)
  else svydesign(ids = ~psu, strata = ~stratum, weights = ~wt, nest = TRUE,
                 data = m)
  g <- svyglm(y ~ cumA + V, design = des)
  tight <- glm.control(epsilon = 1e-14, maxit = 100)
  b <- do.call(svyglm, list(formula = yb ~ cumA + V, design = des,
                            family = quasibinomial(), control = tight))
  b <- do.call(svyglm, list(formula = yb ~ cumA + V, design = des,
                            family = quasibinomial(), control = tight,
                            start = unname(coef(b))))
  res$msm[[nm]] <- list(gaussian = list(coef = dig(coef(g)), se = dig(SE(g))),
                        binomial = list(coef = dig(coef(b)), se = dig(SE(b))),
                        degf = degf(des))
}

res$provenance <- list(
  r_version = R.version.string,
  survey_version = as.character(packageVersion("survey")),
  did_version = as.character(packageVersion("did")),
  data = "tests/reference_parity/_fixtures/survey_did_data.csv",
  generated_by = "tests/reference_parity/_generate_survey_did_R.R")

writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_did_R.json")
cat("wrote survey_did_R.json\n")
