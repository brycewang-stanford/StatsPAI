# Reference values for tests/reference_parity/test_did_synth_misc_parity.py
#
#   sp.harvest_did      building blocks vs did::att_gt(control_group =
#                       "notyettreated", base_period = "universal"); per-horizon
#                       aggregation (weighting = "n_treated") vs
#                       did::aggte(type = "dynamic")
#   sp.spillover_did    single cohort: fixest::feols(dbar ~ treat + ring1 + ring2,
#                       vcov = "hetero", ssc(adj = FALSE)) on the unit-level
#                       average long difference (the regression form of Butts's
#                       ring estimator, as in his rings_example.R);
#                       every design: did::att_gt(control_group = "nevertreated")
#                       + did::aggte(type = "simple") on {group} U {clean
#                       controls}, one call per group (direct, ring 1, ring 2)
#   sp.causal_impact    (a) the filter step: KFAS::KFS on the AR(1)-plus-noise
#                       state space with the parameters recomputed here from
#                       their closed forms (lm / cor / sd);
#                       (b) evidence only: CausalImpact::CausalImpact posterior
#                       summaries across seeds -- a different model, not a
#                       parity target (see the test docstring)
#
# The ring construction is recomputed here independently from the coordinates
# (Euclidean distance to the nearest treated unit, rings (0, 2] and (2, 4],
# clean controls beyond 4; a ring unit's exposure onset is the earliest cohort
# among treated units within the outer edge).
#
# Regenerate (from the repository root):
#   python  tests/reference_parity/_generate_did_synth_misc_data.py
#   Rscript tests/reference_parity/_generate_did_synth_misc_R.R

suppressMessages({
  library(did)
  library(fixest)
  library(KFAS)
  library(CausalImpact)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")
out <- list()

# ------------------------------------------------------------ harvest ------
h <- read.csv(file.path(fixtures, "did_synth_misc_harvest.csv"))
# did 2.3.0 recodes gname == 0 to Inf in place; on an integer column that
# assignment yields NA and the never-treated units are silently dropped
# ("Dropped ... observations that had missing data"). Store gname as double.
h$g <- as.numeric(h$g)
agt <- att_gt(yname = "y", tname = "t", idname = "id", gname = "g", data = h,
              control_group = "notyettreated", base_period = "universal",
              bstrap = FALSE, cband = FALSE)
dyn <- aggte(agt, type = "dynamic", bstrap = FALSE, cband = FALSE)
out$harvest <- list(
  cells = list(group = agt$group, t = agt$t, att = agt$att, se = agt$se),
  dynamic = list(e = dyn$egt, att = dyn$att.egt, se = dyn$se.egt)
)

# ------------------------------------------------------------ spillover ----
edges <- c(0, 2, 4)
rings_of <- function(d) {
  u <- d[!duplicated(d$id), c("id", "g", "cx", "cy")]
  u <- u[order(u$id), ]
  tr <- u$g != 0
  dist <- sqrt(outer(u$cx, u$cx[tr], "-")^2 + outer(u$cy, u$cy[tr], "-")^2)
  nearest <- apply(dist, 1, min)
  ring <- rep(-1L, nrow(u))
  for (r in seq_len(length(edges) - 1)) {
    ring[!tr & nearest > edges[r] & nearest <= edges[r + 1]] <- r
  }
  within_outer <- dist <= edges[length(edges)]
  onset <- apply(within_outer, 1, function(w) {
    if (any(w)) min(u$g[tr][w]) else 0
  })
  u$ring <- ifelse(tr, 0L, ring)          # 0 = treated, -1 = clean
  u$onset <- ifelse(tr | ring == -1L, 0, onset)
  u
}

did_simple <- function(d, units, gcol) {
  s <- d[d$id %in% units$id, ]
  s$gg <- as.numeric(units[[gcol]][match(s$id, units$id)])  # see note above
  a <- att_gt(yname = "y", tname = "t", idname = "id", gname = "gg", data = s,
              control_group = "nevertreated", base_period = "universal",
              bstrap = FALSE, cband = FALSE)
  ag <- aggte(a, type = "simple", bstrap = FALSE, cband = FALSE)
  keep <- a$t >= a$group
  list(att = ag$overall.att, se = ag$overall.se,
       cell_group = a$group[keep], cell_t = a$t[keep],
       cell_att = a$att[keep], cell_se = a$se[keep],
       n_units = sum(units[[gcol]] != 0))
}

spill_case <- function(file) {
  d <- read.csv(file.path(fixtures, file))
  u <- rings_of(d)
  clean <- u[u$ring == -1L, ]
  clean$gdir <- 0; clean$gring <- 0
  res <- list(n_clean = nrow(clean),
              ring_counts = sapply(1:2, function(r) sum(u$ring == r)))
  tr <- u[u$ring == 0L, ]; tr$gdir <- tr$g
  res$direct <- did_simple(d, rbind(tr[, c("id", "gdir")],
                                    clean[, c("id", "gdir")]), "gdir")
  for (r in 1:2) {
    rr <- u[u$ring == r, ]; rr$gring <- rr$onset
    res[[paste0("ring_", r)]] <- did_simple(
      d, rbind(rr[, c("id", "gring")], clean[, c("id", "gring")]), "gring")
  }
  res
}

out$spill_single <- spill_case("did_synth_misc_spill_single.csv")
out$spill_stag <- spill_case("did_synth_misc_spill_stag.csv")

# Regression form (single cohort, treated at 3, base period 2).
d1 <- read.csv(file.path(fixtures, "did_synth_misc_spill_single.csv"))
u1 <- rings_of(d1)
w <- reshape(d1[, c("id", "t", "y")], idvar = "id", timevar = "t",
             direction = "wide")
w <- w[order(w$id), ]
u1$dbar <- ((w$y.3 - w$y.2) + (w$y.4 - w$y.2)) / 2
u1$treat <- as.integer(u1$ring == 0L)
u1$ring1 <- as.integer(u1$ring == 1L)
u1$ring2 <- as.integer(u1$ring == 2L)
m <- feols(dbar ~ treat + ring1 + ring2, data = u1, vcov = "hetero",
           ssc = ssc(adj = FALSE, cluster.adj = FALSE))
out$spill_single_feols <- list(
  coef = as.list(coef(m)[c("treat", "ring1", "ring2")]),
  se = as.list(se(m)[c("treat", "ring1", "ring2")]),
  vcov = "hetero, ssc(adj = FALSE) = HC0"
)

# ------------------------------------------------------------ causal impact
ci <- read.csv(file.path(fixtures, "did_synth_misc_impact.csv"))
pre <- ci$t < 71
X <- cbind(1, ci$x1, ci$x2)
beta <- coef(lm(y ~ x1 + x2, data = ci[pre, ]))
res <- ci$y[pre] - drop(X[pre, ] %*% beta)
npre <- sum(pre)
rho <- cor(res[-npre], res[-1])
rho <- max(min(rho, 0.99), -0.99)
so <- sd(res)
ss <- so * sqrt(1 - rho^2)
yr <- ci$y - drop(X %*% beta)
yr[!pre] <- NA
mod <- SSModel(yr ~ -1 + SSMcustom(Z = matrix(1), T = matrix(rho),
                                   R = matrix(1), Q = matrix(ss^2),
                                   a1 = matrix(0), P1 = matrix(ss^2),
                                   P1inf = matrix(0)),
               H = matrix(so^2))
kf <- KFS(mod, filtering = "state", smoothing = "none")
n <- nrow(ci)
a <- kf$a[1:n, 1]
P <- kf$P[1, 1, 1:n]
out$impact_filter <- list(
  beta = unname(beta), rho = rho, sigma_obs = so, sigma_state = ss,
  y_pred = unname(drop(X %*% beta) + a),
  y_pred_se = unname(sqrt(P + so^2))
)

ci_runs <- lapply(1:4, function(s) {
  set.seed(s)
  fit <- CausalImpact(ci[, c("y", "x1", "x2")], pre.period = c(1, 70),
                      post.period = c(71, 100),
                      model.args = list(niter = 5000))
  sm <- fit$summary
  list(seed = s, niter = 5000,
       avg_abs_effect = sm["Average", "AbsEffect"],
       avg_abs_effect_sd = sm["Average", "AbsEffect.sd"],
       cum_abs_effect = sm["Cumulative", "AbsEffect"],
       cum_abs_effect_sd = sm["Cumulative", "AbsEffect.sd"])
})
out$causalimpact_evidence <- ci_runs

# ------------------------------------------------------------ meta ---------
out$meta <- list(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  did_version = as.character(packageVersion("did")),
  DRDID_version = as.character(packageVersion("DRDID")),
  fixest_version = as.character(packageVersion("fixest")),
  KFAS_version = as.character(packageVersion("KFAS")),
  CausalImpact_version = as.character(packageVersion("CausalImpact")),
  bsts_version = as.character(packageVersion("bsts")),
  generated_by = "_generate_did_synth_misc_R.R"
)

write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null"),
      file.path(fixtures, "did_synth_misc_R.json"))
cat("wrote", file.path(fixtures, "did_synth_misc_R.json"), "\n")
