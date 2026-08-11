# Reference values for sp.staggered_rollout / staggered_cs / staggered_sa
# against R `staggered` 1.2.2 (Roth & Sant'Anna 2023).
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_staggered_extended_R.R
#
# Writes:
#   _fixtures/staggered_extended_reference.json   all reference numbers
#   _fixtures/staggered_rollout_panel.csv         a *randomised* rollout panel
#   _fixtures/staggered_fisher_permutations.csv   permutation draws + per-draw fits
#
# The second panel matters: mpdta has a never-treated cohort (g = Inf), so it
# never exercises the finite-max(g) branch of the weight construction. The
# randomised panel has no never-treated units at all.

suppressMessages({
  library(staggered)
  library(jsonlite)
})

stopifnot(as.character(packageVersion("staggered")) == "1.2.2")

here <- function(...) file.path(dirname(sys.frame(1)$ofile %||% "."), ...)
`%||%` <- function(a, b) if (is.null(a)) b else a

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")
dir.create(fixtures, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------- data ----
mpdta_path <- file.path(script_dir, "..", "orig_parity", "data",
                        "02_mpdta_original.csv")
mpdta <- read.csv(mpdta_path)
# The locked fixture spells the column first_treat (did::mpdta uses
# first.treat); never-treated arrive as 0 and must become Inf for `staggered`.
mpdta$g <- ifelse(mpdta$first_treat == 0, Inf, mpdta$first_treat)

# A genuinely randomised rollout: adoption date assigned at random, no
# never-treated units, and a real treatment effect that grows with exposure.
set.seed(20260811)
n_units <- 240
periods <- 1:6
cohorts <- c(3, 4, 5, 6)
unit_g <- sample(rep(cohorts, length.out = n_units))
unit_fe <- rnorm(n_units, sd = 0.8)
rollout <- do.call(rbind, lapply(seq_len(n_units), function(u) {
  g <- unit_g[u]
  rel <- periods - g
  y <- unit_fe[u] + 0.15 * periods + ifelse(rel >= 0, 0.4 + 0.1 * rel, 0) +
    rnorm(length(periods), sd = 0.5)
  data.frame(unit = u, time = periods, first_treat = g, y = y)
}))
write.csv(rollout, file.path(fixtures, "staggered_rollout_panel.csv"),
          row.names = FALSE)

# The same design with the treatment effect switched off. A randomisation test
# on a panel with a strong effect returns p = 0 for every implementation, which
# pins nothing; the null panel puts the p-value in the interior where an
# implementation error would actually show up.
set.seed(20260812)
unit_g0 <- sample(rep(cohorts, length.out = n_units))
unit_fe0 <- rnorm(n_units, sd = 0.8)
nullpanel <- do.call(rbind, lapply(seq_len(n_units), function(u) {
  y <- unit_fe0[u] + 0.15 * periods + rnorm(length(periods), sd = 0.5)
  data.frame(unit = u, time = periods, first_treat = unit_g0[u], y = y)
}))
write.csv(nullpanel, file.path(fixtures, "staggered_null_panel.csv"),
          row.names = FALSE)

datasets <- list(
  mpdta = list(df = mpdta, i = "countyreal", t = "year", g = "g", y = "lemp"),
  rollout = list(df = rollout, i = "unit", t = "time", g = "first_treat",
                 y = "y"),
  nullpanel = list(df = nullpanel, i = "unit", t = "time", g = "first_treat",
                   y = "y")
)

fit <- function(d, ...) {
  r <- staggered::staggered(df = d$df, i = d$i, t = d$t, g = d$g, y = d$y, ...)
  list(estimate = r$estimate, se = r$se, se_neyman = r$se_neyman)
}

out <- list()

# ------------------------------------------- estimand x beta x controls ----
grid <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]
  for (est in c("simple", "cohort", "calendar")) {
    for (bl in list(list(tag = "efficient", b = NULL),
                    list(tag = "plugin", b = 1))) {
      for (ulto in c(FALSE, TRUE)) {
        key <- paste(dname, est, bl$tag, ifelse(ulto, "last", "notyet"),
                     sep = "|")
        grid[[key]] <- fit(d, estimand = est, beta = bl$b,
                           use_last_treated_only = ulto)
      }
    }
  }
}
out$grid <- grid

# ------------------------------------------- general control set (A_0) ----
# use_DiD_A0 = FALSE uses EVERY pre-period as a control instead of the single
# g-1 DiD contrast, so beta becomes a vector. This is the general form of the
# estimator; the DiD control set is the special case.
general <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]
  for (est in c("simple", "cohort", "calendar")) {
    for (bl in list(list(tag = "efficient", b = NULL),
                    list(tag = "plugin", b = 1))) {
      key <- paste(dname, est, bl$tag, sep = "|")
      res <- tryCatch(
        fit(d, estimand = est, beta = bl$b, use_DiD_A0 = FALSE),
        error = function(err) NULL
      )
      if (!is.null(res)) general[[key]] <- res
    }
  }
}
out$general_a0 <- general

# ------------------------------------------------------- event studies ----
es <- list()
es_feasible <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]
  for (e in -3:4) {
    for (bl in list(list(tag = "efficient", b = NULL),
                    list(tag = "plugin", b = 1))) {
      key <- paste(dname, "e", e, bl$tag, sep = "|")
      res <- tryCatch(
        fit(d, estimand = "eventstudy", eventTime = e, beta = bl$b),
        error = function(err) NULL
      )
      if (!is.null(res)) {
        es[[key]] <- res
        es_feasible[[paste(dname, bl$tag, sep = "|")]] <-
          c(es_feasible[[paste(dname, bl$tag, sep = "|")]], e)
      }
    }
  }
}
out$eventstudy <- es
out$eventstudy_feasible <- es_feasible

# ---------------------------------------------- event-study joint vcv -----
vcvs <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]
  ev <- es_feasible[[paste(dname, "efficient", sep = "|")]]
  ev <- ev[ev >= -1 & ev <= 2]
  if (length(ev) < 2) next
  for (bl in list(list(tag = "efficient", b = NULL),
                  list(tag = "plugin", b = 1))) {
    r <- staggered::staggered(
      df = d$df, i = d$i, t = d$t, g = d$g, y = d$y,
      estimand = "eventstudy", eventTime = ev, beta = bl$b,
      return_full_vcv = TRUE
    )
    vcvs[[paste(dname, bl$tag, sep = "|")]] <- list(
      event_time = ev,
      estimate = r$resultsDF$estimate,
      se = r$resultsDF$se,
      se_neyman = r$resultsDF$se_neyman,
      vcv = r$vcv,
      vcv_neyman = r$vcv_neyman
    )
  }
}
out$eventstudy_vcv <- vcvs

# ------------------------------------------------- cs / sa convenience ----
wrappers <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]
  for (est in c("simple", "cohort", "calendar")) {
    rcs <- staggered::staggered_cs(df = d$df, i = d$i, t = d$t, g = d$g,
                                   y = d$y, estimand = est)
    rsa <- staggered::staggered_sa(df = d$df, i = d$i, t = d$t, g = d$g,
                                   y = d$y, estimand = est)
    wrappers[[paste(dname, est, "cs", sep = "|")]] <-
      list(estimate = rcs$estimate, se = rcs$se, se_neyman = rcs$se_neyman)
    wrappers[[paste(dname, est, "sa", sep = "|")]] <-
      list(estimate = rsa$estimate, se = rsa$se, se_neyman = rsa$se_neyman)
  }
}
out$wrappers <- wrappers

# ------------------------------------------------------------- fisher ----
# The package draws its own permutations with set.seed(k) internally, which no
# other language reproduces bit-for-bit. So we pin two things separately:
#   (a) a run of the package's own randomisation test, to compare against
#       within Monte-Carlo error, and
#   (b) an explicit set of permutations we control, with R's per-draw fits, so
#       the *estimator under permutation* can be pinned exactly.
fisher_pkg <- list()
for (dname in names(datasets)) {
  d <- datasets[[dname]]
  r <- staggered::staggered(df = d$df, i = d$i, t = d$t, g = d$g, y = d$y,
                            estimand = "simple", compute_fisher = TRUE,
                            num_fisher_permutations = 2000)
  fisher_pkg[[dname]] <- list(
    estimate = r$estimate, se = r$se, se_neyman = r$se_neyman,
    fisher_pval = r$fisher_pval,
    fisher_pval_se_neyman = r$fisher_pval_se_neyman,
    num_fisher_permutations = r$num_fisher_permutations
  )
}
out$fisher_package <- fisher_pkg

# Explicit permutations on the null panel, where the p-value is interior.
d <- datasets$nullpanel
ig <- unique(d$df[, c("unit", "first_treat")])
ig <- ig[order(ig$unit), ]
n_i <- nrow(ig)
B <- 40
perms <- matrix(NA_integer_, nrow = B, ncol = n_i)
draw <- data.frame(estimate = numeric(B), se = numeric(B),
                   se_neyman = numeric(B))
for (k in seq_len(B)) {
  set.seed(1000 + k)
  p <- sample(n_i)
  perms[k, ] <- p
  permuted <- d$df
  gmap <- setNames(ig$first_treat[p], ig$unit)
  permuted$first_treat <- as.numeric(gmap[as.character(permuted$unit)])
  r <- staggered::staggered(df = permuted, i = "unit", t = "time",
                            g = "first_treat", y = "y", estimand = "simple")
  draw$estimate[k] <- r$estimate
  draw$se[k] <- r$se
  draw$se_neyman[k] <- r$se_neyman
}
perm_df <- as.data.frame(perms)
colnames(perm_df) <- paste0("p", seq_len(n_i))
perm_df$estimate <- draw$estimate
perm_df$se <- draw$se
perm_df$se_neyman <- draw$se_neyman
write.csv(perm_df, file.path(fixtures, "staggered_fisher_permutations.csv"),
          row.names = FALSE)

out$fisher_explicit <- list(
  units = ig$unit,
  cohort_of_unit = ig$first_treat,
  n_permutations = B
)

# -------------------------------------------------------- edge cases -----
# (a) A cohort with a single unit: `staggered` warns and drops the cohort.
singleton <- nullpanel
singleton$first_treat[singleton$unit == 1] <- 2
write.csv(singleton, file.path(fixtures, "staggered_singleton_panel.csv"),
          row.names = FALSE)
r <- staggered::staggered(df = singleton, i = "unit", t = "time",
                          g = "first_treat", y = "y", estimand = "simple")
out$singleton <- list(estimate = r$estimate, se = r$se, se_neyman = r$se_neyman)

# (b) Units already treated in the first period: staggered_cs / staggered_sa
#     drop them, plain staggered() keeps them. Pin both so the difference is
#     locked down rather than assumed.
early <- nullpanel
early$first_treat[early$unit <= 40] <- 1
write.csv(early, file.path(fixtures, "staggered_early_panel.csv"),
          row.names = FALSE)
r_plain <- staggered::staggered(df = early, i = "unit", t = "time",
                                g = "first_treat", y = "y",
                                estimand = "simple", beta = 1)
r_cs <- staggered::staggered_cs(df = early, i = "unit", t = "time",
                                g = "first_treat", y = "y",
                                estimand = "simple")
r_sa <- staggered::staggered_sa(df = early, i = "unit", t = "time",
                                g = "first_treat", y = "y",
                                estimand = "simple")
out$early <- list(
  plain = list(estimate = r_plain$estimate, se = r_plain$se,
               se_neyman = r_plain$se_neyman),
  cs = list(estimate = r_cs$estimate, se = r_cs$se,
            se_neyman = r_cs$se_neyman),
  sa = list(estimate = r_sa$estimate, se = r_sa$se,
            se_neyman = r_sa$se_neyman)
)

# ------------------------------------------------------------- write -----
out$meta <- list(
  staggered_version = as.character(packageVersion("staggered")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  generated_by = "_generate_staggered_extended_R.R"
)

write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(fixtures, "staggered_extended_reference.json"))
cat("wrote", file.path(fixtures, "staggered_extended_reference.json"), "\n")
