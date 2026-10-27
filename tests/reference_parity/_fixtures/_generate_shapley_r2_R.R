#!/usr/bin/env Rscript
# Second reference for sp.shapley_r2: R domir (general dominance =
# Shapley value of the R2 game) and relaimpo (LMG, the same Shapley
# value). Covers what Stata shapley2 cannot: analytic weights, and
# always-included controls (domir .all / relaimpo always=). Both
# packages are used as black boxes (outputs only).
#
# Install into a private library first, e.g.
#   install.packages(c("domir", "relaimpo"), lib = "<scratch>/rlib")
# Regenerate (from the repo root) with:
#   R_LIBS=<scratch>/rlib Rscript tests/reference_parity/_fixtures/_generate_shapley_r2_R.R

suppressPackageStartupMessages({
  library(domir)
  library(relaimpo)
  library(jsonlite)
})

here <- function(p) file.path("tests", "reference_parity", "_fixtures", p)
df <- read.csv(here("shapley_r2_data.csv"))

r2_lm <- function(fml, data) summary(lm(fml, data = data))$r.squared
r2_wlm <- function(fml, data) summary(lm(fml, data = data, weights = w))$r.squared
full <- y ~ x1 + x2 + x3 + x4 + x5 + x6
full_z <- y ~ z + x1 + x2 + x3 + x4 + x5 + x6

gd <- function(res) unname(res$General_Dominance)

out <- list()
d_plain <- domir(full, r2_lm, data = df)
out$ungrouped <- list(r2 = d_plain$Value, shapley = gd(d_plain))

d_group <- domir(full, r2_lm, data = df,
                 .set = list(g1 = ~ x1 + x2, g2 = ~ x3, g3 = ~ x4 + x5 + x6))
out$grouped <- list(r2 = d_group$Value, shapley = gd(d_group))

d_w <- domir(full, r2_wlm, data = df)
out$weighted <- list(r2 = d_w$Value, shapley = gd(d_w))

d_all <- domir(full_z, r2_lm, data = df, .all = ~ z)
out$fixed <- list(r2 = d_all$Value, r2_fixed = d_all$Value_All,
                  shapley = gd(d_all))

d_all_w <- domir(full_z, r2_wlm, data = df, .all = ~ z,
                 .set = list(g1 = ~ x1 + x2, g2 = ~ x3, g3 = ~ x4 + x5 + x6))
out$fixed_grouped_weighted <- list(r2 = d_all_w$Value,
                                   r2_fixed = d_all_w$Value_All,
                                   shapley = gd(d_all_w))

# relaimpo LMG on the same models (rela = FALSE keeps the R2 scale)
lmg <- calc.relimp(lm(full, data = df), type = "lmg", rela = FALSE)
out$relaimpo_lmg <- unname(lmg@lmg)
lmg_w <- calc.relimp(lm(full, data = df, weights = w), type = "lmg",
                     rela = FALSE)
out$relaimpo_lmg_weighted <- unname(lmg_w@lmg)
lmg_all <- calc.relimp(lm(y ~ z + x1 + x2 + x3 + x4 + x5 + x6, data = df),
                       type = "lmg", rela = FALSE, always = "z")
out$relaimpo_lmg_fixed <- unname(lmg_all@lmg)

out$versions <- list(R = R.version.string,
                     domir = as.character(packageVersion("domir")),
                     relaimpo = as.character(packageVersion("relaimpo")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           here("shapley_r2_R.json"))
