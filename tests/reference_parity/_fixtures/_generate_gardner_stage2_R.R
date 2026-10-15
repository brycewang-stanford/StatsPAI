# Reference for sp.gardner_did(vce='stage2'): the second-stage-only
# clustered standard error, which treats the first stage as known.
#
# This is the number a user gets by running Gardner's two stages by hand
# and reading the second regression's clustered SE. It understates (it
# ignores the estimation of the first-stage fixed effects; the corrected
# variance is vce='analytic', pinned by Track A module 73), and
# sp.gardner_did keeps it only to reproduce earlier output. What is pinned
# here is that the legacy number is the convention it documents:
#
# * static:   lm(y_tilde ~ D) with sandwich::vcovCL(type = "HC1",
#             cadjust = TRUE) clustered on the county, the
#             G/(G-1) (n-1)/(n-k) factor of Stata's regress, vce(cluster);
# * dynamic:  lm(y_tilde ~ 0 + factor(rel)) on the eventually-treated rows
#             at the reported relative times, with
#             sandwich::vcovCL(type = "HC0", cadjust = FALSE), i.e. no
#             small-sample factor.
#
# y_tilde = lemp - Y0_hat, Y0_hat from lm(lemp ~ factor(countyreal) +
# factor(year)) on the untreated rows (exact least squares, no iteration).
# Data: Track A module 73's CSV bytes (sp.datasets.mpdta()).
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_gardner_stage2_R.R
suppressPackageStartupMessages({
  library(sandwich)
  library(jsonlite)
})

df <- read.csv("tests/r_parity/data/73_did2s.csv")
df$D <- as.integer(df$first_treat > 0 & df$year >= df$first_treat)
fs <- lm(lemp ~ factor(countyreal) + factor(year), data = df[df$D == 0, ])
df$y_tilde <- df$lemp - predict(fs, newdata = df)

static <- lm(y_tilde ~ D, data = df)
v_static <- vcovCL(static, cluster = ~countyreal, type = "HC1", cadjust = TRUE)

ev <- df[df$first_treat > 0, ]
ev$rel <- ev$year - ev$first_treat
ev <- ev[ev$rel >= -5 & ev$rel <= 5, ]
dyn <- lm(y_tilde ~ 0 + factor(rel), data = ev)
v_dyn <- vcovCL(dyn, cluster = ~countyreal, type = "HC0", cadjust = FALSE)
rel <- as.integer(sub("^factor\\(rel\\)", "", names(coef(dyn))))

out <- list(
  static = list(att = unname(coef(static)["D"]),
                se = sqrt(v_static["D", "D"]),
                n = nobs(static)),
  event_study = setNames(
    lapply(seq_along(rel), function(j) list(
      att = unname(coef(dyn)[j]), se = sqrt(v_dyn[j, j]))),
    as.character(rel)),
  `_meta` = list(
    generated_by = "_generate_gardner_stage2_R.R",
    sandwich = as.character(packageVersion("sandwich")),
    r_version = R.version.string
  )
)
write_json(out, "tests/reference_parity/_fixtures/gardner_stage2_R.json",
           digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("[gardner-stage2] wrote\n")
