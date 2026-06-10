# Original-data parity for E-value sensitivity analysis (VanderWeele &
# Ding, Ann Intern Med 2017) on the REAL NHEFS / *What If* data.
#
#   Source data : sp.datasets.nhefs() FULL (n=1629; death is complete),
#                 dumped by the Python side to data/11_nhefs_evalue.csv so
#                 both languages read identical bytes.
#   Gold engine : the EValue R package (evalues.RR / evalues.MD) plus a
#                 hand-rolled stabilized-IPW risk ratio (base R glm), the
#                 same procedure the book uses for binary outcomes.
#
# The E-value is the deterministic VanderWeele-Ding closed form
#   E = RR + sqrt(RR*(RR-1))   (applied to 1/RR when RR<1),
# so this R gold must match StatsPAI's sp.evalue to ~1e-10.
#
#   PRIMARY  : crude mortality RR (qsmk) and IP-weighted mortality RR.
#   SECONDARY: Ch12 weight effect via the SMD approx RR=exp(0.91*d).
# Writes results/11_nhefs_evalue_R.json.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
HERE <- .script_dir
DATA_DIR <- file.path(HERE, "data")
RESULTS_DIR <- file.path(HERE, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(EValue)
  library(jsonlite)
})

MODULE <- "11_nhefs_evalue"
VD_SMD_COEF <- 0.91
df <- read.csv(file.path(DATA_DIR, paste0(MODULE, ".csv")))
n_full <- nrow(df)

A <- df$qsmk
Y <- df$death

# ---- PRIMARY: crude mortality risk ratio -----------------------------
risk1 <- mean(Y[A == 1]); risk0 <- mean(Y[A == 0])
rr_crude <- risk1 / risk0
n1 <- sum(A == 1); n0 <- sum(A == 0)
se_log <- sqrt((1 - risk1) / (risk1 * n1) + (1 - risk0) / (risk0 * n0))
z <- qnorm(0.975)
lo_c <- rr_crude * exp(-z * se_log)
hi_c <- rr_crude * exp(z * se_log)

ev_crude <- evalues.RR(est = rr_crude, lo = lo_c, hi = hi_c)
# evalues.RR returns a matrix: row "E-values", cols point/lower/upper.
ev_crude_point <- ev_crude["E-values", "point"]
ev_crude_ci    <- ev_crude["E-values", "lower"]  # CI limit closest to null

# Closed-form anchor.
rr_used <- if (rr_crude >= 1) rr_crude else 1 / rr_crude
closed_crude <- rr_used + sqrt(rr_used * (rr_used - 1))

# ---- PRIMARY: IP-weighted (stabilized) mortality risk ratio ----------
conf <- paste0(
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)
ipw_rr <- function(d) {
  den <- glm(as.formula(paste("qsmk ~", conf)), data = d, family = binomial())
  pd  <- predict(den, type = "response")
  pn  <- mean(d$qsmk)
  sw  <- ifelse(d$qsmk == 1, pn / pd, (1 - pn) / (1 - pd))
  r1  <- sum(sw * d$qsmk * d$death) / sum(sw * d$qsmk)
  r0  <- sum(sw * (1 - d$qsmk) * d$death) / sum(sw * (1 - d$qsmk))
  r1 / r0
}
rr_adj <- ipw_rr(df)
set.seed(42)
B <- 1000
boot <- replicate(B, {
  idx <- sample.int(n_full, n_full, replace = TRUE)
  tryCatch(ipw_rr(df[idx, , drop = FALSE]), error = function(e) NA_real_)
})
boot <- boot[is.finite(boot)]
lo_a <- as.numeric(quantile(boot, 0.025))
hi_a <- as.numeric(quantile(boot, 0.975))
ev_adj <- evalues.RR(est = rr_adj, lo = lo_a, hi = hi_a)
ev_adj_point <- ev_adj["E-values", "point"]
ev_adj_ci    <- ev_adj["E-values", "upper"]  # RR<1 -> upper limit nearest null

# ---- SECONDARY: Ch12 weight effect via SMD approximation -------------
# Use the same complete-case weight effect StatsPAI reports (3.4-3.44 kg).
cc <- df[!is.na(df$wt82_71), ]
den <- glm(as.formula(paste("qsmk ~", conf)), data = cc, family = binomial())
pd  <- predict(den, type = "response")
pn  <- mean(cc$qsmk)
sw  <- ifelse(cc$qsmk == 1, pn / pd, (1 - pn) / (1 - pd))
msm <- lm(wt82_71 ~ qsmk, data = cc, weights = sw)
effect <- unname(coef(msm)["qsmk"])
sd_out <- sd(cc$wt82_71)
d_smd  <- effect / sd_out
rr_approx <- exp(VD_SMD_COEF * d_smd)
# EValue's evalues.MD standardizes MD internally as exp(0.91*d); here we
# already standardized, so feed evalues.RR(exp(0.91 d)) for a like-for-like
# check, AND evalues.MD on the raw (already-standardized) effect with se=0.
ev_smd <- evalues.RR(est = rr_approx)
ev_smd_point <- ev_smd["E-values", "point"]
ev_md  <- evalues.MD(est = d_smd, se = 0)        # exp(0.91*d) internally
ev_md_point <- ev_md["E-values", "point"]

rows <- list(
  list(module = MODULE, side = "R", statistic = "evalue_crude_rr_point",
       estimate = ev_crude_point, se = NULL, n = n_full,
       published = closed_crude,
       citation = "EValue::evalues.RR on crude mortality RR; closed form E=RR+sqrt(RR(RR-1))",
       extra = list(rr = rr_crude, rr_ci = c(lo_c, hi_c),
                    evalue_ci = ev_crude_ci, risk1 = risk1, risk0 = risk0)),
  list(module = MODULE, side = "R", statistic = "evalue_ipw_rr_point",
       estimate = ev_adj_point, se = NULL, n = n_full, published = 1.0,
       citation = "EValue::evalues.RR on IP-weighted mortality RR (~null after adjustment)",
       extra = list(rr = rr_adj, rr_ci = c(lo_a, hi_a),
                    evalue_ci = ev_adj_ci)),
  list(module = MODULE, side = "R", statistic = "evalue_smd_point",
       estimate = ev_smd_point, se = NULL, n = nrow(cc),
       published = ev_md_point,
       citation = "EValue::evalues.RR(exp(0.91 d)) vs evalues.MD on Ch12 weight effect",
       extra = list(effect_kg = effect, sd_outcome = sd_out, d = d_smd,
                    rr_approx = rr_approx,
                    evalue_md_point = ev_md_point))
)
payload <- list(
  module = MODULE, side = "R", rows = rows,
  extra = list(engine = "EValue::evalues.RR/MD + base-R stabilized IPW",
               n_full = n_full, n_complete = nrow(cc),
               vd_smd_coef = VD_SMD_COEF,
               closed_form = "E = RR + sqrt(RR*(RR-1))",
               smd_rr_eq_md = isTRUE(all.equal(ev_smd_point, ev_md_point)))
)
writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", digits = 10),
           file.path(RESULTS_DIR, paste0(MODULE, "_R.json")))
cat(sprintf(paste0("[%s] crude RR=%.4f CI(%.3f,%.3f) E=%.4f (closed %.4f) ",
                   "E_ci=%.4f || ipw RR=%.4f CI(%.3f,%.3f) E=%.4f || ",
                   "SMD d=%.4f RR_approx=%.4f E=%.4f (MD %.4f)\n"),
            MODULE, rr_crude, lo_c, hi_c, ev_crude_point, closed_crude,
            ev_crude_ci, rr_adj, lo_a, hi_a, ev_adj_point,
            d_smd, rr_approx, ev_smd_point, ev_md_point))
