# Original-data parity for Hernán & Robins, *Causal Inference: What If*,
# Chapter 17 (IP-weighted survival analysis).
#
#   Source data : sp.datasets.nhefs() full sample (n=1629), dumped by the
#                 Python side to data/10_nhefs_ch17_survival.csv (with a
#                 book survtime column) so both languages read identical
#                 bytes.
#   Published   : unweighted qsmk hazard ratio ~ 1.39; IP-weighted
#                 120-month survival ~ 80.5% (no quit) vs ~ 80.7% (quit),
#                 a near-null +0.2% difference (§17.4 / Program 17.4).
#
# This script is the independent R GOLD reference.  It re-derives the
# stabilized IP weights from the book's logistic propensity model (it
# does NOT trust the sw column the Python side wrote), then:
#   (1) survival::coxph(Surv(survtime, death) ~ qsmk)              -> unweighted HR
#   (2) survival::coxph(..., weights = sw, robust = TRUE)          -> IP-weighted HR
#   (3) base-R IP-weighted person-month pooled-logistic hazard with
#       qsmk*time interactions -> 120-month survival curves (Prog 17.4)
#   (4) EValue::evalues.HR()  -> E-value for the IP-weighted HR (sensitivity)
# Writes results/10_nhefs_ch17_survival_R.json.

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
  library(survival)
  library(EValue)
  library(jsonlite)
})

MODULE <- "10_nhefs_ch17_survival"
df <- read.csv(file.path(DATA_DIR, paste0(MODULE, ".csv")))
n <- nrow(df)
n_death <- sum(df$death)

# --- Independent stabilized IP weights (book Program 12.3 / 17 model) ---
conf <- paste0(
  "factor(sex)+factor(race)+factor(education)+factor(exercise)+",
  "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+",
  "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)
den <- glm(as.formula(paste("qsmk ~", conf)), data = df, family = binomial())
num <- glm(qsmk ~ 1, data = df, family = binomial())
pd  <- predict(den, type = "response")
pn  <- predict(num, type = "response")
sw  <- ifelse(df$qsmk == 1, pn / pd, (1 - pn) / (1 - pd))

# --- (1) Unweighted Cox hazard ratio (contrast) ---
cox0 <- coxph(Surv(survtime, death) ~ qsmk, data = df)
hr_unw <- unname(exp(coef(cox0)["qsmk"]))

# --- (2) IP-weighted Cox hazard ratio (robust SE) ---
cox1 <- coxph(Surv(survtime, death) ~ qsmk, data = df,
              weights = sw, robust = TRUE)
b_w   <- unname(coef(cox1)["qsmk"])
se_w  <- unname(sqrt(diag(vcov(cox1)))["qsmk"])
hr_ipw <- exp(b_w)
hr_lo  <- exp(b_w - qnorm(0.975) * se_w)
hr_hi  <- exp(b_w + qnorm(0.975) * se_w)

# --- (3) IP-weighted pooled-logistic survival curves (Program 17.4) ---
# Person-month long format: rows for months 0..survtime-1, event in the
# final month iff death.
long_list <- lapply(seq_len(n), function(i) {
  T <- df$survtime[i]
  ev <- rep(0, T); if (df$death[i] == 1) ev[T] <- 1
  data.frame(id = i, qsmk = df$qsmk[i], sw = sw[i],
             time = 0:(T - 1), event = ev)
})
long <- do.call(rbind, long_list)
long$timesq <- long$time^2

# Full Program-17.4 hazard model with qsmk x time interactions.
plr <- suppressWarnings(
  glm(event ~ qsmk + I(qsmk * time) + I(qsmk * timesq) + time + timesq,
      data = long, weights = sw, family = binomial()))
tt <- 0:119
h0 <- predict(plr, newdata = data.frame(qsmk = 0, time = tt, timesq = tt^2),
              type = "response")
h1 <- predict(plr, newdata = data.frame(qsmk = 1, time = tt, timesq = tt^2),
              type = "response")
s0 <- tail(cumprod(1 - h0), 1)   # never quit
s1 <- tail(cumprod(1 - h1), 1)   # quit
surv_diff <- s1 - s0             # quit - no quit (book +0.002)

# Simple IP-weighted pooled-logistic HR (single qsmk log-OR), to match
# the Python "hr_ipweighted" pooled-logistic anchor.
plr_s <- suppressWarnings(
  glm(event ~ qsmk + time + timesq, data = long,
      weights = sw, family = binomial()))
hr_ipw_plr <- unname(exp(coef(plr_s)["qsmk"]))

# --- (4) E-value for the IP-weighted Cox HR (sensitivity) ---
ev <- tryCatch(
  evalues.HR(est = hr_ipw, lo = hr_lo, hi = hr_hi, rare = FALSE),
  error = function(e) NULL)
evalue_point <- if (!is.null(ev)) unname(ev["E-values", "point"]) else NULL

rows <- list(
  list(module = MODULE, side = "R", statistic = "hr_unweighted",
       estimate = hr_unw, se = NULL, n = n, published = 1.39,
       citation = "Hernán-Robins, What If §17 (unadjusted qsmk hazard ratio)",
       extra = list(n_death = n_death, scale = "hazard ratio")),
  list(module = MODULE, side = "R", statistic = "hr_ipweighted",
       estimate = hr_ipw, se = se_w, n = n, published = 1.00,
       citation = "Hernán-Robins, What If §17.4 (IP-weighted Cox hazard ratio)",
       extra = list(log_hr = b_w, ci = c(hr_lo, hr_hi),
                    se_is_on = "log-HR scale", n_death = n_death,
                    hr_pooled_logistic = hr_ipw_plr,
                    published_hint = 1.05,
                    evalue_point = evalue_point)),
  list(module = MODULE, side = "R", statistic = "surv120_noquit",
       estimate = s0, se = NULL, n = n, published = 0.805,
       citation = "Hernán-Robins, What If Program 17.4 (IP-weighted S(120), A=0)",
       extra = list(scale = "survival probability")),
  list(module = MODULE, side = "R", statistic = "surv120_quit",
       estimate = s1, se = NULL, n = n, published = 0.807,
       citation = "Hernán-Robins, What If Program 17.4 (IP-weighted S(120), A=1)",
       extra = list(scale = "survival probability")),
  list(module = MODULE, side = "R", statistic = "surv120_diff",
       estimate = surv_diff, se = NULL, n = n, published = 0.002,
       citation = "Hernán-Robins, What If Program 17.4 (IP-weighted 120-month survival difference)",
       extra = list(scale = "survival difference", s_noquit = s0, s_quit = s1))
)
payload <- list(
  module = MODULE, side = "R", rows = rows,
  extra = list(engine = "survival::coxph + base-R pooled-logistic + EValue",
               survtime_rule = "ifelse(death==0,120,(yrdth-83)*12+modth)",
               sw_mean = mean(sw), sw_min = min(sw), sw_max = max(sw),
               hr_ipw_ci = c(hr_lo, hr_hi),
               hr_pooled_logistic = hr_ipw_plr,
               published_hr_hint = 1.05,
               published_hr_ci_hint = c(0.78, 1.43))
)
writeLines(toJSON(payload, auto_unbox = TRUE, null = "null", digits = 10),
           file.path(RESULTS_DIR, paste0(MODULE, "_R.json")))
cat(sprintf(paste0("[%s] n=%d deaths=%d  HR_unw=%.3f  HR_ipw(Cox)=%.3f ",
                   "95%% CI (%.2f,%.2f)  S(120) noquit=%.4f quit=%.4f ",
                   "diff=%+.4f  (book 0.805/0.807/+0.002)\n"),
            MODULE, n, n_death, hr_unw, hr_ipw, hr_lo, hr_hi,
            s0, s1, surv_diff))
