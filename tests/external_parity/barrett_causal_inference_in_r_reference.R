# ---------------------------------------------------------------------------
# Answer key for tests/external_parity/test_barrett_causal_inference_in_r.py
#
# Barrett, D'Agostino McGowan and Gerke, "Causal Inference in R"
# (https://www.r-causal.org; source https://github.com/r-causal/causal-inference-in-R).
# This script reruns the analyses of chapters 8 to 16 on the book's running
# example, touringplans::seven_dwarfs_train_2018 at 9 am, with the packages
# the book uses, and stores the numbers. The data are not redistributed
# here; the script exports them from the R package.
#
# Requires: touringplans (GitHub LucyMcGowan/touringplans), propensity
#           (0.1.0), halfmoon (0.2.0), tipr (1.0.2), MatchIt, WeightIt,
#           optweight, lmw, sandwich, survey, marginaleffects, dagitty,
#           dplyr, tidyr, broom, jsonlite.
# Run:      STATSPAI_BARRETT_DIR=/some/folder \
#               Rscript tests/external_parity/barrett_causal_inference_in_r_reference.R
#
# It writes data/barrett_causal_inference_in_r_R.json next to itself and
# seven_dwarfs_9.csv and wait_times.csv into $STATSPAI_BARRETT_DIR.
# ---------------------------------------------------------------------------
suppressMessages({
  library(propensity); library(halfmoon); library(tipr); library(touringplans)
  library(MatchIt); library(WeightIt); library(sandwich); library(survey)
  library(marginaleffects); library(jsonlite); library(dplyr); library(broom)
  library(optweight); library(lmw); library(dagitty)
})
options(tipr.verbose = FALSE)
args <- commandArgs(trailingOnly = FALSE)
self <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
D <- normalizePath(Sys.getenv("STATSPAI_BARRETT_DIR"), mustWork = TRUE)
out <- list()
d <- seven_dwarfs_train_2018 |> filter(wait_hour == 9) |> as.data.frame()
d$park_close_num <- as.numeric(d$park_close)
d$season_regular <- as.numeric(d$park_ticket_season == "regular")
d$season_value <- as.numeric(d$park_ticket_season == "value")
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
ps_mod <- glm(park_extra_magic_morning ~ park_ticket_season + park_close + park_temperature_high,
              data = d, family = binomial(), control = ctl)
ps <- unname(predict(ps_mod, type = "response"))
x <- d$park_extra_magic_morning
y <- d$wait_minutes_posted_avg
out$ps_coef <- as.list(coef(ps_mod)); out$ps <- ps

## ch8/10 weights
W <- list(ate = wt_ate(ps, x), att = wt_att(ps, x), atu = wt_atu(ps, x),
          atm = wt_atm(ps, x), ato = wt_ato(ps, x),
          ate_stab = wt_ate(ps, x, stabilize = TRUE))
out$weights <- lapply(W, as.numeric)
out$ess <- lapply(W, function(w) ess(as.numeric(w)))
out$ess_by_group_ate <- as.list(tapply(as.numeric(W$ate), x, ess))

## ch8 trimming / truncation
tr <- ps_trim(ps, method = "adaptive")
out$trim_adaptive <- list(trimmed = which(is_unit_trimmed(tr)), meta = unclass(ps_trim_meta(tr))[c("lower","upper","cutoff","lambda")])
tr_refit <- ps_refit(tr, ps_mod)
out$trim_refit_ps <- as.numeric(tr_refit)
tc <- ps_trunc(ps, method = "pctl", lower = 0.01, upper = 0.99)
out$trunc_pctl <- list(ps = as.numeric(tc), meta = unclass(ps_trunc_meta(tc)))
tc2 <- ps_trunc(ps, method = "pctl", lower = 0.01, upper = 1)
out$trunc_pctl_lower_only <- as.numeric(tc2)
out$ess_trunc <- ess(as.numeric(wt_ate(tc, x)))

## ch8/9 AUC
dd <- d; dd$.fitted <- ps; dd$w_ate <- as.numeric(W$ate); dd$w_att <- as.numeric(W$att)
dd$emm <- factor(dd$park_extra_magic_morning)
auc_tbl <- check_model_auc(dd, .exposure = emm, .fitted = .fitted, .weights = w_ate)
out$auc <- as.list(setNames(auc_tbl$auc, auc_tbl$method))

## ch9 balance
bal <- dd |> mutate(park_close = as.numeric(park_close)) |>
  check_balance(.vars = c(park_ticket_season, park_close, park_temperature_high),
                .exposure = emm, .weights = w_ate)
out$balance <- as.data.frame(bal)
balj <- dd |> mutate(park_close = as.numeric(park_close)) |>
  check_balance(.vars = c(park_ticket_season, park_close, park_temperature_high),
                .exposure = emm, .weights = w_ate, interactions = TRUE, squares = TRUE)
out$balance_joint <- as.data.frame(balj)
# direct helpers on one variable
out$bal_direct <- list(
  smd_temp = bal_smd(dd$park_temperature_high, dd$emm),
  smd_temp_w = bal_smd(dd$park_temperature_high, dd$emm, .weights = dd$w_ate),
  vr_temp = bal_vr(dd$park_temperature_high, dd$emm),
  vr_temp_w = bal_vr(dd$park_temperature_high, dd$emm, .weights = dd$w_ate),
  ks_temp = bal_ks(dd$park_temperature_high, dd$emm),
  ks_temp_w = bal_ks(dd$park_temperature_high, dd$emm, .weights = dd$w_ate),
  smd_temp_att = bal_smd(dd$park_temperature_high, dd$emm, .weights = dd$w_att))
cov_df <- data.frame(regular = d$season_regular, value = d$season_value, close = d$park_close_num, temp = d$park_temperature_high)
out$energy <- list(obs = bal_energy(cov_df, dd$emm), ate = bal_energy(cov_df, dd$emm, .weights = dd$w_ate))

## ch9 optweight / energy weights
ow <- optweight(park_extra_magic_morning ~ park_ticket_season + park_close_num + park_temperature_high,
                data = d, estimand = "ATE", tols = 0.01, min.w = 0)
out$optweight <- list(w = ow$weights, ess = ess(ow$weights), n_zero = sum(ow$weights == 0))
ew <- weightit(park_extra_magic_morning ~ park_ticket_season + park_close_num + park_temperature_high,
               data = d, method = "energy", estimand = "ATE")
out$energy_w <- list(w = ew$weights, ess = ess(ew$weights))

## ch10 lmw implied regression weights
d$emm <- d$park_extra_magic_morning
iw <- lmw(~ emm + park_ticket_season + park_close_num + park_temperature_high, data = d, treat = "emm")
out$lmw <- list(w = iw$weights, ess = ess(iw$weights), sum = sum(iw$weights))
iw2 <- lmw(~ emm * (park_ticket_season + park_close_num + park_temperature_high), data = d, treat = "emm")
out$lmw_int <- list(w = iw2$weights, min = min(iw2$weights))
ols <- lm(wait_minutes_posted_avg ~ emm + park_ticket_season + park_close_num + park_temperature_high, data = d)
out$lmw_check <- list(ols = unname(coef(ols)["emm"]),
                      wdiff = weighted.mean(y[x == 1], iw$weights[x == 1]) - weighted.mean(y[x == 0], iw$weights[x == 0]))

## ch11 outcome models
set.seed(1)
m <- matchit(park_extra_magic_morning ~ park_ticket_season + park_close + park_temperature_high, data = d)
md <- get_matches(m)
mo <- lm(wait_minutes_posted_avg ~ park_extra_magic_morning, data = md)
vcl <- vcovCL(mo, cluster = ~subclass)
out$match <- list(n = nrow(md), est = unname(coef(mo)[2]), se_ols = sqrt(vcov(mo)[2, 2]), se_cl = sqrt(vcl[2, 2]),
                  matched_rows = as.integer(md$id), subclass = as.integer(md$subclass))
for (est in c("ate", "att", "atu", "atm", "ato")) {
  dd$w <- as.numeric(W[[est]])
  wo <- lm(wait_minutes_posted_avg ~ park_extra_magic_morning, data = dd, weights = w)
  des <- svydesign(ids = ~1, weights = ~w, data = dd)
  sv <- svyglm(wait_minutes_posted_avg ~ park_extra_magic_morning, des)
  ipd <- if (est == "atu") NULL else as.data.frame(ipw(ps_mod, wo, .data = dd, estimand = est))
  out$ipw[[est]] <- list(est = unname(coef(wo)[2]), se_ols = sqrt(vcov(wo)[2, 2]),
    se_hc0 = sqrt(sandwich(wo)[2, 2]), se_hc1 = sqrt(vcovHC(wo, "HC1")[2, 2]), se_hc3 = sqrt(vcovHC(wo, "HC3")[2, 2]),
    se_svy = unname(SE(sv)[2]), ipw = ipd)
}
## binary outcome
dd$over60 <- as.numeric(dd$wait_minutes_posted_avg > 60)
for (est in c("ate", "att")) {
  dd$w <- as.numeric(W[[est]])
  wo <- glm(over60 ~ park_extra_magic_morning, data = dd, weights = w, family = quasibinomial())
  ip <- ipw(ps_mod, wo, .data = dd, estimand = est)
  out$ipw_bin[[est]] <- as.data.frame(ip)
}

## ch12 continuous exposure
e8 <- seven_dwarfs_train_2018 |> filter(wait_hour == 8) |> select(-wait_minutes_actual_avg)
n9 <- seven_dwarfs_train_2018 |> filter(wait_hour == 9) |> select(park_date, wait_minutes_actual_avg)
wt <- e8 |> left_join(n9, by = "park_date") |> tidyr::drop_na(wait_minutes_actual_avg) |> as.data.frame()
wt$park_close_num <- as.numeric(wt$park_close)
den <- lm(wait_minutes_posted_avg ~ park_close + park_extra_magic_morning + park_temperature_high + park_ticket_season, data = wt)
num <- lm(wait_minutes_posted_avg ~ 1, data = wt)
dn <- dnorm(wt$wait_minutes_posted_avg, fitted(den), mean(augment(den)$.sigma, na.rm = TRUE))
nm <- dnorm(wt$wait_minutes_posted_avg, fitted(num), mean(augment(num)$.sigma, na.rm = TRUE))
out$cont <- list(n = nrow(wt), swts = nm / dn, sigma_den_loo = mean(augment(den)$.sigma), sigma_den = sigma(den),
                 wt_ate = as.numeric(wt_ate(fitted(den), wt$wait_minutes_posted_avg, .sigma = augment(den)$.sigma, exposure_type = "continuous", stabilize = TRUE)))
write.csv(wt, file.path(D, "wait_times.csv"), row.names = FALSE)

## ch13 g-comp
gc <- lm(wait_minutes_posted_avg ~ park_extra_magic_morning + park_ticket_season + park_close + park_temperature_high, data = d)
gci <- lm(wait_minutes_posted_avg ~ park_extra_magic_morning * park_ticket_season + park_close + park_temperature_high, data = d)
ac <- function(mod, nd = NULL, ...) { a <- if (is.null(nd)) avg_comparisons(mod, variables = "park_extra_magic_morning", ...) else avg_comparisons(mod, variables = "park_extra_magic_morning", newdata = nd, ...); c(est = a$estimate, se = a$std.error, lo = a$conf.low, hi = a$conf.high) }
out$gcomp <- list(ate = ac(gc), ate_int = ac(gci), att_int = ac(gci, filter(d, park_extra_magic_morning == 1)), atc_int = ac(gci, filter(d, park_extra_magic_morning == 0)))
d$over60 <- as.numeric(d$wait_minutes_posted_avg > 60)
gb <- glm(over60 ~ park_extra_magic_morning + park_ticket_season + park_close + park_temperature_high, data = d, family = binomial(), control = ctl)
out$gcomp_bin <- list(rd = ac(gb), rr = ac(gb, comparison = "lnratioavg", transform = exp), or = ac(gb, comparison = "lnoravg", transform = exp),
  lnrr = ac(gb, comparison = "lnratioavg"), lnor = ac(gb, comparison = "lnoravg"),
  att = ac(gb, filter(d, park_extra_magic_morning == 1)), atc = ac(gb, filter(d, park_extra_magic_morning == 0)))
ap <- avg_predictions(gb, variables = list(park_extra_magic_morning = 0)); out$gcomp_bin$p0 <- c(est = ap$estimate, se = ap$std.error)
fw <- lm(wait_minutes_actual_avg ~ splines::ns(wait_minutes_posted_avg, df = 3) + park_extra_magic_morning + park_ticket_season + park_close + park_temperature_high, data = wt)
a <- avg_comparisons(fw, variables = list(wait_minutes_posted_avg = c(30, 60)))
out$gcomp_cont <- c(est = a$estimate, se = a$std.error)

## ch16 tipr
out$tipr <- list(
  adjust_coef = as.data.frame(adjust_coef(6.58, exposure_confounder_effect = -0.17, confounder_outcome_effect = -2.3)),
  tip_coef_1 = as.data.frame(tip_coef(6.58, confounder_outcome_effect = -7)),
  tip_coef_2 = as.data.frame(tip_coef(6.58, confounder_outcome_effect = -2.3)),
  tip_coef_grid = as.data.frame(tip_coef(-10.2, exposure_confounder_effect = 1:5)),
  adjust_bin = as.data.frame(adjust_coef_with_binary(c(-12.5, -13.4, -11.6), exposed_confounder_prev = 0.26, unexposed_confounder_prev = 0.05, confounder_outcome_effect = -10)),
  adjust_rr = as.data.frame(adjust_rr(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8)),
  adjust_rr_bin = as.data.frame(adjust_rr_with_binary(1.5, exposed_confounder_prev = 0.4, unexposed_confounder_prev = 0.1, confounder_outcome_effect = 1.8)),
  tip_bin = as.data.frame(tip_with_binary(1.2, exposed_confounder_prev = 0.5, unexposed_confounder_prev = 0.1)),
  tip_bin_prev = as.data.frame(tip_with_binary(1.2, unexposed_confounder_prev = 0.1, confounder_outcome_effect = 2.5)),
  tip_cont = as.data.frame(tip_with_continuous(1.2, confounder_outcome_effect = 2.5)),
  tip_rr = as.data.frame(tip_rr(1.2, confounder_outcome_effect = 2.5)),
  adjust_or = as.data.frame(adjust_or(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8)),
  adjust_hr = as.data.frame(adjust_hr(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8)),
  adjust_or_rare = as.data.frame(adjust_or(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8, or_correction = TRUE)),
  r2 = as.data.frame(adjust_coef_with_r2(0.5, 0.1, confounder_exposure_r2 = 0.05, confounder_outcome_r2 = 0.1, df = 100)),
  tip_r2 = as.data.frame(tip_coef_with_r2(0.5, 0.1, confounder_outcome_r2 = 0.1, df = 100)),
  e_value = e_value(1.5), r_value = tryCatch(as.data.frame(r_value(0.5, 0.1, 100)), error = function(e) conditionMessage(e)))

## ch16 dagitty
g <- dagitty("dag { emm -> wait ; close -> wait ; season -> wait ; temp -> wait ; temp -> emm ; emm [exposure] ; wait [outcome] }")
out$dag <- list(adj_all = lapply(adjustmentSets(g, type = "all"), as.character),
  n_equiv = length(equivalentDAGs(g)),
  equiv_edges = lapply(equivalentDAGs(g), function(z) { e <- edges(z); paste(e$v, e$e, e$w) }),
  cpdag_edges = { e <- edges(equivalenceClass(g)); paste(e$v, e$e, e$w) },
  ci = sapply(impliedConditionalIndependencies(g), function(z) paste(z$X, "_||_", z$Y, "|", paste(z$Z, collapse = ","))))
write_json(out, file.path(self, "data", "barrett_causal_inference_in_r_R.json"), digits = NA, auto_unbox = TRUE, pretty = FALSE)
d$park_date <- as.character(d$park_date)
write.csv(d, file.path(D, "seven_dwarfs_9.csv"), row.names = FALSE)
cat("OK\n")
