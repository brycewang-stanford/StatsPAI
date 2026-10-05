# Answer key for tests/external_parity/test_gow_ding_accounting.py
#
# Reruns the deterministic parts of Gow and Ding, "Empirical Research in
# Accounting: Tools and Methods" (2024) that need no WRDS subscription: the
# chapters on statistical inference, regression, panel data, instrumental
# variables, extreme values, generalized linear models, and the pieces of
# the earnings-management, accruals, natural-experiment and prediction
# chapters that run on simulated data or on the data sets shipped in the
# book's companion package `farr` (MIT licence; install.packages("farr")).
#
#   export STATSPAI_GOW_DING_DIR=/some/empty/folder
#   Rscript tests/external_parity/gow_ding_accounting_reference.R
#
# writes the data sets used as CSV into that folder and the reference
# values into tests/external_parity/data/gow_ding_accounting_R.json.
out_dir <- Sys.getenv("STATSPAI_GOW_DING_DIR")
if (out_dir == "") stop("set STATSPAI_GOW_DING_DIR to an output folder")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
json_path <- file.path(here, "data", "gow_ding_accounting_R.json")
dpath <- function(name) file.path(out_dir, name)
suppressMessages({
  library(dplyr); library(tidyr); library(purrr); library(farr); library(fixest)
  library(sandwich); library(lmtest); library(plm); library(robustbase)
  library(jsonlite); library(MASS); library(car)
})
out <- list()
ct <- function(fm, vc = NULL) {
  b <- coef(fm)
  V <- if (is.null(vc)) vcov(fm) else vc
  list(names = names(b), coef = unname(b), se = unname(sqrt(diag(V))))
}
fx <- function(fm) list(names = names(coef(fm)), coef = unname(coef(fm)),
                        se = unname(se(fm)), p = unname(pvalue(fm)),
                        n = nobs(fm))

# ---------------------------------------------------------------- stat-inf
set.seed(2021)
test <- get_got_data(N = 500, T = 10, Xvol = 0.75, Evol = 0.75,
                     rho_X = 0.5, rho_E = 0.5)
write.csv(test, dpath("got.csv"), row.names = FALSE)
fm <- lm(y ~ x, data = test)
out$got_ols <- ct(fm)
out$got_hc1 <- ct(fm, vcovHC(fm, type = "HC1"))
pl <- plm(y ~ x, test, index = c("firm", "year"), model = "pooling")
out$got_nw <- ct(pl, vcovNW(pl))
out$got_nw_l3 <- ct(pl, vcovNW(pl, maxlag = 3))
pm <- pmg(y ~ x, test, index = "year")
spm <- summary(pm)
out$got_fm <- list(names = names(coef(pm)), coef = unname(coef(pm)),
                   se = unname(sqrt(diag(vcov(pm)))),
                   r2 = spm$rsqr, n = nrow(test))
coefs_df <- test |> group_by(year) |> nest() |>
  mutate(co = map(data, \(d) coef(lm(y ~ x, data = d)))) |>
  unnest_wider(co) |> ungroup() |> arrange(year)
out$got_fm_years <- list(year = coefs_df$year, b0 = coefs_df$`(Intercept)`,
                         b1 = coefs_df$x)
f0 <- lm(`(Intercept)` ~ 1, data = coefs_df); f1 <- lm(x ~ 1, data = coefs_df)
nw <- function(f, lag) sqrt(diag(NeweyWest(f, lag = lag, prewhite = FALSE, adjust = TRUE)))
out$got_fm_t <- list(coef = c(coef(f0), coef(f1)), se = c(sqrt(diag(vcov(f0))), sqrt(diag(vcov(f1)))))
out$got_fm_nw1 <- list(se = c(nw(f0, 1), nw(f1, 1)))
out$got_fm_nw3 <- list(se = c(nw(f0, 3), nw(f1, 3)))
out$got_cl_i <- fx(feols(y ~ x, vcov = ~firm, data = test))
out$got_cl_t <- fx(feols(y ~ x, vcov = ~year, data = test))
out$got_cl_2 <- fx(feols(y ~ x, vcov = ~year + firm, data = test))

# -------------------------------------------------------------- reg-basics
camp_scores <- test_scores |> inner_join(camp_attendance, by = "id") |>
  rename(treat = camp) |> mutate(post = grade >= 7)
write.csv(camp_scores, dpath("camp_scores.csv"), row.names = FALSE)
out$dd_67 <- ct(lm(score ~ treat * post, data = camp_scores, subset = grade %in% 6:7L))
out$dd_all <- ct(lm(score ~ treat * post, data = camp_scores))
out$dd_trend <- ct(lm(score ~ treat * post + grade, data = camp_scores))
out$dd_grade <- ct(lm(score ~ factor(grade), data = test_scores))
fm_id <- lm(score ~ treat * post + factor(grade) + factor(id), data = camp_scores)
k <- which(names(coef(fm_id)) == "treatTRUE:postTRUE")
out$dd_id <- list(coef = unname(coef(fm_id)[k]), se = unname(sqrt(diag(vcov(fm_id)))[k]),
                  n_na = sum(is.na(coef(fm_id))), rank = fm_id$rank)
out$dd_fe_grade <- ct(lm(score ~ treat * post + factor(grade), data = camp_scores))
out$dd_feols <- fx(feols(score ~ I(post * treat) | grade + id, data = camp_scores))
out$dd_feols_iid <- fx(feols(score ~ I(post * treat) | grade + id, vcov = "iid", data = camp_scores))
out$dd_feols_cl2 <- fx(feols(score ~ I(post * treat) | grade + id, vcov = ~grade + id, data = camp_scores))

fm <- lm(ta ~ big_n + cfo + size + lev + mtb +
           factor(fyear) * (inv_at + I(d_sale - d_ar) + ppe),
         data = comp, na.action = na.exclude)
keep <- c("big_nTRUE", "cfo", "size", "lev", "mtb")
out$comp_raw_lm <- list(names = keep, coef = unname(coef(fm)[keep]),
                        se = unname(sqrt(diag(vcov(fm)))[keep]), n = nobs(fm),
                        k = length(coef(fm)), r2 = summary(fm)$r.squared)

# ------------------------------------------------------------ extreme-vals
set.seed(42); xx <- c(rnorm(997), NA, 50, -50)
write.csv(data.frame(x = xx), dpath("wins_x.csv"), row.names = FALSE, na = "")
out$wins <- list(w = winsorize(xx), t = truncate(xx),
                 w05 = winsorize(xx, prob = 0.05),
                 w_asym = winsorize(xx, p_low = 0, p_high = 0.99),
                 q2 = unname(quantile(xx, c(0.01, 0.99), type = 2, na.rm = TRUE)),
                 q7 = unname(quantile(xx, c(0.01, 0.99), type = 7, na.rm = TRUE)))
yy <- c(1:10, 10, 10, 3, 3, NA, 7.5)
out$deciles <- list(x = yy, d = form_deciles(yy), nt = ntile(yy, 10), nt4 = ntile(yy, 4))

comp <- comp |> mutate(fyear = as.factor(fyear))   # as the chapter does
comp_win <- comp |> mutate(across(ta:ppe, winsorize))
comp_trunc <- comp |> mutate(across(ta:ppe, truncate))
write.csv(comp_trunc, dpath("comp_trunc.csv"), row.names = FALSE, na = "")
write.csv(comp, dpath("comp_full.csv"), row.names = FALSE, na = "")
write.csv(comp_win, dpath("comp_win.csv"), row.names = FALSE, na = "")
f_raw <- feols(ta ~ big_n + cfo + size + lev + mtb + fyear * (inv_at + I(d_sale - d_ar) + ppe),
               ~ gvkey + fyear, data = comp)
g <- function(f) { r <- fx(f); i <- match(keep, r$names); list(names = keep, coef = r$coef[i], se = r$se[i], n = r$n) }
out$comp_raw <- g(f_raw)
out$comp_win <- g(update(f_raw, data = comp_win))
out$comp_trunc <- g(update(f_raw, data = comp_trunc))
out$comp_win_roa <- g(feols(ta ~ big_n + roa + cfo + size + lev + mtb + fyear * (inv_at + I(d_sale - d_ar) + ppe),
                            ~ gvkey + fyear, data = comp_win))
# The same three fits with the unused year level dropped, so that the base
# year is the first one observed (1996), as in StatsPAI. With 21 year
# clusters the two-way covariance is not positive semi-definite; fixest
# sets its negative eigenvalues to zero, and that adjustment depends on
# which year is the base. In the fits above the base level 1995 is empty
# and fixest drops the 2015 terms instead.
used <- complete.cases(comp[, c("ta", "big_n", "cfo", "size", "lev", "mtb", "inv_at", "d_sale", "d_ar", "ppe")])
same_base <- function(d) {
  d <- d[complete.cases(d[, c("ta", "big_n", "cfo", "size", "lev", "mtb", "inv_at", "d_sale", "d_ar", "ppe")]), ]
  d$fyear <- droplevels(d$fyear)
  m <- suppressWarnings(update(f_raw, data = d))
  V <- vcov(update(f_raw, data = d, vcov = "iid"), vcov = ~ gvkey + fyear, vcov_fix = FALSE)
  c(g(m), list(n_negative = sum(eigen(V, symmetric = TRUE)$values < 0),
               se_unfixed = unname(sqrt(pmax(diag(V)[keep], 0)))))
}
out$comp_raw_base96 <- same_base(comp)
out$comp_win_base96 <- same_base(comp_win)
out$comp_trunc_base96 <- same_base(comp_trunc)
cd <- cooks.distance(fm)
out$cooks <- list(n_extreme = sum(cd > 4 / nobs(fm), na.rm = TRUE),
                  head = unname(cd[1:8]), max = max(cd, na.rm = TRUE),
                  sum = sum(cd, na.rm = TRUE))
comp_cd <- comp |> mutate(cooksd = cd, extreme = cooksd > 4 / nobs(fm), ta = if_else(extreme, NA, ta))
out$comp_cooks <- g(update(f_raw, data = comp_cd))

# robust regression (MM, tuning.psi = 3.4437 as in the chapter), iterated to
# a tight tolerance so the comparison is not limited by lmrob's defaults
tight <- lmrob.control(tuning.psi = 3.4437, rel.tol = 1e-14, refine.tol = 1e-14,
                       solve.tol = 1e-14, scale.tol = 1e-14, max.it = 5000,
                       k.max = 5000, maxit.scale = 5000)
set.seed(2021)
n <- 2000
rb <- tibble(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), v = rnorm(n),
             y = 0.8 * x1 + 0.4 * x2 + 0.2 * x3 + v, z = rnorm(n, 3, 1)) |>
  mutate(id = row_number(), cy = if_else(id < 0.25 * n & x1 < -1.5, y + z, y))
write.csv(rb, dpath("robust_sim.csv"), row.names = FALSE)
rr <- function(f, d) {
  m <- lmrob(f, data = d, method = "MM", control = tight)
  list(names = names(coef(m)), coef = unname(coef(m)), se = unname(sqrt(diag(vcov(m)))),
       scale = m$scale, w_sum = sum(weights(m, type = "robustness")),
       w_zero = sum(weights(m, type = "robustness") == 0))
}
out$lmrob_sim1 <- rr(cy ~ x1, rb)
out$lmrob_sim3 <- rr(cy ~ x1 + x2 + x3, rb)
comp_s <- na.omit(comp[, c("ta", "big_n", "cfo", "size", "lev", "mtb")])
write.csv(comp_s, dpath("comp_simple.csv"), row.names = FALSE)
out$lmrob_comp <- rr(ta ~ big_n + cfo + size + lev + mtb, comp_s)

# cmsw: Poisson + OLS with HC1, ITCV, impacts
yvars <- c("firmpenalty", "emppenalty", "empprisonmos")
cmsw <- cmsw_2018 |>
  mutate(across(c(blckownpct, initabret, pctinddir, mkt2bk, lev), winsorize),
         across(any_of(yvars), \(x) log(1 + x), .names = "ln_{.col}"),
         ff12 = as.factor(ff12), across(where(is.logical), as.integer)) |>
  filter(tousesox == 1)
write.csv(cmsw, dpath("cmsw.csv"), row.names = FALSE, na = "")
x <- "wbflag"
controls <- c("selfdealflag", "blckownpct", "initabret", "lnvioperiod",
              "bribeflag", "mobflag", "deter", "lnempcleveln", "lnuscodecnt",
              "viofraudflag", "misledflag", "audit8flag", "exectermflag",
              "coopflag", "impedeflag", "pctinddir", "recidivist",
              "lnmktcap", "mkt2bk", "lev", "lndistance", "ff12")
for (y in yvars) {
  form <- as.formula(paste(y, "~", paste(c(x, controls), collapse = " + ")))
  pf <- glm(form, family = "poisson", data = cmsw, control = glm.control(maxit = 100))
  c1 <- coeftest(pf, vcov = vcovHC(pf, type = "HC1"))
  out[[paste0("pois_", y)]] <- list(coef = c1["wbflag", 1], se = c1["wbflag", 2], z = c1["wbflag", 3],
                                    se_model = sqrt(diag(vcov(pf)))[["wbflag"]],
                                    se_hc0 = sqrt(diag(vcovHC(pf, type = "HC0")))[["wbflag"]],
                                    n = nobs(pf), k = length(coef(pf)))
  form2 <- as.formula(paste0("ln_", y, " ~ ", paste(c(x, controls), collapse = " + ")))
  of <- lm(form2, data = cmsw)
  c2 <- coeftest(of, vcov = vcovHC(of, type = "HC1"))
  tstat <- c2["wbflag", "t value"]; df <- df.residual(of)
  numer <- tstat^2 / df; r_yx <- sqrt(numer / (1 + numer))
  al <- c(0.01, 0.05, 0.1); cv <- qt(1 - al / 2, df); rh <- cv / sqrt(df + cv^2)
  out[[paste0("ols_", y)]] <- list(coef = c2["wbflag", 1], se = c2["wbflag", 2], t = tstat, df = df,
                                   itcv = (r_yx - rh) / (1 - abs(rh)), r_yx = r_yx, crit = cv,
                                   t_iid = summary(of)$coefficients["wbflag", 3])
}
pcor <- function(x) { cvx <- cov(as.matrix(x)); icvx <- if (det(cvx) < .Machine$double.eps) MASS::ginv(cvx) else solve(cvx)
  p <- -cov2cor(icvx); diag(p) <- 1; p[-1, 1] }
get_impacts <- function(df, y, x, controls) {
  cd <- df[, c(y, x, controls)]
  for (v in names(which(sapply(cd, is.factor)))) {
    for (val in as.vector(unique(cd[[v]]))) cd[[paste0(v, "_", val)]] <- as.integer(cd[[v]] == val)
    cd[[v]] <- NULL }
  cd <- na.omit(cd)
  r_yz <- pcor(cd[, setdiff(names(cd), x)]); r_xz <- pcor(cd[, setdiff(names(cd), y)])
  tibble(var = setdiff(names(cd), c(x, y)), r_yz, r_xz, impact = r_yz * r_xz)
}
imp <- get_impacts(as.data.frame(cmsw), "ln_firmpenalty", "wbflag", controls)
out$impacts <- as.list(imp)
fmz <- lm(as.formula(paste(x, "~", paste(controls, collapse = " + "))), data = cmsw)
cz <- as.data.frame(cmsw); cz$z <- fitted(fmz)
out$impact_combined <- as.list(get_impacts(cz, "ln_firmpenalty", "wbflag", "z"))

# ---------------------------------------------------------------------- iv
set.seed(2019)
n <- 1000
ivd <- as_tibble(mvrnorm(n, mu = c(0, 0), Sigma = matrix(c(1, .2, .2, 1), 2)), .name_repair = ~c("X", "e")) |>
  mutate(y = e, z_1 = X + rnorm(n, sd = 0.3), z_2 = rnorm(n, sd = 0.3), z_3 = rnorm(n, sd = 0.3))
write.csv(ivd, dpath("iv_sim.csv"), row.names = FALSE)
iv <- feols(y ~ 1 | X ~ z_1 + z_2 + z_3, data = ivd)
out$iv <- c(fx(iv), list(sargan = iv$iv_sargan$stat, sargan_p = iv$iv_sargan$p,
                         f1 = fitstat(iv, "ivf1", simplify = TRUE)$stat,
                         f1_p = fitstat(iv, "ivf1", simplify = TRUE)$p,
                         wh = fitstat(iv, "wh", simplify = TRUE)$stat,
                         wh_p = fitstat(iv, "wh", simplify = TRUE)$p))

# ------------------------------------------------------------- glms (mfx)
set.seed(2024)
n <- 1000; xg <- rnorm(n); yb <- runif(n) < pnorm(-2 + 0.3 * xg); yc <- rpois(n, exp(-2 + 0.3 * xg))
gd <- data.frame(x = xg, yb = as.integer(yb), yc = yc)
write.csv(gd, dpath("glm_sim.csv"), row.names = FALSE)
for (t in c("probit", "logit")) {
  m <- glm(yb ~ x, family = binomial(link = t), data = gd)
  d <- if (t == "probit") dnorm else dlogis
  out[[paste0("glm_", t)]] <- list(coef = unname(coef(m)), se = unname(sqrt(diag(vcov(m)))),
                                   mfx = mean(d(predict(m, type = "link"))) * coef(m)[["x"]])
}
m <- glm(yc ~ x, family = poisson, data = gd)
out$glm_poisson <- list(coef = unname(coef(m)), se = unname(sqrt(diag(vcov(m)))),
                        mfx = mean(predict(m, type = "response")) * coef(m)[["x"]],
                        se_hc1 = unname(sqrt(diag(vcovHC(m, type = "HC1")))))

# ------------------------------------------------- earnings-mgt / accruals
out$binom <- list(p10 = binom.test(10, 1000, p = 0.05)$p.value,
                  p90 = binom.test(90, 1000, p = 0.05)$p.value,
                  ci90 = unname(binom.test(90, 1000, p = 0.05)$conf.int))
set.seed(7)
ad <- data.frame(dec = factor(rep(1:10, each = 40))); ad$r <- rnorm(400, mean = as.integer(ad$dec) * 0.01)
write.csv(ad, dpath("decile_sim.csv"), row.names = FALSE)
fmd <- lm(r ~ dec - 1, data = ad)
lh <- linearHypothesis(fmd, "dec1 = dec10")
out$linhyp <- list(F = lh$F[2], p = lh$`Pr(>F)`[2], coef = unname(coef(fmd)))

# ---------------------------------------------- natural-revisited (ANCOVA)
set.seed(11)
n <- 400
nd <- tibble(id = 1:n, treat = rep(c(TRUE, FALSE), n / 2), a = rnorm(n),
             y_pre = a + rnorm(n), y_post = 0.6 * a + 0.5 * treat + rnorm(n))
long <- nd |> pivot_longer(c(y_pre, y_post), names_to = "t", values_to = "y") |> mutate(post = t == "y_post")
write.csv(nd, dpath("ancova_wide.csv"), row.names = FALSE)
write.csv(long, dpath("ancova_long.csv"), row.names = FALSE)
out$nr_did <- ct(lm(y ~ treat * post, data = long))
out$nr_post <- ct(lm(y_post ~ treat, data = nd))
out$nr_change <- ct(lm(I(y_post - y_pre) ~ treat, data = nd))
out$nr_ancova <- ct(lm(y_post ~ y_pre + treat, data = nd))

# ---------------------------------------------------------- auc / ndcg
set.seed(3)
sc <- round(runif(3000), 3); rs <- rbinom(3000, 1, plogis(-4 + 3 * sc))
write.csv(data.frame(score = sc, response = rs), dpath("auc_sim.csv"), row.names = FALSE)
out$auc <- list(auc = unname(farr::auc(sc, rs)), ndcg01 = unname(farr::ndcg(sc, rs, 0.01)),
                ndcg02 = unname(farr::ndcg(sc, rs, 0.02)), ndcg03 = unname(farr::ndcg(sc, rs, 0.03)))

out$versions <- list(R = R.version.string, fixest = as.character(packageVersion("fixest")),
                     plm = as.character(packageVersion("plm")), sandwich = as.character(packageVersion("sandwich")),
                     robustbase = as.character(packageVersion("robustbase")), MASS = as.character(packageVersion("MASS")),
                     car = as.character(packageVersion("car")),
                     farr = as.character(packageVersion("farr")))
write_json(out, json_path, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null")
cat("done\n")
