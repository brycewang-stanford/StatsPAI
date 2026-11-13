# R reference numbers for tests/reference_parity/test_gaps_econ_parity.py.
# One "tag,kind,name,value" row per number. Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_gaps_econ_R.R
# R 4.5.2; fixest 0.14.0 and marginaleffects 0.32.0, both run as black boxes.
suppressMessages({
  library(fixest)
  library(marginaleffects)
})
setFixest_nthreads(1)
rows <- list()
put <- function(tag, kind, name, value) {
  rows[[length(rows) + 1]] <<- data.frame(
    tag = tag, kind = kind, name = name,
    value = sprintf("%.17e", as.numeric(value)), stringsAsFactors = FALSE
  )
}

# ---- fixest fit statistics: rmse, log-likelihood, AIC, BIC -----------------
castle <- read.csv("src/statspai/datasets/data/castle_2013.csv")
fits <- list(
  feols_fe = feols(l_homicide ~ post + unemployrt | sid + year, castle, cluster = ~sid),
  feols_nofe = feols(l_homicide ~ post + unemployrt, castle),
  fepois_fe = fepois(homicide ~ post + unemployrt | sid + year, castle, cluster = ~sid),
  fepois_nofe = fepois(homicide ~ post + unemployrt, castle),
  fepois_w = fepois(homicide ~ post + unemployrt | sid + year, castle, weights = ~popwt),
  feols_w = feols(l_homicide ~ post + unemployrt | sid + year, castle, weights = ~popwt)
)
for (tag in names(fits)) {
  fs <- fitstat(fits[[tag]], ~ rmse + ll + aic + bic)
  for (nm in names(fs)) put(tag, "s", nm, fs[[nm]])
  put(tag, "s", "nparams", fits[[tag]]$nparams)
}

# ---- marginaleffects: predictions on new and counterfactual data ----------
card <- read.csv("src/statspai/datasets/data/card_1995.csv")
card$hi <- as.integer(card$lwage > median(card$lwage))
# Tight IRLS stop: the default (1e-8 on the deviance) leaves the covariance
# 6e-5 from the converged one.
m <- glm(hi ~ educ + exper + black + south + smsa, data = card, family = binomial,
         control = glm.control(epsilon = 1e-14, maxit = 100))
dump_ap <- function(tag, ap, keys = character()) {
  for (i in seq_len(nrow(ap))) {
    nm <- if (length(keys)) paste(sapply(keys, function(k) paste0(k, "=", ap[[k]][i])), collapse = ";") else "all"
    put(tag, "b", nm, ap$estimate[i])
    put(tag, "se", nm, ap$std.error[i])
  }
}
# numderiv = "richardson": the default forward difference leaves the
# standard errors 1e-6 from the analytic delta method.
dump_ap("ap_subset", avg_predictions(m, numderiv = "richardson", newdata = subset(card, black == 1)))
dump_ap("ap_setto1", avg_predictions(m, numderiv = "richardson", newdata = transform(card, black = 1)))
dump_ap("ap_cf", avg_predictions(m, numderiv = "richardson", variables = list(black = c(0, 1))), "black")
dump_ap("ap_cf_name", avg_predictions(m, numderiv = "richardson", variables = "black"), "black")
dump_ap("ap_cf_by", avg_predictions(m, numderiv = "richardson", variables = list(black = c(0, 1)), by = "south"), "south")
dump_ap("ap_cf_grid",
        avg_predictions(m, numderiv = "richardson", variables = list(black = c(0, 1), educ = c(12, 16))),
        c("black", "educ"))
dump_ap("ap_cf_link",
        avg_predictions(m, numderiv = "richardson", variables = list(black = c(0, 1)), type = "link"), "black")
ols <- lm(lwage ~ educ + exper + black * south + smsa, data = card)
dump_ap("ap_ols_cf", avg_predictions(ols, numderiv = "richardson", variables = list(black = c(0, 1))), "black")
put("ap_cf", "s", "n_unit_rows", nrow(predictions(m, variables = list(black = c(0, 1)))))

out <- do.call(rbind, rows)
write.csv(out, "tests/reference_parity/_fixtures/gaps_econ_R.csv",
          row.names = FALSE, quote = FALSE)
