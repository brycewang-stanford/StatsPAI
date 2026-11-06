#!/usr/bin/env Rscript
# Reference values for test_tte_longitudinal_r_parity.py.
# Run from the repo root after _generate_tte_longitudinal_data.py:
#   Rscript tests/reference_parity/_fixtures/_generate_tte_longitudinal_R.R
# Writes tte_longitudinal_R.json.
#
# Packages (all run as black boxes; ipw, ltmle, sandwich, SuperLearner are
# GPL): ipw::ipwpoint / ipw::ipwtm for the weights, stats::lm +
# sandwich::vcovHC(type = "HC0") for weighted means and their SEs,
# ltmle::ltmle(gcomp = TRUE) with a single linear learner (SuperLearner
# "SL.lm") for the iterated-conditional-expectation g-formula.
suppressMessages({
  library(ipw); library(sandwich); library(ltmle); library(SuperLearner)
  library(jsonlite)
})
fx <- "tests/reference_parity/_fixtures"
out <- list()

hc0 <- function(fit) sqrt(diag(sandwich::vcovHC(fit, type = "HC0")))

# ---- sp.target_trial_emulate -----------------------------------------------
d <- read.csv(file.path(fx, "tte_point.csv"))
el <- d[d$age >= 45 & d$ldl > 110, ]
f_u <- lm(Y ~ A, data = el)
w_ipw <- ipwpoint(exposure = A, family = "binomial", link = "logit",
                  denominator = ~ x + ldl, data = el,
                  control = glm.control(epsilon = 1e-14, maxit = 100))$ipw.weights
w_stab <- ipwpoint(exposure = A, family = "binomial", link = "logit",
                   numerator = ~ 1, denominator = ~ x + ldl, data = el,
                   control = glm.control(epsilon = 1e-14, maxit = 100))$ipw.weights
el$w_ipw <- w_ipw; el$w_stab <- w_stab
f_w <- lm(Y ~ A, data = el, weights = w_ipw)
f_s <- lm(Y ~ A, data = el, weights = w_stab)
out$tte <- list(
  eligible_id = el$id, n_eligible = nrow(el),
  w_ipw = w_ipw, w_stab = w_stab,
  unweighted = list(est = unname(coef(f_u)["A"]), se = unname(hc0(f_u)["A"])),
  ipw = list(est = unname(coef(f_w)["A"]), se = unname(hc0(f_w)["A"])),
  stabilized = list(est = unname(coef(f_s)["A"]), se = unname(hc0(f_s)["A"]))
)

# ---- sp.longitudinal_analyze / _contrast: IPW path -------------------------
d <- read.csv(file.path(fx, "long_ipw.csv"))
last <- d[d$time == max(d$time), ]
last <- last[order(last$id), ]
ps_w <- ipwpoint(exposure = A, family = "binomial", link = "logit",
                 denominator = ~ B, data = last,
                 control = glm.control(epsilon = 1e-14, maxit = 100))
last$w <- ps_w$ipw.weights
ps <- fitted(ps_w$den.mod)
f1 <- lm(Y ~ 1, data = last[last$A == 1, ], weights = w)
f0 <- lm(Y ~ 1, data = last[last$A == 0, ], weights = w)
fc <- lm(Y ~ A, data = last, weights = w)
out$long_ipw <- list(
  ps_range = range(ps),
  always = list(est = unname(coef(f1)[1]), se = unname(hc0(f1)[1])),
  never = list(est = unname(coef(f0)[1]), se = unname(hc0(f0)[1])),
  contrast = list(est = unname(coef(fc)["A"]), se = unname(hc0(fc)["A"]))
)

# ---- g-formula path: ltmle(gcomp = TRUE), linear Q ------------------------
d <- read.csv(file.path(fx, "long_tv.csv"))
d <- d[order(d$id, d$time), ]
wide <- data.frame(B = d$B[d$time == 0])
for (t in 0:2) {
  s <- d[d$time == t, ]
  wide[[paste0("L_", t)]] <- s$L
  wide[[paste0("A_", t)]] <- s$A
}
wide$Y <- d$Y[d$time == 2]
wide <- wide[, c("B", "L_0", "A_0", "L_1", "A_1", "L_2", "A_2", "Y")]
fit_lt <- function(abar) {
  set.seed(1)
  suppressMessages(ltmle(wide, Anodes = c("A_0", "A_1", "A_2"), Lnodes = c("L_1", "L_2"),
        Ynodes = "Y", abar = abar, gcomp = TRUE,
        SL.library = list(Q = "SL.lm", g = "glm"), estimate.time = FALSE))
}
r1 <- fit_lt(c(1, 1, 1))
r0 <- fit_lt(c(0, 0, 0))
rc <- fit_lt(list(treatment = c(1, 1, 1), control = c(0, 0, 0)))
# self-check: the same sequential regression written with lm
ice <- function(a) {
  q <- wide$Y
  for (t in 2:0) {
    hist <- c("B", unlist(lapply(0:t, function(s) c(paste0("L_", s), paste0("A_", s)))))
    df <- wide[, hist]; df$q <- q
    f <- lm(q ~ ., data = df)
    nd <- df; nd[[paste0("A_", t)]] <- a[t + 1]
    q <- predict(f, newdata = nd)
  }
  mean(q)
}
stopifnot(abs(r1$estimates[["gcomp"]] - ice(c(1, 1, 1))) < 1e-10)
stopifnot(abs(r0$estimates[["gcomp"]] - ice(c(0, 0, 0))) < 1e-10)
sc <- summary(rc)$effect.measures$ATE$estimate
out$long_tv <- list(
  always = r1$estimates[["gcomp"]], never = r0$estimates[["gcomp"]],
  contrast = unname(sc), y_range = range(wide$Y)
)

# ---- sp.clone_censor_weight: ipw::ipwtm(type = "cens") ---------------------
d <- read.csv(file.path(fx, "ccw_panel.csv"))
d <- d[order(d$id, d$time), ]
ccw <- list()
for (strat in c("always", "never")) {
  target <- if (strat == "always") 1 else 0
  risk <- do.call(rbind, lapply(split(d, d$id), function(b) {
    dev <- which(b$treat != target)
    if (length(dev) == 0) { b$C <- 0L; return(b) }
    k <- dev[1]; b <- b[seq_len(k), ]; b$C <- c(rep(0L, k - 1), 1L); b
  }))
  risk <- risk[order(risk$id, risk$time), ]
  ws <- ipwtm(exposure = C, family = "binomial", link = "logit",
              numerator = ~ 1, denominator = ~ L + age, id = id,
              timevar = time, type = "cens", data = risk,
              control = glm.control(epsilon = 1e-14, maxit = 100))
  wu <- ipwtm(exposure = C, family = "binomial", link = "logit",
              denominator = ~ L + age, id = id, timevar = time,
              type = "cens", data = risk,
              control = glm.control(epsilon = 1e-14, maxit = 100))
  keep <- risk$C == 0
  ccw[[strat]] <- list(
    n_at_risk = nrow(risk), n_censor_events = sum(risk$C),
    id = risk$id[keep], time = risk$time[keep],
    w_stab = ws$ipw.weights[keep], w_unstab = wu$ipw.weights[keep]
  )
}
out$ccw <- ccw

out$versions <- list(
  R = R.version.string,
  ipw = as.character(packageVersion("ipw")),
  ltmle = as.character(packageVersion("ltmle")),
  sandwich = as.character(packageVersion("sandwich")),
  SuperLearner = as.character(packageVersion("SuperLearner"))
)
write_json(out, file.path(fx, "tte_longitudinal_R.json"), digits = NA,
           auto_unbox = TRUE, pretty = FALSE)
cat("wrote tte_longitudinal_R.json\n")
