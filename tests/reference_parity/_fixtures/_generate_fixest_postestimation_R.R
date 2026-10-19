# Reference values for test_fixest_postestimation_parity.py.
# R fixest and marginaleffects (both GPL-3) are used as black boxes: only
# their output is read.
suppressPackageStartupMessages({
  library(fixest)
  library(marginaleffects)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
d <- read.csv(file.path(here, "fixest_postestimation_data.csv"))
setFixest_nthreads(1)

fe_list <- function(m) {
  fe <- fixef(m)
  lapply(fe, function(v) list(level = names(v), value = unname(as.numeric(v))))
}
nd <- d[1:6, ]
nd$f1[2] <- 99  # a level the fits never saw: fixest predicts NA

m_ols1 <- feols(y ~ x1 + x2 | f1, d)
m_ols2 <- feols(y ~ x1 + x2 | f1 + f2, d)
m_ols3 <- feols(y ~ x1 + x2 | f1 + f2 + f3, d)
m_int <- feols(y ~ x1 * x2 | f1 + f2, d)
m_iv <- feols(yiv ~ x1 | f1 | d1 ~ z1 + z2, d)
m_pois <- fepois(yp ~ x1 + x2 | f1 + f2, d)
m_logit <- feglm(yb ~ x1 + x2 | f1, d, family = binomial())

num <- function(x) unname(as.numeric(x))
ame <- function(tab) list(term = as.character(tab$term), estimate = num(tab$estimate),
                          std.error = if ("std.error" %in% names(tab)) num(tab$std.error) else NULL)

out <- list(
  versions = list(fixest = as.character(packageVersion("fixest")),
                  marginaleffects = as.character(packageVersion("marginaleffects")),
                  R = R.version.string),
  fixef = list(ols1 = fe_list(m_ols1), ols2 = fe_list(m_ols2), ols3 = fe_list(m_ols3),
               iv = fe_list(m_iv), pois = fe_list(m_pois), logit = fe_list(m_logit)),
  coef = list(ols2 = num(coef(m_ols2)), iv = num(coef(m_iv)),
              pois = num(coef(m_pois)), logit = num(coef(m_logit))),
  predict = list(
    ols2 = num(predict(m_ols2, nd)),
    ols3 = num(predict(m_ols3, nd)),
    iv = num(predict(m_iv, nd)),
    pois = num(predict(m_pois, nd)),
    pois_link = num(predict(m_pois, nd, type = "link")),
    logit = num(predict(m_logit, nd))
  ),
  avg_slopes = list(
    int = ame(avg_slopes(m_int)),
    pois = ame(avg_slopes(m_pois, vcov = FALSE)),
    logit = ame(avg_slopes(m_logit, vcov = FALSE))
  ),
  avg_predictions = list(
    ols2 = num(avg_predictions(m_ols2, vcov = FALSE)$estimate),
    pois = num(avg_predictions(m_pois, vcov = FALSE)$estimate),
    logit = num(avg_predictions(m_logit, vcov = FALSE)$estimate)
  ),
  avg_comparisons = list(
    pois = ame(avg_comparisons(m_pois, vcov = FALSE)),
    int = ame(avg_comparisons(m_int))
  )
)

# Joint Wald tests (marginaleffects::hypotheses(joint=)) on an
# unabsorbed logit, tight convergence (see the marginaleffects test).
g <- glm(yb ~ x1 + x2 + d1, data = d, family = binomial(),
         control = glm.control(epsilon = 1e-14, maxit = 100))
jf <- hypotheses(g, joint = c("x1", "x2"))
jc <- hypotheses(g, joint = c("x1", "x2", "d1"), joint_test = "chisq")
ja <- hypotheses(g, joint = TRUE)
hv <- hypotheses(g, hypothesis = c("x1 = 0", "x2 = 0"))
out$hypotheses <- list(
  joint_f = list(statistic = num(jf$statistic), p.value = num(jf$p.value),
                 df1 = num(jf$df1), df2 = num(jf$df2)),
  joint_chisq = list(statistic = num(jc$statistic), p.value = num(jc$p.value),
                     df = num(jc$df)),
  joint_all = list(statistic = num(ja$statistic), df2 = num(ja$df2)),
  vector = list(estimate = num(hv$estimate), std.error = num(hv$std.error))
)

# Driscoll-Kraay / Newey-West panel HAC with two lags.
m_hac <- feols(y ~ x1 + x2 | id, d, panel.id = ~ id + t)
out$hac <- list(DK = num(se(m_hac, vcov = DK(2))), NW = num(se(m_hac, vcov = NW(2))))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null"),
           file.path(here, "fixest_postestimation_R.json"))
