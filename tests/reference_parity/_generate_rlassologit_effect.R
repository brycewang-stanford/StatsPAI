# Ground-truth generator for sp.rlassologit_effect / rlassologit_effects —
# faithful port of hdm::rlassologitEffect(s) (logistic high-dimensional
# treatment effect, Belloni-Chernozhukov-Wei double-selection for GLMs).
#
# Outputs (tests/reference_parity/_fixtures/):
#   hdm_rlassologit_effect_data.csv   y (binary), d (binary), x1..xp
#   rlassologit_effect_R.json         hdm alpha/se/t/pval (single + multi)
#
# Re-run only on a contract change:
#   Rscript tests/reference_parity/_generate_rlassologit_effect.R
suppressMessages(library(hdm)); suppressMessages(library(jsonlite))
set.seed(20260626)
n <- 400L; p <- 30L
x <- matrix(rnorm(n * p), n, p)
colnames(x) <- paste0("x", 1:p)
beta <- c(1.0, 0.8, -0.6, rep(0, p - 3))           # sparse controls -> y
d_lin <- x %*% c(0.9, 0.7, rep(0, p - 2))           # confounded treatment
d <- rbinom(n, 1, plogis(d_lin))
y_lin <- 0.7 * d + x %*% beta                        # true effect 0.7
y <- rbinom(n, 1, plogis(y_lin))

here <- file.path("tests", "reference_parity", "_fixtures")
write.csv(data.frame(y = y, d = d, x), file.path(here, "hdm_rlassologit_effect_data.csv"),
          row.names = FALSE)

s_post <- rlassologitEffect(x, y, d, post = TRUE)
s_nopost <- rlassologitEffect(x, y, d, post = FALSE)
# multi-target: treat x[,1], x[,2] as the variables of interest
m <- rlassologitEffects(x, y, index = c(1L, 2L), post = TRUE)
m_coef <- m$coefficients
m_se <- m$se
m_names <- names(m_coef)

ref <- list(
  meta = list(R_version = as.character(getRversion()),
              hdm_version = as.character(packageVersion("hdm")),
              n = n, p = p, note = "hdm::rlassologitEffect(s) ground truth"),
  single_post = list(alpha = as.numeric(s_post$alpha), se = as.numeric(s_post$se),
                     t = as.numeric(s_post$t), pval = as.numeric(s_post$pval)),
  single_nopost = list(alpha = as.numeric(s_nopost$alpha), se = as.numeric(s_nopost$se),
                       t = as.numeric(s_nopost$t), pval = as.numeric(s_nopost$pval)),
  multi = list(
    targets = lapply(seq_along(m_coef), function(i)
      list(name = m_names[i], coef = as.numeric(m_coef[i]), se = as.numeric(m_se[i])))
  )
)
writeLines(toJSON(ref, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           file.path(here, "rlassologit_effect_R.json"))
cat(sprintf("single post  alpha=%.10f se=%.10f t=%.6f\n", s_post$alpha, s_post$se, s_post$t))
cat(sprintf("single nopost alpha=%.10f se=%.10f\n", s_nopost$alpha, s_nopost$se))
cat(sprintf("multi x1 coef=%.8f se=%.8f | x2 coef=%.8f se=%.8f\n",
            m_coef[1], m_se[1], m_coef[2], m_se[2]))
