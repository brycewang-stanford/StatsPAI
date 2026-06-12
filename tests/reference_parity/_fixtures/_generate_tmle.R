#!/usr/bin/env Rscript
# Base-R TMLE reference values for sp.tmle parity.
#
# Implements the canonical ATE TMLE of van der Laan & Rubin (2006)
# with unpenalised logistic nuisance models — BASE R ONLY (stats::glm),
# no tmle/SuperLearner/jsonlite dependencies, so the fixture can be
# regenerated on a vanilla R install.  The algorithm mirrors the
# published recipe (and the spec of src/statspai/tmle/tmle.py):
#
#   1. Q(A, W): glm(y ~ a + w1 + w2, family = binomial)
#      — matches sp.tmle with outcome_library =
#        [LogisticRegression(penalty=None)] on the design [A, W].
#   2. g(W):    glm(a ~ w1 + w2, family = binomial),
#      predictions truncated to [0.025, 0.975] (sp.tmle default
#      propensity_bounds).
#   3. Clever covariate H(A, W) = A/g - (1-A)/(1-g); fluctuation
#      epsilon fit by the offset logistic MLE
#        glm(y ~ -1 + H, offset = qlogis(QA), family = binomial)
#      — the same one-dimensional score equation sp.tmle solves by
#      Newton-Raphson.
#   4. psi = mean(Q*(1,W) - Q*(0,W)); SE from the efficient influence
#      function, sd(EIF)/sqrt(n) with the n-1 denominator.
#
# Output: tmle_R.json (written with base-R sprintf — flat structure,
# 15 significant digits).
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_tmle.R

df <- read.csv("tests/reference_parity/_fixtures/tmle_data.csv")
n <- nrow(df)

# --- Step 1: outcome model Q(A, W) -----------------------------------
q_fit <- glm(y ~ a + w1 + w2, data = df, family = binomial(),
             control = glm.control(epsilon = 1e-12, maxit = 100))
Qbar_A <- predict(q_fit, newdata = df, type = "response")
df1 <- df; df1$a <- 1
df0 <- df; df0$a <- 0
Qbar_1 <- predict(q_fit, newdata = df1, type = "response")
Qbar_0 <- predict(q_fit, newdata = df0, type = "response")

# --- Step 2: propensity model g(W), truncated ------------------------
g_fit <- glm(a ~ w1 + w2, data = df, family = binomial(),
             control = glm.control(epsilon = 1e-12, maxit = 100))
g_raw <- predict(g_fit, newdata = df, type = "response")
g <- pmin(pmax(g_raw, 0.025), 0.975)

# --- Step 3: clever covariate + fluctuation epsilon ------------------
A <- df$a
Y <- df$y
H_A <- A / g - (1 - A) / (1 - g)
eps_fit <- glm(Y ~ -1 + H_A, offset = qlogis(Qbar_A),
               family = binomial(),
               control = glm.control(epsilon = 1e-12, maxit = 100))
epsilon <- unname(coef(eps_fit))

Qstar_A <- plogis(qlogis(Qbar_A) + epsilon * H_A)
Qstar_1 <- plogis(qlogis(Qbar_1) + epsilon / g)
Qstar_0 <- plogis(qlogis(Qbar_0) - epsilon / (1 - g))

# --- Step 4: plug-in estimate + EIF standard error -------------------
psi <- mean(Qstar_1 - Qstar_0)
EIF <- (Qstar_1 - Qstar_0) +
  A * (Y - Qstar_A) / g -
  (1 - A) * (Y - Qstar_A) / (1 - g) - psi
se <- sd(EIF) / sqrt(n)   # stats::sd uses the n-1 denominator

# --- Write JSON with base R (no jsonlite) -----------------------------
num <- function(x) sprintf("%.15g", x)
vec <- function(x) paste0("[", paste(sprintf("%.15g", x), collapse = ", "), "]")

json <- paste0(
  "{\n",
  "  \"meta\": {\n",
  "    \"R_version\": \"", R.version.string, "\",\n",
  "    \"generator\": \"_generate_tmle.R\",\n",
  "    \"data\": \"tmle_data.csv\",\n",
  "    \"n\": ", n, ",\n",
  "    \"propensity_bounds\": [0.025, 0.975],\n",
  "    \"nuisance\": \"stats::glm binomial, epsilon=1e-12\"\n",
  "  },\n",
  "  \"ate\": {\n",
  "    \"psi\": ", num(psi), ",\n",
  "    \"se\": ", num(se), ",\n",
  "    \"epsilon\": ", num(epsilon), ",\n",
  "    \"mean_EIF\": ", num(mean(EIF)), ",\n",
  "    \"q_coef\": ", vec(unname(coef(q_fit))), ",\n",
  "    \"g_coef\": ", vec(unname(coef(g_fit))), ",\n",
  "    \"n_g_truncated\": ", sum(g_raw < 0.025 | g_raw > 0.975), "\n",
  "  }\n",
  "}\n"
)
writeLines(json, "tests/reference_parity/_fixtures/tmle_R.json")
cat(sprintf("TMLE ATE: psi=%.10f se=%.10f epsilon=%.6e\n", psi, se, epsilon))
cat(sprintf("g truncated: %d / %d\n", sum(g_raw < 0.025 | g_raw > 0.975), n))
