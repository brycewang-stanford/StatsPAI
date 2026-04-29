#!/usr/bin/env Rscript
# R reference values for DiD-family parity:
#   did_imputation (Borusyak-Jaravel-Spiess)  ↔  R didimputation::did_imputation
#   gardner_did (two-stage)                   ↔  R did2s::did2s
#   wooldridge_did (etwfe)                    ↔  R etwfe::etwfe
#
# All three consume the shared cs_data.csv staggered DGP so we can
# read off all references in one R invocation.
suppressMessages({
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/cs_data.csv")
# Convert never-treated 0 → 9999 for packages that interpret "never" as Inf
df_inf <- df
df_inf$first_treat[df_inf$first_treat == 0] <- 9999

# ── BJS did_imputation ────────────────────────────────────────────────
bjs_simple_att <- NA
bjs_simple_se  <- NA
bjs_meta <- list(available = FALSE, error = "")
tryCatch({
  suppressMessages(library(didimputation))
  m_bjs <- did_imputation(
    data = df, yname = "y", gname = "first_treat",
    tname = "year", idname = "id"
  )
  # Single overall ATT row
  bjs_simple_att <- m_bjs$estimate[1]
  bjs_simple_se  <- m_bjs$std.error[1]
  bjs_meta <- list(available = TRUE,
                   version = as.character(packageVersion("didimputation")))
}, error = function(e) {
  bjs_meta <<- list(available = FALSE, error = conditionMessage(e))
})

# ── Gardner did2s ─────────────────────────────────────────────────────
gardner_att <- NA
gardner_se  <- NA
gardner_meta <- list(available = FALSE, error = "")
tryCatch({
  suppressMessages(library(did2s))
  # Two-stage estimator with a single ``post`` indicator → static ATT.
  m_d2s <- did2s(
    data = df_inf, yname = "y",
    first_stage = ~ 0 | id + year,
    second_stage = ~ i(post, ref = 0),
    treatment = "post", cluster_var = "id"
  )
  cf <- coef(m_d2s)
  ses <- sqrt(diag(vcov(m_d2s)))
  # The post=1 indicator is the static ATT.
  idx <- grep("post::1", names(cf), value = FALSE)
  if (length(idx) == 0) idx <- 1
  gardner_att <- cf[idx[1]]
  gardner_se  <- ses[idx[1]]
  gardner_meta <- list(available = TRUE,
                       version = as.character(packageVersion("did2s")))
}, error = function(e) {
  gardner_meta <<- list(available = FALSE, error = conditionMessage(e))
})

# ── Wooldridge etwfe ──────────────────────────────────────────────────
etwfe_att <- NA
etwfe_se  <- NA
etwfe_meta <- list(available = FALSE, error = "")
tryCatch({
  suppressMessages(library(etwfe))
  # etwfe has its own gname/cohort grammar; use 9999 for never-treated
  m_etwfe <- etwfe(
    fml = y ~ 0,                      # no controls
    tvar = year, gvar = first_treat,
    data = df_inf,
    cgroup = "never"                  # never-treated control
  )
  # emfx pulls the marginal effect (single overall ATT)
  emfx_res <- emfx(m_etwfe, type = "simple")
  etwfe_att <- emfx_res$estimate[1]
  etwfe_se  <- emfx_res$std.error[1]
  etwfe_meta <- list(available = TRUE,
                     version = as.character(packageVersion("etwfe")))
}, error = function(e) {
  etwfe_meta <<- list(available = FALSE, error = conditionMessage(e))
})

# ── Save ──────────────────────────────────────────────────────────────
out <- list(
  bjs = list(meta = bjs_meta, estimate = bjs_simple_att, se = bjs_simple_se),
  gardner = list(meta = gardner_meta, estimate = gardner_att, se = gardner_se),
  etwfe = list(meta = etwfe_meta, estimate = etwfe_att, se = etwfe_se)
)
write_json(out, "tests/reference_parity/_fixtures/did_variants_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "null")
cat(sprintf("BJS:     %.4f / %.4f  (%s)\n",
            ifelse(is.na(bjs_simple_att), -999, bjs_simple_att),
            ifelse(is.na(bjs_simple_se), -999, bjs_simple_se),
            ifelse(bjs_meta$available, "OK", paste("err:", substring(bjs_meta$error, 1, 60)))))
cat(sprintf("Gardner: %.4f / %.4f  (%s)\n",
            ifelse(is.na(gardner_att), -999, gardner_att),
            ifelse(is.na(gardner_se), -999, gardner_se),
            ifelse(gardner_meta$available, "OK", paste("err:", substring(gardner_meta$error, 1, 60)))))
cat(sprintf("ETWFE:   %.4f / %.4f  (%s)\n",
            ifelse(is.na(etwfe_att), -999, etwfe_att),
            ifelse(is.na(etwfe_se), -999, etwfe_se),
            ifelse(etwfe_meta$available, "OK", paste("err:", substring(etwfe_meta$error, 1, 60)))))
