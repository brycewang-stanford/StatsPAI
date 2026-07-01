# StatsPAI panel GLM (feglm / fepois) parity — Module 67 (R side).
#
# Reads the two CSVs dumped by 67_panel_glm.py and runs the canonical
# fixest references for the absorbed-FE high-dimensional GLM family:
#   * feglm(family="logit")  for the Bernoulli panel
#   * fepois                  for the Poisson panel
# Both sides absorb the same single entity fixed effect (``id``) and emit
# every coefficient with its IID standard error so compare.py grades
# the point estimates and the SEs jointly.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages(library(fixest))

MODULE <- "67_panel_glm"

df_logit <- read_csv_strict(paste0(MODULE, "_logit"))
df_pois  <- read_csv_strict(paste0(MODULE, "_poisson"))

n_logit <- nrow(df_logit)
n_pois  <- nrow(df_pois)

m_logit <- fixest::feglm(y ~ x1 + x2 | id, data = df_logit, family = "logit")
m_pois  <- fixest::fepois(y ~ x1 + x2 | id, data = df_pois)

l_co <- coef(m_logit)
l_se <- se(m_logit)
p_co <- coef(m_pois)
p_se <- se(m_pois)

rows <- list(
  # --- feglm (logit) ---
  parity_row(MODULE, "feglm_logit_x1",
             estimate = unname(l_co["x1"]), se = unname(l_se["x1"]),
             n = n_logit),
  parity_row(MODULE, "feglm_logit_x2",
             estimate = unname(l_co["x2"]), se = unname(l_se["x2"]),
             n = n_logit),
  # --- fepois ---
  parity_row(MODULE, "fepois_x1",
             estimate = unname(p_co["x1"]), se = unname(p_se["x1"]),
             n = n_pois),
  parity_row(MODULE, "fepois_x2",
             estimate = unname(p_co["x2"]), se = unname(p_se["x2"]),
             n = n_pois)
)

write_results(MODULE, rows, extra = list(
  feglm_logit = "fixest::feglm(family='logit')",
  fepois = "fixest::fepois",
  fixed_effects = "absorbed id FE (single factor)"
))