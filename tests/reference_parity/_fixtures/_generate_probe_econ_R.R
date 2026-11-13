# Reference numbers for tests/reference_parity/test_probe_econ_parity.py
# (2026-11 blind applied-economics probe). R fixest 0.14.0 and did2s 1.2.1
# (run as black boxes) on the CSVs bundled with StatsPAI; one
# "tag,kind,name,value" row per number.
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_probe_econ_R.R
suppressMessages({library(fixest); library(did2s)})
out <- file("tests/reference_parity/_fixtures/probe_econ_R.csv", "w")
writeLines("tag,kind,name,value", out)
put <- function(tag, kind, name, value) {
  writeLines(sprintf("%s,%s,%s,%.17e", tag, kind, name, value), out)
}
put("versions", "s", "fixest", as.numeric(gsub("\\.", "", as.character(packageVersion("fixest")))))

castle <- read.csv("src/statspai/datasets/data/castle_2013.csv")
card <- read.csv("src/statspai/datasets/data/card_1995.csv")
card$g <- floor(card$exper / 3)

# ---- Poisson with absorbed effects: log-likelihood and pseudo R2 ---------
f <- homicide ~ post + unemployrt + poverty + l_police | sid + year
m <- fepois(f, castle, cluster = ~sid)
put("fepois", "s", "ll", as.numeric(logLik(m)))
put("fepois", "s", "pr2", unname(r2(m, "pr2")))
put("fepois", "s", "deviance", deviance(m))
mw <- fepois(f, castle, cluster = ~sid, weights = ~popwt)
put("fepois_w", "s", "ll", as.numeric(logLik(mw)))

# ---- 2SLS with absorbed effects: first-stage F p-value, Wu-Hausman, RMSE --
for (v in c("iid", "hetero", "cluster")) {
  vc <- if (v == "cluster") ~g else v
  m <- feols(lwage ~ black + south | g + smsa | educ ~ nearc4 + nearc2, card, vcov = vc)
  w1 <- fitstat(m, "ivwald1")[[1]]
  wh <- fitstat(m, "wh")[[1]]
  put(paste0("feiv_", v), "s", "ivwald1", w1$stat)
  put(paste0("feiv_", v), "s", "ivwald1_p", w1$p)
  put(paste0("feiv_", v), "s", "wh", wh$stat)
  put(paste0("feiv_", v), "s", "wh_p", wh$p)
  put(paste0("feiv_", v), "s", "rmse", unname(fitstat(m, "rmse")[[1]]))
}

# ---- Gardner two-stage event study ----------------------------------------
castle$g0 <- ifelse(is.na(castle$effyear), 0, castle$effyear)
castle$treat <- as.numeric(castle$g0 > 0 & castle$year >= castle$g0)
castle$rel <- ifelse(castle$g0 == 0, Inf, castle$year - castle$g0)
m <- did2s(castle, yname = "l_homicide", first_stage = ~ 0 | sid + year,
           second_stage = ~ i(rel, ref = c(Inf)), treatment = "treat",
           cluster_var = "sid", verbose = FALSE)
ct <- coeftable(m)
for (k in -4:5) {
  nm <- paste0("rel::", k)
  put("did2s_es", "b", as.character(k), ct[nm, 1])
  put("did2s_es", "se", as.character(k), ct[nm, 2])
  put("did2s_es", "p", as.character(k), ct[nm, 4])
}

# ---- event study with binned end points: the clustered pre-trend F -------
castle$relb <- ifelse(is.na(castle$effyear), -1000, pmax(pmin(castle$year - castle$effyear, 4), -4))
m <- feols(l_homicide ~ i(relb, ref = c(-1, -1000)) | sid + year, castle, cluster = ~sid)
w <- wald(m, "relb::-", print = FALSE)
put("es_pretrend", "s", "F", w$stat)
put("es_pretrend", "s", "p", w$p)
put("es_pretrend", "s", "df1", w$df1)
put("es_pretrend", "s", "df2", w$df2)
close(out)
