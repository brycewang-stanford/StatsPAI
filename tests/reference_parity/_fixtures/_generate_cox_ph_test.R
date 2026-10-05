# ---------------------------------------------------------------------------
# Reference for tests/reference_parity/test_cox_ph_test_parity.py
#
# survival::cox.zph (the Grambsch-Therneau score test) for a Cox model with a
# non-proportional covariate, under both tie rules, with and without strata,
# on untied and heavily tied times, for the four time transforms.
#
# Requires: survival (3.8-3), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_cox_ph_test.R
#           (from the repository root; rewrites the CSV as well)
# ---------------------------------------------------------------------------
here <- "tests/reference_parity/_fixtures"
suppressMessages({library(survival); library(jsonlite)})
set.seed(20261006); n <- 300
x1 <- round(rnorm(n),3); x2 <- rbinom(n,1,0.4); x3 <- round(runif(n),3); g <- sample(1:3, n, replace=TRUE)
# non-proportional effect of x2: hazard ratio changes with time
t0 <- rexp(n, rate=exp(0.5*x1 - 0.3*x3)); t0 <- ifelse(x2==1, t0^1.6, t0)
cens <- rexp(n, 0.3); time <- pmax(round(pmin(t0,cens),3), 0.001); event <- as.integer(t0<=cens)
time_tied <- ceiling(time*4)/4      # heavy ties
d <- data.frame(time,time_tied,event,x1,x2,x3,g); write.csv(d,file.path(here, "cox_ph_test.csv"),row.names=FALSE)
tab <- function(z) { t <- as.data.frame(z$table); cbind(term=rownames(t), t) }
out <- list()
for (tv in c("time","time_tied")) for (ti in c("efron","breslow")) for (st in c(FALSE,TRUE)) {
  f <- as.formula(paste0("Surv(",tv,",event) ~ x1 + x2 + x3", if (st) " + strata(g)" else ""))
  m <- coxph(f, data=d, ties=ti)
  key <- paste(tv,ti,if (st) "strata" else "nostrata", sep="|")
  out[[key]] <- list(coef=unname(coef(m)), km=tab(cox.zph(m)), identity=tab(cox.zph(m,transform="identity")), rank=tab(cox.zph(m,transform="rank")), log=tab(cox.zph(m,transform="log")))
}
out$version <- as.character(packageVersion("survival"))
write_json(out,file.path(here, "cox_ph_test_R.json"),digits=NA,auto_unbox=TRUE)
