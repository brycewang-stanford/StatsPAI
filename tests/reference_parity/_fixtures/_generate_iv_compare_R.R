# Reference for sp.iv_compare on rd_iv_ivw.csv: ivmodel's k-class
# estimators (KClass(k = 1) is 2SLS, LIML, Fuller(b = 1)), point estimate
# and standard error, homoskedastic and with heteroSE = TRUE.
# ivmodel is GPL-2; black box.
#
#   Rscript tests/reference_parity/_fixtures/_generate_iv_compare_R.R
suppressMessages({
  library(ivmodel)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
d <- read.csv(file.path(here, "rd_iv_ivw.csv"))
m <- ivmodel(Y = d$y, D = d$d, Z = as.matrix(d[, c("z1", "z2", "z3")]),
             X = as.matrix(d[, c("x1", "x2")]))
one <- function(fit) list(b = unname(fit$point.est[1]),
                          se = unname(fit$std.err[1]), k = unname(fit$k[1]))
res <- list(
  versions = list(R = R.version.string,
                  ivmodel = as.character(packageVersion("ivmodel"))),
  homoskedastic = list(`2sls` = one(KClass(m, k = 1)), liml = one(LIML(m)),
                       fuller = one(Fuller(m, b = 1))),
  heteroSE = list(`2sls` = one(KClass(m, k = 1, heteroSE = TRUE)),
                  liml = one(LIML(m, heteroSE = TRUE)),
                  fuller = one(Fuller(m, b = 1, heteroSE = TRUE)))
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "iv_compare_R.json"))
