# R ivmodel 1.9.1 (GPL-2; used as a black box, outputs only) on many_iv.csv.
suppressMessages(library(ivmodel))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
df <- read.csv(file.path(here, "many_iv.csv"))
Z <- as.matrix(df[, grep("^z", names(df))])
X <- as.matrix(df[, c("w1", "w2")])
m <- ivmodel(Y = df$y, D = df$d, Z = Z, X = X)
out <- list()
for (mw in c(FALSE, TRUE)) {
  l <- LIML(m, manyweakSE = mw)
  f <- Fuller(m, b = 1, manyweakSE = mw)
  tag <- if (mw) "manyweak" else "classical"
  out[[paste0("liml_", tag)]] <- c(l$k, l$point.est, l$std.err)
  out[[paste0("fuller1_", tag)]] <- c(f$k, f$point.est, f$std.err)
}
out$version <- as.character(packageVersion("ivmodel"))
writeLines(jsonlite::toJSON(out, digits = NA, auto_unbox = TRUE),
           file.path(here, "ivmodel_many_iv.json"))
