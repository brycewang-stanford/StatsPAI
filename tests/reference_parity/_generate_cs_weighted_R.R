suppressMessages(library(did))
set.seed(20260810)
# Deterministic panel written to CSV so Python reads the *same bytes*.
n <- 300; Tt <- 7
rows <- list(); k <- 1
for (i in 1:n) {
  g <- sample(c(3,4,5,0), 1)
  big <- runif(1) < 0.25
  w <- if (big) 100 else 5
  x1 <- rnorm(1)
  fe <- rnorm(1)
  for (t in 1:Tt) {
    te <- if (g == 0 || t < g) 0 else (if (big) 4 else -1)
    rows[[k]] <- data.frame(i=i, t=t, g=g, w=w, x1=x1,
                            y = fe + 0.2*t + 0.3*x1 + te + rnorm(1, 0, 0.3))
    k <- k + 1
  }
}
df <- do.call(rbind, rows)
write.csv(df, "_fixtures/cs_weighted_panel.csv", row.names = FALSE)

out <- list()
for (est in c("dr", "reg", "ipw")) {
  for (cg in c("nevertreated", "notyettreated")) {
    for (wt in c(FALSE, TRUE)) {
      a <- att_gt(yname="y", tname="t", idname="i", gname="g",
                  xformla = ~ x1, data=df,
                  weightsname = if (wt) "w" else NULL,
                  est_method = est, control_group = cg,
                  bstrap = FALSE, cband = FALSE, base_period = "universal")
      s <- aggte(a, type="simple", bstrap=FALSE, cband=FALSE)
      d <- aggte(a, type="dynamic", bstrap=FALSE, cband=FALSE)
      gg<- aggte(a, type="group",  bstrap=FALSE, cband=FALSE)
      out[[length(out)+1]] <- data.frame(
        est=est, control_group=cg, weighted=wt,
        simple_att=s$overall.att, simple_se=s$overall.se,
        dynamic_att=d$overall.att, dynamic_se=d$overall.se,
        group_att=gg$overall.att, group_se=gg$overall.se)
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, "_fixtures/cs_weighted_reference.csv", row.names=FALSE)
print(res, digits=10)
cat("\ndid version:", as.character(packageVersion("did")), "\n")
