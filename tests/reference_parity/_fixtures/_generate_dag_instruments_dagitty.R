# dagitty 0.3.4 (GPL-2, used as a black box: outputs only) reference for
# sp.dag(...).instrumental_variables. Reads the 200 random graphs of
# dag_instruments_graphs.json (from _generate_dag_instruments_graphs.py) and
# writes instrumentalVariables(d, "x", "y") for each.
#   cd tests/reference_parity/_fixtures && Rscript _generate_dag_instruments_dagitty.R
suppressMessages(library(dagitty)); library(jsonlite)
g <- fromJSON("dag_instruments_graphs.json", simplifyVector = FALSE)
res <- lapply(g, function(gr) {
  s <- c(unlist(gr$nodes),
         sapply(gr$edges, function(e) paste(e[[1]], "->", e[[2]])),
         sapply(gr$bi, function(e) paste(e[[1]], "<->", e[[2]])))
  d <- dagitty(paste0("dag{ ", paste(unlist(s), collapse = "; "), " }"))
  iv <- instrumentalVariables(d, "x", "y")
  lapply(iv, function(i) list(I = i$I, Z = as.list(i$Z)))
})
writeLines(toJSON(res, auto_unbox = TRUE), "dag_instruments_dagitty.json")
