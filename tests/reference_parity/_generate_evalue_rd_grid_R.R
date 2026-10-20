#!/usr/bin/env Rscript
# Black-box references for sp.evalue_rd's grid rule (clean-room record:
# docs/dev/cleanroom/evalue.md). EValue (GPL-2) is only run, never read.
#
# Regenerate from the repository root:
#   Rscript tests/reference_parity/_generate_evalue_rd_grid_R.R
#
# Coarse grids make the choice of grid point visible (first point whose lower
# confidence limit reaches `true`, not the nearest one); the last case is a
# table so precise that no grid point below the point-estimate bias factor
# qualifies, where EValue returns NA.
suppressPackageStartupMessages({
  library(jsonlite)
  library(EValue)
})
num <- function(x) if (is.na(x)) NA else unname(as.numeric(x))
case <- function(n11, n10, n01, n00, true = 0, alpha = 0.05, grid = 1e-4) {
  e <- evalues.RD(n11, n10, n01, n00, true = true, alpha = alpha, grid = grid)
  list(cells = c(n11, n10, n01, n00), true = true, alpha = alpha, grid = grid,
       est = num(e$est.Evalue), lower = num(e$lower.Evalue))
}
out <- list(
  `_meta` = list(EValue = as.character(packageVersion("EValue")),
                 R = paste(R.version$major, R.version$minor, sep = ".")),
  cases = list(
    case(200, 150, 100, 250, grid = 0.1),
    case(200, 150, 100, 250, grid = 0.05),
    case(200, 150, 100, 250, grid = 0.01),
    case(397, 78557, 51, 108778, true = 0.001, grid = 0.1),
    case(397, 78557, 51, 108778, true = 0.001, grid = 0.05),
    case(397, 78557, 51, 108778, true = 0.001, grid = 0.01),
    case(397, 78557, 51, 108778),
    case(200, 150, 100, 250, true = -0.1),
    case(30, 70, 100, 900),
    case(100, 0, 50, 50),
    case(22, 78, 18, 82),
    case(40000, 60000, 20000, 80000, grid = 0.3)
  )
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), na = "null", pretty = TRUE),
           "tests/reference_parity/_fixtures/evalue_rd_grid_R.json")
