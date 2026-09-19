#!/usr/bin/env Rscript
# Writes the public datasets used by the spatial parity fixture to CSV, ONCE.
# Both sides (_generate_spatial_survey_R.R and
# test_spatial_survey_R_parity.py) then read the same bytes.
#
#   spatial_survey_columbus.csv  Columbus OH crime polygons (spData 2.3.5,
#                                shapes/columbus.gpkg): attributes + polygon
#                                WKT at 17 significant digits, plus a
#                                deterministic 4-level `regime` (POLYID %% 4)
#                                for block weights.
#   spatial_survey_georgia.csv   Georgia county education data (GWmodel,
#                                data(Georgia) -> Gedu.df), the GWR / MGWR
#                                benchmark; projected coordinates X, Y (m).
#   spatial_survey_produc.csv    US state production panel (plm::Produc,
#                                48 states x 17 years), log-transformed as in
#                                the splm examples.
#   spatial_survey_usaww.csv     splm::usaww, the 48 x 48 binary contiguity
#                                matrix aligned with Produc's states.
#
# Regenerate (only if the source packages change):
#   Rscript tests/reference_parity/_prepare_spatial_survey_data.R
suppressPackageStartupMessages({
  library(sf); library(GWmodel); library(plm); library(splm)
})
args_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(args_file)) dirname(normalizePath(args_file)) else "tests/reference_parity"
fx <- file.path(here, "_fixtures")

col <- st_read(system.file("shapes/columbus.gpkg", package = "spData"), quiet = TRUE)
columbus <- data.frame(
  POLYID = col$POLYID, CRIME = col$CRIME, INC = col$INC, HOVAL = col$HOVAL,
  DISCBD = col$DISCBD, X = col$X, Y = col$Y, regime = col$POLYID %% 4,
  wkt = st_as_text(st_geometry(col), digits = 17)
)
write.csv(columbus, file.path(fx, "spatial_survey_columbus.csv"), row.names = FALSE)

data(Georgia, package = "GWmodel")
write.csv(Gedu.df[, c("AreaKey", "X", "Y", "PctBach", "PctRural", "PctPov", "PctBlack")],
          file.path(fx, "spatial_survey_georgia.csv"), row.names = FALSE)

data(Produc, package = "plm")
data(usaww, package = "splm")
pr <- data.frame(state = as.character(Produc$state), year = Produc$year,
                 lgsp = log(Produc$gsp), lpcap = log(Produc$pcap),
                 lpc = log(Produc$pc), lemp = log(Produc$emp), unemp = Produc$unemp)
stopifnot(identical(sort(unique(pr$state)), rownames(usaww)))
write.csv(pr, file.path(fx, "spatial_survey_produc.csv"), row.names = FALSE)
write.csv(usaww, file.path(fx, "spatial_survey_usaww.csv"))
cat("wrote 4 CSVs to", fx, "\n")
