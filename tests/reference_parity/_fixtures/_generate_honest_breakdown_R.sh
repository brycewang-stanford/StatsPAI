#!/bin/sh
# Runs that make honest_breakdown_R.json, one short R process each
# (see _generate_honest_breakdown_R.R). From this directory:
#   rm -f honest_breakdown_R.json && sh _generate_honest_breakdown_R.sh
# Each grid brackets the restriction's crossing: a coarse sweep plus a fine
# one around it, so the bracket HonestDiD gives is narrow.
set -e
while read -r d m; do
  [ -z "$d" ] && continue
  Rscript _generate_honest_breakdown_R.R "$d" "$m"
  echo "done $d"
done <<RUNS
SD 0.0,0.00419068,0.00754323,0.00835622,0.00840651,0.0092195,0.01676273
SDPB 0.0,0.00423226,0.00761806,0.00843912,0.00848991,0.00931096,0.01692903
SDNB 0,0.01,0.05,0.2,1.0
SDD 0,0.01,0.05,0.2,1.0
RM 0.0,0.28381338,0.51086409,0.56592388,0.56932964,0.62438944,1.13525352
RMPB 0.0,0.25849954,0.46529917,0.51544808,0.51855007,0.56869898,1.03399815
RMNB 0.0,0.00129781,0.00233606,0.00258783,0.00260341,0.00285518,0.00519124
RMI 0.0,0.248757,0.44776261,0.49602146,0.49900655,0.54726541,0.99502801
SDRM 0.0,0.07745664,0.13942195,0.15444854,0.15537802,0.1704046,0.30982655
SDRMPB 0.0,0.07745664,0.13942195,0.15444854,0.15537802,0.1704046,0.30982655
RUNS
