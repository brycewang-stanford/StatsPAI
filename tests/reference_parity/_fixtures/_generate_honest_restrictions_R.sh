#!/bin/sh
# Runs that make honest_restrictions_R.json, one short R process each
# (see _generate_honest_restrictions_R.R). From this directory:
#   rm -f honest_restrictions_R.json && sh _generate_honest_restrictions_R.sh
set -e
RUNS="
corr33 SD Conditional
corr33 SDPB Conditional
corr33 SDNB Conditional
corr33 SDI Conditional
corr33 SDD Conditional
nonuis21 SD Conditional
nonuis21 SDPB Conditional
nonuis21 SDNB Conditional
nonuis21 SDI Conditional
nonuis21 SDD Conditional
corr33 SD C-F
corr33 SDPB C-F
corr33 SDI C-F
corr33 SD C-F|shipped
nonuis21 SD C-F
nonuis21 SDNB C-F
nonuis21 SDD C-F
corr33 SD C-LF
corr33 SDNB C-LF
corr33 RMPB Conditional
corr33 RMNB Conditional
corr33 RMI Conditional
corr33 RMD Conditional
corr33 SDRM Conditional
corr33 SDRMPB Conditional
corr33 SDRMNB Conditional
corr33 SDRMI Conditional
corr33 SDRMD Conditional
nonuis21 RMPB Conditional
nonuis21 RMD Conditional
nonuis21 SDRM Conditional
nonuis21 SDRMNB Conditional
nonuis21 SDRMI Conditional
corr42 RMI Conditional
corr42 SDRM Conditional
corr42 SDRMPB Conditional
corr33 RMPB C-LF
corr33 SDRM C-LF
"
echo "$RUNS" | while read -r c d m; do
  [ -z "$c" ] && continue
  Rscript _generate_honest_restrictions_R.R "$c" "$d" "$m"
  echo "done $c $d $m"
done
