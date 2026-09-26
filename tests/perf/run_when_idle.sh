#!/usr/bin/env bash
# Run the Track C benchmarks once the machine is idle, and keep it idle.
#
#   tests/perf/run_when_idle.sh [SRC]
#
# SRC is the statspai source tree to benchmark (default: this checkout's
# src/). It (1) writes the shared inputs (tests/perf/_data.py), (2) pins
# every thread pool to one thread on both sides, (3) before every timed
# step waits until the 1-minute load average has stayed below $LOAD_MAX
# (default 1.5) for $IDLE_SAMPLES one-minute samples (default 5), (4) samples
# the load every 15 s while the step runs and repeats the step, up to
# $MAX_TRIES times (default 3), if it ever exceeded $LOAD_MAX_DURING
# (default 2.5; the step itself contributes about 1), and (5) runs both
# sides of each module under one run id, which compare_perf.py requires.
# Wall-clock timings from a busy machine are not comparable: a run under
# load ~9 once inflated ratios by 20-30%, and on 2026-09-27 a concurrent
# test suite pushed the load to ~30 half-way through the SCM leg of a run
# that had started idle -- a start-only check cannot see that.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$HERE/../../src}"
LOAD_MAX="${LOAD_MAX:-1.5}"
LOAD_MAX_DURING="${LOAD_MAX_DURING:-2.5}"
IDLE_SAMPLES="${IDLE_SAMPLES:-5}"
MAX_TRIES="${MAX_TRIES:-3}"
export PYTHONPATH="$SRC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export STATSPAI_PERF_RUN_ID="${STATSPAI_PERF_RUN_ID:-$(date +%Y%m%dT%H%M%S)}"
PY="${PYTHON:-python3}"
command -v Rscript >/dev/null || { echo "Rscript not found: the reference legs cannot run" >&2; exit 2; }
load1() { if command -v sysctl >/dev/null && sysctl -n vm.loadavg >/dev/null 2>&1; then sysctl -n vm.loadavg | awk '{print $2}'; else awk '{print $1}' /proc/loadavg; fi; }
cd "$HERE"
"$PY" _data.py
wait_idle() {
  local streak=0 l
  while (( streak < IDLE_SAMPLES )); do
    l=$(load1)
    if awk -v l="$l" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then streak=$((streak+1)); else streak=0; fi
    (( streak >= IDLE_SAMPLES )) && break
    echo "$(date +%T) waiting, load=$l"; sleep 60
  done
}

# guarded LABEL CMD... : run CMD on an idle machine; rerun if the load
# rose above LOAD_MAX_DURING at any 15-second sample while it ran.
guarded() {
  local label="$1"; shift
  local try maxfile peak sampler
  for (( try = 1; try <= MAX_TRIES; try++ )); do
    wait_idle
    maxfile=$(mktemp)
    echo 0 > "$maxfile"
    ( while true; do
        l=$(load1)
        awk -v l="$l" -v f="$maxfile" 'BEGIN{getline m < f; if (l > m) print l > f}'
        sleep 15
      done ) &
    sampler=$!
    echo "=== $label try $try $(date +%T) load=$(load1)"
    "$@"
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    peak=$(cat "$maxfile"); rm -f "$maxfile"
    if awk -v p="$peak" -v m="$LOAD_MAX_DURING" 'BEGIN{exit !(p < m)}'; then
      echo "    $label peak load $peak < $LOAD_MAX_DURING: kept"
      return 0
    fi
    echo "    $label peak load $peak >= $LOAD_MAX_DURING: timing discarded, rerunning"
  done
  echo "FAIL -- $label never ran on an idle machine in $MAX_TRIES tries" >&2
  return 3
}

"$PY" -c "import statspai; print('benchmarking statspai', statspai.__version__, 'from', statspai.__file__)"
echo "run id $STATSPAI_PERF_RUN_ID"
for m in 01_hdfe 02_csdid 03_scm; do
  guarded "$m py" "$PY" "${m}_perf.py"
  guarded "$m R" Rscript "${m}_perf.R"
done
guarded "04_dml py + doubleml-for-py" "$PY" 04_dml_perf.py
"$PY" compare_perf.py
echo "=== done $(date +%T) load=$(load1)"
