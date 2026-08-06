* tests/stata_parity/70_policy_tree.do
*
* Module 70: exact welfare-maximising policy tree.
*   StatsPAI:  sp.policy_tree(depth=1, 2)
*   R:         policytree::policy_tree
*   Stata:     audited Mata exhaustive search (depth 1 only)   <-- this file
*
* Scope, stated up front: this bridge covers **depth 1 only**. The four
* depth-2 statistics stay py<->R rows.
*
* Depth 1 is an exact, cheap search and it is where the conventions live. On
* a supplied score vector the objective is
*
*     V(pi) = (1/n) * sum_i gamma_i * pi(x_i)
*
* and the depth-1 optimum is the best over every (variable, threshold) pair
* and both leaf orientations. That is O(k n log n) with a sort and a running
* sum, so it is reproducible exactly rather than approximately, and it pins
* the three things a reviewer would actually doubt: the objective's scaling,
* the `x <= t` split convention, and the threshold grid (the distinct
* observed values of each covariate, not a quantile grid).
*
* Depth 2 is deliberately not attempted. The exact depth-2 optimum is a
* joint search over a root split and an independent depth-1 tree in each
* child; policytree makes it tractable with an incremental update scheme,
* and a naive re-derivation here would either be too slow to run in the
* harness or -- worse -- a heuristic that agrees on this fixture and
* silently disagrees on the next one. A bridge that might be a local optimum
* is not evidence about an exact optimiser. Porting policytree's search is
* the promotion path; until then the depth-2 rows are honestly two-sided.
*
* Tolerance: rel < 1e-6 on the depth-1 statistics. Both sides maximise the
* same finite objective over the same finite grid, so agreement should be at
* floating-point noise.

version 18
clear all

do _common.do
stata_parity_init, module(70_policy_tree)
stata_parity_open, module(70_policy_tree)

import delimited "${STATA_PARITY_DATA}/70_policy_tree.csv", clear case(preserve)
count
local n = r(N)

mata:
X     = st_data(., ("x1", "x2", "x3"))
gamma = st_data(., "gamma")
n     = rows(gamma)
k     = cols(X)

// Exhaustive depth-1 search. For a split (j, t) the policy is
//   pi_i = 1 if x_ij <= t  (orientation "le"), else 0
// or its complement. Sorting by x_j turns "sum of gamma over {x_j <= t}"
// into a running total, so every candidate threshold is O(1) after the
// sort. Ties matter: all observations sharing a value must fall on the
// same side, so a candidate threshold is only evaluated at the last index
// of each run of equal values.
best_val  = .
best_var  = .
best_thr  = .
best_frac = .

for (j = 1; j <= k; j++) {
    xj  = X[., j]
    ord = order(xj, 1)
    xs  = xj[ord]
    gs  = gamma[ord]
    run = 0
    for (i = 1; i <= n; i++) {
        run = run + gs[i]
        // Only a boundary between distinct values is a real split point.
        if (i == n) {
            continue
        }
        if (xs[i] == xs[i + 1]) {
            continue
        }
        // Orientation A: treat the <= side.
        v = run / n
        if (best_val == . || v > best_val) {
            best_val  = v
            best_var  = j
            best_thr  = xs[i]
            best_frac = i / n
        }
        // Orientation B: treat the > side.
        v = (sum(gamma) - run) / n
        if (v > best_val) {
            best_val  = v
            best_var  = j
            best_thr  = xs[i]
            best_frac = (n - i) / n
        }
    }
}

// The all-treat and no-treat constant policies are also depth-1 trees and
// must be in the feasible set, or a fixture whose optimum is constant would
// report a spurious split.
v = sum(gamma) / n
if (v > best_val) {
    best_val  = v
    best_var  = 0
    best_thr  = .
    best_frac = 1
}
if (0 > best_val) {
    best_val  = 0
    best_var  = 0
    best_thr  = .
    best_frac = 0
}

st_numscalar("pt_val",  best_val)
st_numscalar("pt_var",  best_var)
st_numscalar("pt_thr",  best_thr)
st_numscalar("pt_frac", best_frac)
end

stata_parity_row, stat(value_policy_d1)        est(`=scalar(pt_val)')  nob(`n')
stata_parity_row, stat(fraction_treated_d1)    est(`=scalar(pt_frac)') nob(`n')
stata_parity_row, stat(root_split_variable_d1) est(`=scalar(pt_var)')  nob(`n')
stata_parity_row, stat(root_split_value_d1)    est(`=scalar(pt_thr)')  nob(`n')

stata_parity_extra, key(stata_bridge_status) val("audited Mata algorithm bridge, materialized 2026-08-06 with licensed Stata 18")
stata_parity_extra, key(algorithm) val("exhaustive depth-1 search over every (covariate, distinct observed threshold) pair and both leaf orientations, plus the two constant policies; objective V = mean(gamma * pi)")
stata_parity_extra, key(split_convention) val("x <= t goes left; ties are kept on the same side by only evaluating thresholds at the last index of each run of equal values")
stata_parity_extra, key(depth2_not_bridged) val("the four depth-2 statistics are not emitted: the exact depth-2 optimum needs policytree's incremental search, and a heuristic that happens to agree on this fixture would be worse evidence than an honest two-sided row")

stata_parity_close, module(70_policy_tree)
