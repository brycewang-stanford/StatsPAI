# Draft note to the xtevent authors (SSC) on trend() after the window start

Status: draft, not sent. Bryce sends it himself, for example as an issue
on the package's public repository or by email to the maintainers listed
in the help file.

The evidence in this repository is
`tests/reference_parity/test_xtevent_extensions_Stata_parity.py`
(`test_trend_after_window_start_is_the_delta_method`,
`test_stata_trend_variance_uses_the_window_start`), fixture
`tests/reference_parity/_fixtures/xtevent_continuous.csv`, Stata output
`_fixtures/xtevent_extensions_Stata.json` from
`_fixtures/_generate_xtevent_extensions_Stata.do`, and the note on
`sp.xtevent(trend=, proxy=)` in `docs/dev/modules/did.md`. xtevent ships
without a stated licence, so it was used as a black box. Only its help
file and outputs were read.

The snippet below was run on 2026-10-20 with xtevent 3.1.0 (July 11, 2024)
on Stata/MP 18.0. If SSC has a newer version, rerun it first.

---

Subject: xtevent trend() standard errors when the trend starts after the window start

Thank you for xtevent. We have been comparing our own implementation with
it and found one case we cannot reconcile.

With `trend(-2)` and `window(3)`, the point estimates after the trend
adjustment are what we expect. The standard errors are not. They equal
the delta-method variance with the trend weight placed on the coefficient
at k = -3 (the window start), while the trend was fitted on the
coefficient at k = -2. With `trend(-3)`, where the two coincide, our
standard errors and xtevent's agree to 1e-9.

The clearest symptom is the coefficient at k = -2 itself. After
`trend(-2)` it is identically zero by construction, yet its reported
standard error is 0.2125.

Reproduction. The data are a public CSV, 80 units by 20 periods.

```stata
version 18
clear all
* ssc install xtevent
import delimited using "https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/6fdc22bef90d03defabfc45b5b26efd9cde60e8f/tests/reference_parity/_fixtures/xtevent_continuous.csv", clear asdouble
xtset id t

* 1. no trend adjustment: raw event-time coefficients and covariance
quietly xtevent y x, policyvar(z) window(3) vce(cluster cl)
matrix b_raw = e(b)
matrix V_raw = e(V)

* 2. trend(-2)
quietly xtevent y x, policyvar(z) window(3) trend(-2) vce(cluster cl)
matrix b_tr = e(b)
matrix V_tr = e(V)

mata:
    names = st_matrixcolstripe("b_raw")[., 2]
    br = st_matrix("b_raw"); Vr = st_matrix("V_raw")
    bt = st_matrix("b_tr");  Vt = st_matrix("V_tr")
    namt = st_matrixcolstripe("b_tr")[., 2]
    ks = J(0, 1, .); pos = J(0, 1, .)
    for (j = 1; j <= rows(names); j++) {
        s = names[j]
        if (substr(s, 1, 6) == "_k_eq_") {
            k = strtoreal(substr(s, 8, .)) * (substr(s, 7, 1) == "m" ? -1 : 1)
            ks = ks \ k; pos = pos \ j
        }
    }
    V = Vr[pos, pos]; b = br[pos]'
    h = ks :+ 1
    // delta method: delta*_k = delta_k + (k + 1) * delta_{-2}
    e2 = (ks :== -2)
    A2 = I(rows(ks)) + h * e2'
    // the same with the weight placed on the coefficient at -3
    e3 = (ks :== -3)
    A3 = I(rows(ks)) + h * e3'
    se2 = sqrt(diagonal(A2 * V * A2'))
    se3 = sqrt(diagonal(A3 * V * A3'))
    printf("\n%10s %12s %12s %12s %12s %12s\n", "k", "b_trend", "b_delta", "se_reported", "se_delta", "se_w_at_-3")
    for (i = 1; i <= rows(ks); i++) {
        jt = selectindex(namt :== names[pos[i]])
        printf("%10.0f %12.6f %12.6f %12.6f %12.6f %12.6f\n", ks[i], bt[jt], (A2 * b)[i], sqrt(Vt[jt, jt]), se2[i], se3[i])
    }
end
```

The trend line passes through zero at k = -1 and through the coefficient
at k = -2, so its slope is -delta_{-2} and the adjusted coefficients are
delta*_k = delta_k + (k + 1) delta_{-2}. The columns are the reported
estimate after `trend(-2)`, that formula applied to the raw estimates, the
reported standard error, the delta-method standard error of the formula,
and the same with delta_{-3} in place of delta_{-2} in the variance only.

Output on our machine (first six rows).

| k | b_trend | b_delta | se_reported | se_delta | se_w_at_-3 |
| --- | --- | --- | --- | --- | --- |
| -4 | 0.595711 | 0.595711 | 0.631540 | 0.690472 | 0.631540 |
| -3 | 0.242547 | 0.242547 | 0.278989 | 0.428449 | 0.278989 |
| -2 | 0.000000 | 0.000000 | 0.212509 | 0.000000 | 0.212509 |
| 0 | 0.154348 | 0.154348 | 0.455232 | 0.447986 | 0.455232 |
| 1 | 0.343008 | 0.343008 | 0.708634 | 0.674176 | 0.708634 |
| 2 | -0.065658 | -0.065658 | 1.122695 | 1.156785 | 1.122695 |

The estimates agree with the formula. The reported standard errors equal
the last column in every row, to 1e-9 in our tests.

Could the covariance adjustment be indexing the first coefficient of the
window rather than the first coefficient used for the trend? We have not
read the code, so this is only a guess from the outputs.
