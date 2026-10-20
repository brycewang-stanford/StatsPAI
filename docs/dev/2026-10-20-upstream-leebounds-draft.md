# Draft note to Harald Tauchmann on leebounds with weights

Status: draft, not sent. Bryce sends it himself to the author of
`leebounds` (SSC, version 1.5, 2013-07-17).

The evidence in this repository is
`tests/reference_parity/test_lee_bounds_weights_Stata_parity.py`
(`test_weighted_bounds_match_leebounds`, which rebuilds Stata's number from
ours by swapping in the unweighted variance, to 1e-8), fixture
`tests/reference_parity/_fixtures/leebounds_weights.csv`, Stata output
`_fixtures/leebounds_weights_stata.json` from
`_fixtures/_generate_leebounds_weights_stata.do`, and the note on
`bounds/lee_manski.py` in `docs/dev/modules/treatment_effects.md`.
`leebounds` was used as a black box (help file and outputs only).

The snippet below was run on 2026-10-20 on Stata/MP 18.0 to check that it
reproduces the gap.

---

Subject: leebounds with weights, one variance term appears unweighted

Dear Professor Tauchmann,

Thank you for `leebounds`. We have been checking our own implementation of
Lee bounds against it, including with probability weights. Bounds and
standard errors agree to 1e-8 or better in every case but one, and that
one looks like an unweighted term in an otherwise weighted formula.

The term is the variance of the outcome in the untrimmed arm, as it enters
the standard error of one of the two bounds. It is the bound built from the
"upper" trimmed mean. That is the upper bound when the treated arm is
trimmed and the lower bound when the control arm is trimmed. With weights, that
variance seems to be computed without them. The same term in the other
bound, and every other term, is weighted. The gap shows only when there is
no tie at the trimming threshold.

Reproduction. Sample 2 of a public CSV, 700 observations, continuous
outcome. The control arm is more often observed, so the control arm is
trimmed and the treated arm is the untrimmed one.

```stata
version 18
clear all
* ssc install leebounds
import delimited using "https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/6fdc22bef90d03defabfc45b5b26efd9cde60e8f/tests/reference_parity/_fixtures/leebounds_weights.csv", clear asdouble
keep if sample == 2
leebounds y d [pw=w], select(s)
scalar se_lo = _se[lower]
scalar se_up = _se[upper]
* the weights rescaled to mean one, in single precision
gen float w32 = w
quietly summarize w32
gen float wn = w32/r(mean)
* untrimmed arm: observed outcomes of the treated
gen byte unt = s == 1 & d == 1
quietly summarize unt [iw=wn]
scalar esnt = r(mean)
quietly summarize wn
scalar N = round(r(sum))
quietly summarize y if unt
scalar v_unw = r(Var)
quietly summarize y [iw=wn] if unt
scalar v_w = r(Var)
* replace the unweighted variance by the weighted one in that term
scalar delta = (v_unw - v_w)/esnt/N
display "unweighted var = " v_unw "   weighted var = " v_w
display "reported se(lower)                   = " %12.10f se_lo
display "se(lower) with the weighted variance = " %12.10f sqrt(se_lo^2 - delta)
display "reported se(upper)                   = " %12.10f se_up
```

Output on our machine.

| Quantity | Value |
| --- | --- |
| unweighted variance, untrimmed arm | 1.1308379 |
| weighted variance, untrimmed arm | 1.1590096 |
| reported se(lower) | 0.1077312866 |
| se(lower) with the weighted variance | 0.1083964226 |
| reported se(upper) | 0.1053948386 |

Our implementation weights that variance throughout. It gives se(lower) =
0.1083964226 and se(upper) = 0.1053948386 here, so the swap above accounts
for the whole difference. The bounds themselves (0.1240039 and 0.8826137)
agree to 1e-12.

On the other three samples of the same file the gap does not appear. In
sample 1 the treated arm is trimmed, and our code finds a tie at the
threshold of the upper bound. Samples 3 and 4 have integer outcomes with ties.

Is the unweighted variance intended there? If not, weighting it as in the
other bound would make the two standard errors consistent.

With best regards,
