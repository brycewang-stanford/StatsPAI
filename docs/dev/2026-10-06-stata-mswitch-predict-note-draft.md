# Draft note to StataCorp on `predict` after `mswitch ar ..., switch()`

Status: draft, not sent. For Bryce to edit and send to Stata technical
support if he wishes. Found while building `sp.mswitch` (2026-10-06). The
evidence is the test
`tests/reference_parity/test_mswitch_parity.py::test_ar_switching_regressor_probabilities_follow_statsmodels`,
on the committed file `tests/reference_parity/_fixtures/mswitch.csv`
(columns `t`, `y_z`, `z`) and the Stata output in
`tests/reference_parity/_fixtures/mswitch_Stata_3.json` (case `ar1z`).

This one is less clear-cut than the `tsfilter cf, stationary` note: it
rests on a comparison with a second implementation, not on Stata's own
manual. It is worth sending as a question.

---

Subject: mswitch ar with switch(): probabilities from predict do not match the fitted likelihood

Stata 18.0 MP, macOS.

After

```stata
import delimited "mswitch.csv", clear asdouble
tsset t
mswitch ar y_z, ar(1) switch(z)
predict double pf*, pr smethod(filter)
predict double ps*, pr smethod(smooth)
```

the filtered and smoothed state probabilities differ from the ones implied
by the model whose likelihood `mswitch` reports.

What was compared, all at Stata's own `e(b)`:

- The log likelihood. An independent implementation of the
  Markov-switching autoregression with a switching regressor,
  `(y_t - mu_s - z_t b_s) = phi (y_{t-1} - mu_s' - z_{t-1} b_s') + e_t`
  with the expanded state `(s_t, s_{t-1})` and the ergodic initial
  distribution, reproduces `e(ll)` to 6e-14. So does statsmodels'
  `MarkovAutoregression` with `exog=z` and `switching_exog=True`
  (relative difference 1e-12). The likelihood is therefore the one of
  that model.
- The probabilities. The two independent implementations agree with each
  other on the filtered and smoothed probabilities to 1e-11. Stata's
  `predict, pr smethod(filter)` differs from both by up to 0.04 (and
  `smethod(smooth)` similarly).

For every other `mswitch` specification tried, 17 of them, including
`mswitch ar` without `switch()`, with a non-switching regressor, with
`arswitch` and with `varswitch`, and `mswitch dr` with `switch()`, Stata's
probabilities agree with the independent computation to 1e-11 or better. The
difference appears only when `switch()` is combined with `mswitch ar`.

A guess, not verified: in the prediction step the lagged term
`y_{t-1} - mu_s' - z_{t-1} b_s'` may be formed without the switching
regressor of the previous period, or with the current state's coefficient.

Is the behaviour of `predict` intended here?
