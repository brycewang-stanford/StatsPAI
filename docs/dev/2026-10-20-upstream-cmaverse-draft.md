# Draft issue for R CMAverse on delta-method SEs of CDE and INTref

Status: draft, not sent. Bryce posts it himself on CMAverse's GitHub issue
tracker.

The evidence in this repository is
`tests/reference_parity/test_four_way_nonlinear_parity.py`
(`test_ols_estimates_match_cmaverse`; its docstring records why the normal
mediator SEs are not compared), fixture
`tests/reference_parity/_fixtures/four_way_nonlinear.csv`, CMAverse output
`_fixtures/four_way_nonlinear_R.json` from
`_fixtures/_generate_four_way_nonlinear_R.R`, Stata `med4way` output
`_fixtures/four_way_nonlinear_Stata.json`, and the note on
`mediation/four_way.py` in `docs/dev/modules/treatment_effects.md`.

CMAverse is GPL-3. It was used as a black box, outputs only. Nobody
preparing this draft read its source. Keep it that way when following up.

The snippet below was run on 2026-10-20 with CMAverse 0.1.0 on R 4.5.2.
The bootstrap comparison is new for this draft and was run once
(`set.seed(1)`, 1,000 replicates).

---

Title: cmest delta-method SEs for ERcde and ERintref with a normal mediator and a log-linear outcome

CMAverse 0.1.0, R 4.5.2.

With `model = "rb"`, `estimation = "paramfunc"`, `inference = "delta"`, a
linear (normal) mediator model and a logistic or Poisson outcome model,
the standard errors of `ERcde` and `ERintref` differ from two independent
delta-method computations and from CMAverse's own bootstrap. The point
estimates agree with ours to 1e-8. The SEs of `ERintmed` and `ERpnie`
agree with ours to about 1e-4.

Reproduction. A public CSV with 1,500 rows. Exposure `a`, normal mediator
`mc`, covariates `c1 c2`, binary outcome `ybc` and count outcome `ypc`.

```r
library(CMAverse)
url <- paste0("https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/",
              "6fdc22bef90d03defabfc45b5b26efd9cde60e8f/",
              "tests/reference_parity/_fixtures/four_way_nonlinear.csv")
df <- read.csv(url)
keep <- c("Rte", "ERcde", "ERintref", "ERintmed", "ERpnie")
fit <- function(y, yreg, mval, inference, nboot = 200) {
  set.seed(1)
  cmest(data = df, model = "rb", outcome = y, exposure = "a", mediator = "mc",
        basec = c("c1", "c2"), EMint = TRUE, mreg = list("linear"), yreg = yreg,
        astar = 0, a = 1, mval = list(mval), yval = 1,
        estimation = "paramfunc", inference = inference, nboot = nboot)
}
for (cs in list(list("ybc", "logistic", 0), list("ypc", "poisson", 1))) {
  d <- fit(cs[[1]], cs[[2]], cs[[3]], "delta")
  invisible(capture.output(b <- fit(cs[[1]], cs[[2]], cs[[3]], "bootstrap", 1000)))
  cat("\noutcome", cs[[1]], "yreg =", cs[[2]], "\n")
  print(round(cbind(estimate = d$effect.pe[keep], se_delta = d$effect.se[keep],
                    se_bootstrap = b$effect.se[keep]), 5))
}
```

Logistic outcome `ybc`, `mval = 0`.

| Effect | Estimate | SE, cmest delta | SE, cmest bootstrap | SE, our delta method | SE, Stata `med4way` |
| --- | --- | --- | --- | --- | --- |
| Rte | 3.20115 | 0.50312 | 0.53213 | | 0.50289 |
| ERcde | 0.28996 | 0.07628 | 0.22206 | 0.21288 | 0.21299 |
| ERintref | 0.40492 | 0.19925 | 0.16292 | 0.15690 | 0.15656 |
| ERintmed | 0.85683 | 0.35520 | 0.37793 | 0.35525 | 0.35484 |
| ERpnie | 0.64944 | 0.16887 | 0.17153 | 0.16888 | 0.16885 |

Poisson outcome `ypc`, `mval = 1`.

| Effect | Estimate | SE, cmest delta | SE, cmest bootstrap | SE, our delta method |
| --- | --- | --- | --- | --- |
| Rte | 1.84483 | 0.10243 | 0.10686 | |
| ERcde | 0.42043 | 0.03322 | 0.09531 | 0.08959 |
| ERintref | -0.09387 | 0.04832 | 0.02233 | 0.02144 |
| ERintmed | 0.24571 | 0.06039 | 0.06179 | 0.06040 |
| ERpnie | 0.27257 | 0.04652 | 0.04751 | 0.04652 |

Our column is an independent implementation of the same estimator with
the mediator variance estimated as RSS/(n - p), as in CMAverse. Stata
`med4way` 2.3.2 uses RSS/n, so its numbers differ slightly for that reason
alone. For ERcde the cmest delta SE is about 2.8 (logistic) and 2.7
(Poisson) times smaller than the other three. For ERintref it is about 1.3
and 2.3 times larger. The bootstrap of cmest itself sides with the other
two delta-method computations.

With a logistic mediator the cmest delta SEs agree with ours to 5e-5 on
the same data, so the gap seems specific to the normal mediator. With a
normal mediator and a log-linear outcome, CDE and INTref depend on the
estimated mediator variance (through the normal moment generating
function), while PIE does not. One more observation, from comparing
outputs only. The cmest delta SEs behave as if the sampling error of the
estimated mediator variance were left out. We have not checked whether that
explains the size of the gap.

Could the delta-method gradient for ERcde and ERintref be missing a term
in this model combination?
