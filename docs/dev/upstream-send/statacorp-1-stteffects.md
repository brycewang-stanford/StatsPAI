Stata/MP 18.0, macOS.

`stteffects ipwra ..., atet` seems to average the treated-weighted outcome
models' predictions over all observations, not over the treated. When the
outcome model is correct this estimates the ATE. `stteffects ra ..., atet`
and `stteffects ipw ..., atet` do average over the treated.

Reproduction. Weibull outcomes with a treatment effect that varies with
`x1`, and treatment selected on `x1`, so the ATE and the ATET differ.

```stata
version 18
clear all
set seed 20261020
set obs 4000
gen double x1 = rnormal()
gen byte   x2 = runiform() < 0.5
gen byte   a  = runiform() < invlogit(1.2*x1)
* Weibull AFT outcome, shape s = 1.5; mean exp(xb)*Gamma(1 + 1/s)
local s = 1.5
gen double xb0 = 0.0*x1 - 0.2*x2 + 1.0
gen double xb1 = 0.6*x1 - 0.2*x2 + 1.5
gen double m0 = exp(xb0 + lngamma(1 + 1/`s'))
gen double m1 = exp(xb1 + lngamma(1 + 1/`s'))
gen double T = exp(cond(a, xb1, xb0)) * (-ln(runiform()))^(1/`s')
gen double C = exp(3) * (-ln(runiform()))^(1/1.5)
gen double time  = min(T, C)
gen byte   event = T <= C
* sample estimands from the true potential-outcome means
gen double te = m1 - m0
quietly summarize te
display "true sample ATE  = " %9.4f r(mean)
quietly summarize te if a == 1
display "true sample ATET = " %9.4f r(mean)
stset time, failure(event)
stteffects ra    (x1 x2) (a), atet
stteffects ipwra (x1 x2) (a x1 x2)
stteffects ipwra (x1 x2) (a x1 x2), atet
```

Output on our machine.

| Quantity | Value |
| --- | --- |
| true sample ATE | 2.2506 |
| true sample ATET | 3.5395 |
| `stteffects ra, atet` | 3.4056 (SE 0.1432) |
| `stteffects ipwra` (ATE) | 2.1766 (SE 0.1017) |
| `stteffects ipwra, atet` | 2.1948 (SE 0.0972) |

The RA estimate of the ATET is close to the ATET. The IPWRA estimate of
the ATET is close to the ATE and about 14 standard errors from the ATET.

A second data set gives the same picture. On 800 observations
([`stteffects_data.csv`](https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/6fdc22bef90d03defabfc45b5b26efd9cde60e8f/tests/reference_parity/_fixtures/stteffects_data.csv),
variables `time event a x1 x2`, `stset time, failure(event)`) Stata reports 2.3146 for `ra, atet` and 1.7575 for
`ipwra, atet`, against 1.8010 for the IPWRA ATE. We took the coefficients
of the two outcome models that `ipwra, atet` fits and averaged their
predicted means ourselves. Averaging over all 800 rows gives 1.7575, which
matches Stata to 1e-9. Averaging over the 320 treated rows gives 2.2684.
The same holds with a censoring model, `stteffects ipwra (x1 x2) (a x1 x2) (x1), atet`.
Stata reports 1.4122, the all-rows average reproduces it, and the treated
average is 1.6542.

As we read Methods and formulas for `stteffects`, the moment conditions
for the potential-outcome means under `ipwra` with `atet` (equations 15e
and 15f) carry no treatment indicator. That would explain the numbers.

Is this intended? If so, the ATET label seems misleading. If not, the
averages should run over the treated observations.
