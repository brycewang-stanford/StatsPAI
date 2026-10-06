# Draft note to StataCorp on `tsfilter cf, stationary`

Status: final text, to be sent by Bryce to Stata technical support
(decision delegated 2026-10-06: send). Not sent by the assistant.

---

Subject: tsfilter cf, stationary: weight on the first and last observation is off by one lag

Stata 18.0 MP, macOS.

The entry [TS] tsfilter cf says that with the `stationary` option all
weights are set to the ideal band-pass weights. What `tsfilter cf,
stationary` computes differs from that at the two ends of the sample:

1. At an interior date `t`, the weight on the last observation `y_T` is
   the ideal weight of lag `T - t - 1` where the documented filter has lag
   `T - t`, and the weight on `y_1` is that of lag `t - 2` where it has
   `t - 1`. All other weights are the ideal ones.
2. At the first and the last date the result is the value of the default
   (random-walk) filter, with weights that sum to zero.
3. With `smaorder(q) stationary` the two outermost weights are the ideal
   weight of lag `q - 1` where the documented filter has lag `q`.

A unit impulse at the last observation shows the first point. The filtered
value at date `t` is then the weight the filter puts on `y_T`:

```stata
clear all
set obs 40
gen t = _n
tsset t
gen double x = (t == 40)
tsfilter cf c = x, stationary minperiod(6) maxperiod(32)
gen int j = 40 - t
gen double b = (sin(j*2*_pi/6) - sin(j*2*_pi/32)) / (_pi*j) if j > 0
replace b = (2*_pi/6 - 2*_pi/32)/_pi if j == 0
gen double b_next_smaller = b[_n+1]
format c b b_next_smaller %12.8f
list t j c b b_next_smaller in 30/40, noobs sep(0)
```

Output (Stata 18.0 MP):

```
  t    j             c             b   b_next_sm~r
 30   10   -0.03468818   -0.05697444   -0.03468818
 31    9   -0.00533068   -0.03468818   -0.00533068
 32    8   -0.00521846   -0.00533068   -0.00521846
 33    7   -0.04901333   -0.00521846   -0.04901333
 34    6   -0.10806589   -0.04901333   -0.10806589
 35    5   -0.12518588   -0.10806589   -0.12518588
 36    4   -0.05894783   -0.12518588   -0.05894783
 37    3    0.07692626   -0.05894783    0.07692626
 38    2    0.21356528    0.07692626    0.21356527
 39    1    0.27083334    0.21356527    0.27083333
 40    0    0.13541667    0.27083333             .
```

Column `c` is what `tsfilter` returns, `b` is the ideal weight at lag `j`,
and `b_next_smaller` is the ideal weight at lag `j - 1`. `c` equals the
third column, not the second. At `t = 39` the weight on the next
observation is `b_0`, the weight an observation should get only at its own
date.

On a series with a non-zero level the effect is large, because the
misplaced weights no longer sum to anything close to the documented sum.
On a simulated quarterly series with a level near 1000 and a cycle of
standard deviation about 1.5, the returned "cycle" ranges from -131 to
615.

The default (random-walk) filter, `smaorder()` without `stationary`,
`drift`, and the `hp`, `bk` and `bw` filters all agree with independent
computations to 1e-12 or better, so this is confined to the `stationary`
option.

Either the code or the manual entry could be the intended behaviour; I
could not tell which from the Methods and formulas section.
