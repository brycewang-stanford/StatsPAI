Stata/MP 18.0, macOS.

```stata
gsdesign onemean 10 15, sd(12) onesided power(0.9) futility(errpocock) ///
    nfractional intpointsscale(60) infotol(1e-12)
return list
```

This returns `r(N_max) = 59.1659` and `r(ESS1) = 59.4171`. The default
tolerances give the same picture (59.1656 and 59.4168). An expected sample
size cannot exceed the maximum sample size.

We can reproduce Stata's `r(ESS0)` and `r(ESS1)` to 1e-7, for this design
and for eight other designs with nonbinding futility bounds, with this
rule. The weight on look k is the efficacy-crossing probability computed
with the futility bounds ignored, plus the futility-crossing probability
computed with the futility bounds followed. A path that crosses a futility
bound and would later cross the efficacy bound then counts twice, and the
weights sum to more than one.

For efficacy-only designs and for designs with binding futility, our
expected sample sizes match Stata's to 1e-7. For nonbinding designs we
compute the expected sample size when the futility rule is followed.
Those stopping probabilities sum to one. On one of these designs a
simulation of 200,000 Brownian paths agrees with it within 4 Monte Carlo
standard errors. For the design
above that gives ESS1 = 57.33 and ESS0 = 36.77 (Stata 59.42 and 37.23).

As we read Methods and formulas, it describes a third rule. The futility
bound is dropped when delta = 0 and kept otherwise. That rule reproduces
neither of Stata's two numbers.

Which quantity is `r(ESS1)` meant to be under nonbinding futility?
