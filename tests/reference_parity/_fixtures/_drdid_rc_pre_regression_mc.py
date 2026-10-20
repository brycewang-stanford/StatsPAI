"""Monte Carlo behind the ``variance_convention`` note of ``_rcs.drdid_rc``.

Not a test (about 8 s). Repeated cross-sections, n = 2,000, a propensity
score that the logit on ``x`` misspecifies, comparison-group outcome
regressions that are correct (so the DR estimator is consistent), true
ATT = 1. Compares the sd of the estimates with the mean analytic SE under
the delta-method influence function and under the reference convention.

Result on 2026-10-20 (seed 11, 1,000 draws):
drdid_rc1: sd 0.2770, mean SE 0.2790 delta (95% coverage 0.952),
0.1876 reference (0.813); drdid_rc: sd 0.2769, mean SE 0.2786 delta
(0.952), 0.1879 reference (0.815). See docs/dev/cleanroom/did_drdid.md.

Run from the repository root with ``PYTHONPATH=src``.
"""

import warnings

import numpy as np

from statspai.did._rcs import drdid_rc


def main(seed: int = 11, draws: int = 1000) -> None:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        n = 2000
        x = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(-1.5 + 3.0 * np.maximum(x, 0) ** 2)))
        d = (rng.random(n) < p).astype(float)
        t = (rng.random(n) < 0.5).astype(float)
        e = rng.normal(size=n) * np.where((d == 0) & (t == 0), 4.0, 0.5)
        y = 1 + 3 * x + 0.5 * d + t * (0.5 + 1.0 * x + 1.0 * d) + e
        row = []
        for eff in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = drdid_rc(y, t, d, x[:, None], locally_efficient=eff)
                dlt = drdid_rc(
                    y,
                    t,
                    d,
                    x[:, None],
                    locally_efficient=eff,
                    variance_convention="delta",
                )
            row += [ref.att, dlt.se, ref.se]
        out.append(row)
    out = np.asarray(out)
    for k, name in ((0, "drdid_rc1"), (3, "drdid_rc")):
        att, se_d, se_r = out[:, k], out[:, k + 1], out[:, k + 2]
        cov_d = np.mean(np.abs(att - 1.0) / se_d < 1.959964)
        cov_r = np.mean(np.abs(att - 1.0) / se_r < 1.959964)
        print(
            f"{name}: sd {att.std(ddof=1):.4f}  SE delta {se_d.mean():.4f} "
            f"(cover {cov_d:.3f})  SE reference {se_r.mean():.4f} (cover {cov_r:.3f})"
        )


if __name__ == "__main__":
    main()
