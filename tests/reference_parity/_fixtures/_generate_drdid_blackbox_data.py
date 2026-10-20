"""Designs for the DRDID black-box fixture (``drdid_blackbox_R.csv``).

Clean-room record for ``did/_rcs.py`` and the panel 2x2 engines of
``did/callaway_santanna.py`` (``docs/dev/cleanroom/did_drdid.md``). R
``DRDID`` is GPL-3 and was used only as a black box: these designs are fed
to it by ``_generate_drdid_blackbox_R.R`` and only its outputs are kept.

Designs 1-2: moderate / strong propensity gradient. Design 3: a few
comparison units placed far in the covariate tail so their propensity
scores exceed the 0.995 trimming level (6 panel, 13 cross-section rows).
Cross-section design 4: design 1 with ``x2`` constant within two cells
(1 for treated/pre, 0 for comparison/post), so those cell regressions are
singular; the reference's improved estimators drop the column there and its
other regression-based estimators stop with an error.

Run from this directory: ``python _generate_drdid_blackbox_data.py``.
"""

import numpy as np
import pandas as pd


def _covariates(r, n):
    return r.normal(size=n), (r.random(n) < 0.4).astype(float)


def panel(n, seed, slope):
    r = np.random.default_rng(seed)
    x1, x2 = _covariates(r, n)
    lin = (-1.0 if slope > 3 else -0.3) + slope * x1 + 0.5 * x2
    d = (r.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    y0 = 1 + x1 + 0.5 * x2 + 0.7 * d + r.normal(size=n)
    y1 = y0 + 0.5 + 0.4 * x1 * (1 + d) + 1.5 * d + r.normal(size=n)
    w = r.gamma(2.0, 1.0, n)
    return pd.DataFrame(
        dict(id=np.arange(1, n + 1), D=d, x1=x1, x2=x2, y0=y0, y1=y1, w=w)
    )


def cross_section(n, seed, slope):
    r = np.random.default_rng(seed)
    x1, x2 = _covariates(r, n)
    lin = (-1.0 if slope > 3 else -0.3) + slope * x1 + 0.5 * x2
    d = (r.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    post = (r.random(n) < 0.55).astype(float)
    y = (
        1
        + x1
        + 0.5 * x2
        + 0.7 * d
        + post * (0.5 + 0.4 * x1 * (1 + d) + 1.5 * d)
        + r.normal(size=n)
    )
    w = r.gamma(2.0, 1.0, n)
    return pd.DataFrame(dict(D=d, post=post, x1=x1, x2=x2, y=y, w=w))


def _push_comparisons_into_tail(df, k):
    idx = np.where(df["D"] == 0)[0][:k]
    df.loc[idx, "x1"] = np.linspace(2.6, 3.4, k)
    return df


if __name__ == "__main__":
    panel(400, 1, 0.8).to_csv("drdid_blackbox_panel_1.csv", index=False)
    panel(400, 2, 2.6).to_csv("drdid_blackbox_panel_2.csv", index=False)
    _push_comparisons_into_tail(panel(600, 3, 3.2), 8).to_csv(
        "drdid_blackbox_panel_3.csv", index=False
    )
    cross_section(900, 11, 0.8).to_csv("drdid_blackbox_rc_1.csv", index=False)
    cross_section(900, 12, 2.6).to_csv("drdid_blackbox_rc_2.csv", index=False)
    _push_comparisons_into_tail(cross_section(1500, 13, 3.2), 14).to_csv(
        "drdid_blackbox_rc_3.csv", index=False
    )
    rc4 = cross_section(900, 11, 0.8)
    rc4.loc[(rc4["D"] == 1) & (rc4["post"] == 0), "x2"] = 1.0
    rc4.loc[(rc4["D"] == 0) & (rc4["post"] == 1), "x2"] = 0.0
    rc4.to_csv("drdid_blackbox_rc_4.csv", index=False)
