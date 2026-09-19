"""Write the panel used by ``test_did_synth_mc_parity.py``.

Run once from the repository root::

    python tests/reference_parity/_generate_did_synth_mc_data.py

and then ``Rscript tests/reference_parity/_generate_did_synth_mc_R.R``, which
reads these exact bytes back. The draws use a fixed seed; the CSV is
committed, so the test never depends on this script being re-run.

Panel ``did_synth_mc_panel.csv``
--------------------------------
Balanced, 40 units x 25 periods (``unit`` 1..40, ``time`` 1..25).
``Y(0) = 10 + alpha_i + xi_t + L_it + eps`` with unit effects
``alpha_i ~ N(0, 2^2)``, a time trend-plus-noise ``xi_t``, a rank-2 factor
term ``L = F Lambda'`` and ``eps ~ N(0, 0.5^2)``. The outcome level is far
from zero on purpose: a nuclear-norm fit WITHOUT unpenalised fixed effects
shrinks that level towards zero, which is what separates the
``fixed_effects='none'`` and ``'two-way'`` estimators.

Two designs share the bytes:

* ``d`` -- staggered absorbing adoption for ``sp.mc_panel``: units 29..40
  adopt at periods 16..21 (two units per cohort) and get a heterogeneous
  effect ``2 + 0.1 * (t - g)`` in treated cells.
* ``sp.mc_synth`` treats unit 40 alone from period 21 onward (``d`` is not
  read by ``mc_synth``; its post cells are the masked entries, and the
  other staggered units stay in the donor pool with their observed --
  partly treated -- outcomes, which is exactly what ``mc_synth`` does).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

_FIX = pathlib.Path(__file__).resolve().parent / "_fixtures"


def _panel(seed: int = 20260918) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, periods = 40, 25
    alpha = rng.normal(0.0, 2.0, n)
    xi = 0.15 * np.arange(periods) + rng.normal(0.0, 0.5, periods)
    lam = rng.normal(0.0, 1.0, (n, 2))
    f = rng.normal(0.0, 1.0, (periods, 2))
    low_rank = lam @ f.T
    eps = rng.normal(0.0, 0.5, (n, periods))
    y0 = 10.0 + alpha[:, None] + xi[None, :] + low_rank + eps

    adopt = np.zeros(n, dtype=int)
    for k, g in enumerate(range(16, 22)):
        adopt[28 + 2 * k] = g
        adopt[29 + 2 * k] = g

    rows = []
    for i in range(n):
        for t in range(periods):
            period = t + 1
            treated = int(adopt[i] > 0 and period >= adopt[i])
            eff = (2.0 + 0.1 * (period - adopt[i])) if treated else 0.0
            rows.append((i + 1, period, y0[i, t] + eff, treated))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "d"])


if __name__ == "__main__":
    _panel().to_csv(_FIX / "did_synth_mc_panel.csv", index=False, float_format="%.17g")
