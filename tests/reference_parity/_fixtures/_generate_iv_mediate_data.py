"""Simulated data for ``test_iv_mediate_stata_parity.py``.

Writes ``iv_mediate.csv``. The design follows the structure under which the
IV mediation estimator of Dippel, Ferrara and Heblich is consistent: ``v``
confounds the treatment and the mediator, ``u`` confounds the mediator and
the outcome, and the instrument ``z`` is independent of both. True effects:
T -> M 0.6, M -> Y 0.7, direct T -> Y 0.4, so the total effect is 0.82 and
the indirect effect 0.42. Two covariates, 40 groups (``g``, used for fixed
effects and clustering) and a second, coarser cluster variable ``cl``.
Values are rounded to six decimals.

Run from the repository root:
    python tests/reference_parity/_fixtures/_generate_iv_mediate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261028)
n, G = 600, 40
g = rng.integers(0, G, n) + 1
cl = (g - 1) // 4 + 1  # 10 coarser clusters
a_g = rng.normal(0, 1, G)[g - 1]
w1 = rng.standard_normal(n) + 0.5 * a_g
w2 = rng.binomial(1, 0.4, n).astype(float)
z = rng.standard_normal(n) + 0.3 * w1
v = rng.standard_normal(n)
u = rng.standard_normal(n)
t = 0.8 * z + v + 0.5 * w1 - 0.3 * w2 + a_g + 0.5 * rng.standard_normal(n)
m = 0.6 * t + v + u + 0.2 * w1 + 0.4 * w2 - 0.5 * a_g
m += (0.6 + 0.3 * np.abs(w1)) * rng.standard_normal(n)
y = 0.4 * t + 0.7 * m + u - 0.3 * w1 + 0.2 * w2 + 0.8 * a_g
y += (0.8 + 0.4 * np.abs(z)) * rng.standard_normal(n)
out = pd.DataFrame(
    {"y": y, "t": t, "m": m, "z": z, "w1": w1, "w2": w2, "g": g, "cl": cl}
)
out[["y", "t", "m", "z", "w1"]] = out[["y", "t", "m", "z", "w1"]].round(6)
path = Path(__file__).parent / "iv_mediate.csv"
out.to_csv(path, index=False, float_format="%.6f")
print(path, out.shape)
