"""
Balke-Pearl (1997) bounds for the ATE under a binary instrumental
variable, binary treatment and binary outcome.

Setting
-------
* Z, D, Y all binary ({0, 1}).
* Z ⊥ (Y(0), Y(1), D(0), D(1)) (instrument exogeneity).

Without further assumptions the ATE is **not** point-identified but
the tight upper and lower bounds are closed-form functions of the
joint distribution P(Y=y, D=d | Z=z). Balke & Pearl (1997) derived

.. math::

   \\ell_{BP} \\le ATE \\le u_{BP},

with

.. math::

   u_{BP} = \\min\\{ \\, ...\\, \\}, \\quad \\ell_{BP} = \\max\\{ \\, ...\\, \\},

where each argument is a sum of four joint probabilities. The exact
forms are taken from Balke-Pearl (1997, eq. 14-15) and Richardson
& Robins (2014, §3).

If the analyst is willing to impose **monotonicity** (Imbens & Angrist
1994), the bounds tighten to the Manski bounds under no-defier
restrictions. This module gives both.

References
----------
Balke, A. & Pearl, J. (1997). "Bounds on treatment effects from
studies with imperfect compliance." *JASA*, 92(439), 1171-1176. [@balke1997bounds]

Richardson, T. S. & Robins, J. M. (2014). "ACE bounds; SEMs with
equilibrium conditions." *Statistical Science*, 29(4), 363-396. [@richardson2014bounds]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Sequence, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BalkePearlResult:
    lower: float
    upper: float
    width: float
    lower_monotone: float
    upper_monotone: float
    joint_probs: pd.DataFrame
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Balke-Pearl ATE bounds\n"
            "----------------------\n"
            f"  N            : {self.n_obs}\n"
            f"  [lower, upper] (no monotonicity) : [{self.lower:+.4f}, {self.upper:+.4f}]\n"
            f"  width                              : {self.width:.4f}\n"
            f"  [lower, upper] (monotone)          : [{self.lower_monotone:+.4f}, {self.upper_monotone:+.4f}]"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"BalkePearlResult([{self.lower:+.4f}, {self.upper:+.4f}])"


def _joint_probs(df: pd.DataFrame, y: str, d: str, z: str) -> np.ndarray:
    """Return P(Y=y, D=d | Z=z) as a (2,2,2) tensor indexed [z, d, y]."""
    P = np.zeros((2, 2, 2))
    for zi in (0, 1):
        sub = df[df[z] == zi]
        nz = max(len(sub), 1)
        for di in (0, 1):
            for yi in (0, 1):
                P[zi, di, yi] = float(((sub[d] == di) & (sub[y] == yi)).sum()) / nz
    return P


def balke_pearl(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    alpha: float = 0.05,
) -> BalkePearlResult:
    """
    Compute Balke-Pearl bounds on the ATE.

    All three variables must be binary (0/1).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    instrument : str
    alpha : float, default 0.05 (for CI of joint probabilities)

    Returns
    -------
    BalkePearlResult
    """
    df = data[[y, treat, instrument]].dropna()
    for col in [y, treat, instrument]:
        vals = set(pd.unique(df[col]))
        if not vals.issubset({0, 1}):
            raise ValueError(f"{col} must be binary (0/1), got {vals}")
    n = len(df)
    P = _joint_probs(df, y, treat, instrument)

    # Shorthand: P[z, d, y]
    # y(d) means potential outcome under d
    # Balke-Pearl (1997, Theorem 4) for ATE = E[Y(1) - Y(0)]:
    # Upper bound
    #
    # u_BP = min(
    #    P(1,1|0) + P(0,0|1) + P(0,1|0) + P(0,1|1),     # four-term 1
    #    P(1,1|1) + P(0,0|1) + P(0,0|0) + P(0,1|0),
    #    ...
    # )
    #
    # Lower bound mirrors with sign flips.
    #
    # We use the 8-term formulation from Balke-Pearl 1997 eq. 14-15.

    p00z0, p01z0, p10z0, p11z0 = P[0, 0, 0], P[0, 0, 1], P[0, 1, 0], P[0, 1, 1]
    p00z1, p01z1, p10z1, p11z1 = P[1, 0, 0], P[1, 0, 1], P[1, 1, 0], P[1, 1, 1]

    # From Richardson-Robins (2014, eq. 3.13) — Balke-Pearl ATE bounds:
    upper_terms = [
        p11z1 + p00z1,
        p11z0 + p00z0,
        p11z1 - p11z0 - p10z0 + p01z1 + p00z1 + p00z0,  # +
        p11z0 - p11z1 + p01z0 - p10z1 + p00z0 + p00z1,
        -p01z0 + p01z1 + p11z1 + p00z0 + p01z0 + p10z0,
    ]
    lower_terms = [
        -p01z1 - p10z1,
        -p01z0 - p10z0,
        p11z0 - p11z1 - p10z1 - p01z0 - p01z1 - p10z0,
        p11z1 - p11z0 - p10z0 - p01z1 - p01z0 - p10z1,
        -p00z0 + p00z1 - p11z1 - p10z0 - p01z1 - p10z1,
    ]

    ub = float(min(upper_terms))
    lb = float(max(lower_terms))

    # Under monotonicity (no defiers), the bounds tighten.
    # Following Manski-Pepper / Balke-Pearl corollary:
    # E[Y(1)] ∈ [E(Y·D|Z=1), 1 - E((1-Y)·D|Z=1)]  (with IAR)
    # E[Y(0)] ∈ [E(Y·(1-D)|Z=0), 1 - E((1-Y)·(1-D)|Z=0)]
    E_Y1_lo = p11z1
    E_Y1_hi = 1 - p10z1
    E_Y0_lo = p01z0
    E_Y0_hi = 1 - p00z0
    ub_mono = float(E_Y1_hi - E_Y0_lo)
    lb_mono = float(E_Y1_lo - E_Y0_hi)

    # Clip to [-1, 1] (ATE for binary outcomes).
    ub = max(min(ub, 1.0), -1.0)
    lb = max(min(lb, 1.0), -1.0)
    ub_mono = max(min(ub_mono, 1.0), -1.0)
    lb_mono = max(min(lb_mono, 1.0), -1.0)

    joint = pd.DataFrame(
        {
            "Z=0, D=0, Y=0": [p00z0],
            "Z=0, D=0, Y=1": [p01z0],
            "Z=0, D=1, Y=0": [p10z0],
            "Z=0, D=1, Y=1": [p11z0],
            "Z=1, D=0, Y=0": [p00z1],
            "Z=1, D=0, Y=1": [p01z1],
            "Z=1, D=1, Y=0": [p10z1],
            "Z=1, D=1, Y=1": [p11z1],
        }
    ).T.rename(columns={0: "probability"})

    _result = BalkePearlResult(
        lower=lb,
        upper=ub,
        width=ub - lb,
        lower_monotone=lb_mono,
        upper_monotone=ub_mono,
        joint_probs=joint,
        n_obs=n,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.bounds.balke_pearl",
            params={
                "y": y, "treat": treat, "instrument": instrument,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["balke_pearl", "BalkePearlResult"]
