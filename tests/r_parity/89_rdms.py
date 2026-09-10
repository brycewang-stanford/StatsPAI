"""StatsPAI multi-score RD parity (Python side) -- Module 89.

Pins ``sp.rdms`` against ``rdmulti::rdms`` (R) and the ``rdms`` ado
(Stata), both maintained by the Cattaneo group, so this is a canonical
dual reference rather than a bridge (CLAUDE.md §5.1).

Why this module exists
----------------------
``sp.rdms`` had no cross-language evidence and, measured against the
reference it named in its own docstring, was not computing the same
estimator at all. It fitted a two-dimensional local linear inside a
Euclidean window whose width came from Silverman's *kernel density* rule
of thumb -- a category error, not a tuning choice -- with a
homoskedastic variance and no bias correction. On the design below it
admitted 8 to 27 observations where ``rdmulti::rdms`` used 519 to 743,
and returned 4.804 at the middle boundary point against the reference's
1.778, with a standard error of 4.21 against 0.13.

The reference collapses the two scores onto one::

    d = sqrt((X - C)^2 + (X2 - C2)^2) * (2 * zvar - 1)

-- Euclidean distance to the boundary point, signed by treatment status
-- and runs an ordinary sharp RD at ``d = 0`` on it. ``sp.rdms`` now does
the same and delegates to ``sp.rdrobust``, so it inherits the CCT
cascade and robust bias correction rather than re-deriving them.

The sweep pins three boundary points, and at each one both the
bias-corrected and the conventional estimate, the robust standard error,
the selected bandwidth and the effective sample size on each side. The
effective sample sizes are the row that would have caught the original
defect on its own: they are integers, so they cannot be argued into
agreement by a tolerance.

Tolerance: rel < 1e-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "89_rdms"

#: Boundary points, given as the ``x2`` coordinate along a vertical
#: boundary at ``x1 = 0``. Three points rather than one so the module
#: exercises a boundary rather than a single cutoff, which is the whole
#: point of the multi-score design.
BOUNDARY_X2 = (-0.5, 0.0, 0.5)


def build_frame() -> pd.DataFrame:
    """A geographic-RD replica with a vertical boundary at ``x1 = 0``.

    ``x2`` runs *along* the boundary and the effect varies with position
    on it (``tau(x2) = 2 + x2``), so the three boundary points have
    genuinely different true effects -- a design where a boundary-point
    estimator that silently pooled across the boundary would be caught.

    Drawn once from a fixed seed and then frozen as CSV bytes, which is
    what both other languages read.
    """
    rng = np.random.default_rng(PARITY_SEED)
    n = 6000
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    treat = (x1 >= 0.0).astype(float)
    tau = 2.0 + 1.0 * x2
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + tau * treat + rng.normal(0.0, 0.4, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": treat})


def main() -> None:
    df = build_frame()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []
    for c2 in BOUNDARY_X2:
        tag = f"b{c2:+.1f}".replace("+", "p").replace("-", "m").replace(".", "")
        res = sp.rdms(df, y="y", x1="x1", x2="x2", cutoff1=0.0, cutoff2=c2, treat="z")
        mi = res.model_info
        h = mi["bandwidth_h"]
        h = float(h["left"] if isinstance(h, dict) else h)

        def rec(stat: str, est: float, se: float | None = None) -> ParityRecord:
            return ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{tag}_{stat}",
                estimate=float(est),
                se=None if se is None else float(se),
                n=int(len(df)),
            )

        rows.append(rec("biascorrected_est", res.estimate, res.se))
        rows.append(
            rec(
                "conventional_est",
                mi["conventional"]["estimate"],
                mi["conventional"]["se"],
            )
        )
        rows.append(rec("bandwidth_h", h))
        rows.append(rec("n_eff_left", mi["n_effective_left"]))
        rows.append(rec("n_eff_right", mi["n_effective_right"]))

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "boundary_points": [f"(0, {c2})" for c2 in BOUNDARY_X2],
            "n_rows": len(rows),
            "reference": "rdmulti::rdms (R) / rdms (Stata, rdpackages GitHub)",
            "score": "signed Euclidean distance to the boundary point",
            "note": (
                "sp.rdms delegates to sp.rdrobust on the signed-distance "
                "score, which is the construction rdmulti::rdms uses. The "
                "reported point estimate is the bias-corrected one paired "
                "with the robust standard error, matching R's Estimate[2] / "
                "ci[3, ] convention; the conventional pair is pinned "
                "separately rather than folded in."
            ),
        },
    )


if __name__ == "__main__":
    main()
