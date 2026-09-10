"""StatsPAI RD bandwidth selection parity (Python side) -- Module 88.

Pins ``sp.rdbwselect`` against ``rdrobust::rdbwselect`` and the Stata
``rdbwselect`` ado, both maintained by the Calonico-Cattaneo-Farrell-
Titiunik group, so this is a canonical dual reference rather than a
bridge (CLAUDE.md §5.1).

Why this module exists
----------------------
``sp.rdbwselect`` is the bandwidth a user reads *before* fitting, and it
had no cross-language evidence of any kind. Packaging it exposed two
correctness defects that no unit test caught, because nothing in the
suite compared the published function against the reference it names in
its own docstring:

1. The public entry point ran a single-step rule of thumb of its own
   rather than the CCT cascade ``sp.rdrobust`` uses. Its exponent 1/5
   equals CCT's 1/(2p+3) only at p == 1 and it produced no separate bias
   bandwidth, so it returned ``h`` 2.8x-4.8x too narrow -- 4.63 against
   R's 17.75 here -- while advertising Calonico, Cattaneo and Farrell
   (2020).
2. Inside the cascade, the four ``comb`` selectors were never
   implemented: they fell through to the plain ``rd`` form. ``comb1``
   masked this wherever ``rd`` is already the smaller of the pair, and
   ``comb2`` came out 2.8e-4 (MSE) and 5.5e-3 (CER) low.

The sweep below is deliberately wider than the default call. A selector
is a function of (method, p, q, deriv, kernel, covariates, clustering),
and defect 2 lived in a corner that the default never reaches, so a
module that pinned only ``mserd`` at ``p=1`` would have certified the
function while leaving the broken branch uncovered.

Tolerance: rel < 1e-6 on every bandwidth, both sides, all cells.
"""

from __future__ import annotations

import numpy as np
import statspai as sp

from _common import ParityRecord, dump_csv, write_results

MODULE = "88_rdbwselect"

#: Every selector ``rdrobust::rdbwselect`` offers. ``msesum`` / ``cersum``
#: were unreachable from the StatsPAI entry point until this module.
SELECTORS = (
    "mserd",
    "msetwo",
    "msesum",
    "msecomb1",
    "msecomb2",
    "cerrd",
    "certwo",
    "cersum",
    "cercomb1",
    "cercomb2",
)

QUANTITIES = ("h_left", "h_right", "b_left", "b_right")


def build_frame() -> "sp.pd.DataFrame":
    """Lee 2008 senate replica plus two deterministic auxiliary columns.

    The covariate and the cluster id are derived from the running
    variable by closed form rather than drawn, so the CSV both languages
    read is a pure function of the shipped dataset -- no seed, no RNG
    version, nothing for the two sides to disagree about before the
    estimator is even called.
    """
    df = sp.datasets.lee_2008_senate().copy()
    x = df["x"].to_numpy(float)
    # Smooth, bounded, and -- the part that matters -- well conditioned
    # against the local polynomial basis inside the selected bandwidth
    # (condition number ~22 for [1, x, z1] on |x| <= h).
    #
    # The first draft used cos(x/10), which is nearly 1 - x^2/200 near the
    # cutoff and therefore close to collinear with the local basis; the two
    # implementations' linear-solve fallbacks then parted company at 2.3e-8
    # while every other cell in this module agreed at ~1e-12. That is a
    # property of the fixture, not of the estimator, and pinning it would
    # have budgeted four orders of slack for a degenerate design. The
    # collinear case is a real (and separate) defect -- StatsPAI returns a
    # number rather than failing loudly -- tracked as O1 in
    # PARITY_SWEEP_FINDINGS.md.
    df["z1"] = np.sin(x / 3.0)
    # 20 clusters, assigned by position in the running variable so both
    # sides derive identical integer ids from identical bytes.
    df["cl"] = (np.argsort(np.argsort(x)) % 20).astype(int)
    return df


def rows_for(df, label: str, *, statistic_prefix: str, **kwargs) -> list[ParityRecord]:
    """One ParityRecord per bandwidth quantity for a single call."""
    out = sp.rdbwselect(df, y="y", x="x", c=0.0, **kwargs)
    assert len(out) == 1, f"expected a single row for {label}, got {len(out)}"
    rec = out.iloc[0]
    return [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic=f"{statistic_prefix}_{q}",
            estimate=float(rec[q]),
            n=int(len(df)),
        )
        for q in QUANTITIES
    ]


def main() -> None:
    df = build_frame()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []

    # --- Block A: all ten selectors at the p=1 default. -----------------
    # comb1/comb2 are the cells defect 2 lived in.
    for method in SELECTORS:
        rows += rows_for(df, method, statistic_prefix=method, bwselect=method)

    # --- Block B: polynomial order. -------------------------------------
    # The retired rule of thumb was numerically identical at p=1 and p=2
    # because its exponent did not depend on p. These rows make that
    # class of defect impossible to reintroduce silently.
    for p in (2, 3):
        rows += rows_for(df, f"p{p}", statistic_prefix=f"mserd_p{p}", p=p)

    # --- Block C: kernel. ------------------------------------------------
    for kern in ("uniform", "epanechnikov"):
        rows += rows_for(df, kern, statistic_prefix=f"mserd_{kern}", kernel=kern)

    # --- Block D: covariate-adjusted selection. --------------------------
    rows += rows_for(df, "covs", statistic_prefix="mserd_covs", covs=["z1"])

    # --- Block E: clustered selection. -----------------------------------
    # The cascade's V term is a sandwich, so clustering moves the
    # bandwidth itself. R additionally promotes cluster + vce='nn' to
    # 'cr1' with hc1 residuals; if StatsPAI missed that substitution the
    # two sides would part company here and nowhere else.
    rows += rows_for(df, "cluster", statistic_prefix="mserd_cluster", cluster="cl")

    # --- Block F: derivative order (regression kink). --------------------
    rows += rows_for(df, "deriv1", statistic_prefix="mserd_deriv1", deriv=1, p=2)

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "selectors": list(SELECTORS),
            "n_rows": len(rows),
            "reference": "rdrobust::rdbwselect (R) / rdbwselect (Stata)",
            "note": (
                "sp.rdbwselect delegates to rd/_cct_bandwidth.cct_bandwidth, "
                "the same three-stage CCT cascade sp.rdrobust uses, so this "
                "module pins the selector and the estimator's default-h path "
                "with one artifact."
            ),
        },
    )


if __name__ == "__main__":
    main()
