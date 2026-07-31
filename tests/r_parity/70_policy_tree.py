"""StatsPAI policy-tree parity -- Module 70 (Python side).

Pins ``sp.policy_tree`` against ``policytree::policy_tree`` (Sverdrup,
Kanodia, Zhou, Athey & Wager 2020), the reference implementation of the
Athey-Wager (2021) welfare-maximising policy-learning problem.

Shared-nuisance design
----------------------
Both engines solve

    max_{pi in Pi_d}  sum_i Gamma_i * pi(X_i)

over depth-``d`` axis-aligned trees.  The doubly-robust scores
``Gamma_i`` are computed **once**, on the Python side, and dumped into
``data/70_policy_tree.csv`` alongside the policy covariates, so the R
side reads the identical bytes and calls
``policy_tree(X, cbind(0, Gamma), depth = d, min.node.size = 1)``.
Passing ``Gamma_0 = 0`` and ``Gamma_1 = Gamma`` makes ``policytree``'s
two-action objective ``mean(Gamma[i, pi(X_i)])`` identical to StatsPAI's
``mean(Gamma * pi)``.

This decoupling is deliberate: with the nuisance step removed, any
residual gap is a pure tree-search difference, so the tolerance can sit
at the machine floor instead of absorbing cross-fitting noise.  It is
also the only honest way to grade an optimiser -- a wide band around two
differently-regularised AIPW score vectors would grade the nuisance
models, not the search.

What is compared
----------------
``value_policy`` (the objective itself), ``fraction_treated``, and the
root split's variable index and threshold, for both depth 1 and depth 2.
The full per-row policy vectors are additionally written into the JSON
``extra`` block and asserted elementwise by
``tests/reference_parity/test_policy_tree_r_parity.py``.

Registered tolerance (``compare.py``): rel_est < 1e-6 (machine tier).
Both sides maximise the same finite objective over the same finite grid
of distinct covariate values under the same ``x <= t`` split convention,
so the optimum -- and therefore every reported statistic -- agrees to
floating-point noise.

References
----------
[@athey2021policy], [@zhou2023offline]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "70_policy_tree"
N = 1200
K = 3
COVARIATES = [f"x{j + 1}" for j in range(K)]
DEPTHS = (1, 2)
MIN_NODE_SIZE = 1


def make_data(seed: int = PARITY_SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomised-treatment DGP with a genuinely two-dimensional rule.

    ``tau(X) = x1 + x2`` makes the optimal depth-2 policy split on *both*
    x1 and x2, so a search that only gets the root split right still
    fails -- exactly the discriminating power this module needs.  The
    treatment is randomised (e = 0.5), so the AIPW scores are well-posed
    and the module grades the optimiser rather than overlap.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, K))
    d = rng.integers(0, 2, size=N)
    tau = X[:, 0] + X[:, 1]
    y = 0.5 * X[:, 2] + tau * d + rng.normal(scale=0.5, size=N)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"] = d
    df["y"] = y

    # Doubly-robust (AIPW) scores under the known randomisation e = 0.5,
    # with an OLS working model for the two arms. Computed here, shared
    # with R through the CSV.
    e = 0.5
    design = np.column_stack([np.ones(N), X])
    gamma = np.empty(N)
    mu = {}
    for arm in (0, 1):
        m = d == arm
        beta, *_ = np.linalg.lstsq(design[m], y[m], rcond=None)
        mu[arm] = design @ beta
    gamma = (
        (mu[1] - mu[0])
        + d * (y - mu[1]) / e
        - (1 - d) * (y - mu[0]) / (1 - e)
    )
    df["gamma"] = gamma
    return df, gamma


def _root(tree: dict) -> tuple[float, float]:
    """Root split variable (1-based, R convention) and threshold."""
    if tree.get("type") != "split":
        return float("nan"), float("nan")
    return float(tree["feature"] + 1), float(tree["threshold"])


def main() -> None:
    df, gamma = make_data()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []
    policies: dict[str, list[int]] = {}
    for depth in DEPTHS:
        res = sp.policy_tree(
            df,
            y="y",
            treat="d",
            covariates=COVARIATES,
            policy_covariates=COVARIATES,
            max_depth=depth,
            min_leaf_size=MIN_NODE_SIZE,
            scores=gamma,
            search="exact",
        )
        assert res["search_mode"] == "exact", res["search_mode"]
        policy = np.asarray(res["policy"], dtype=int)
        policies[f"depth{depth}"] = [int(v) for v in policy]
        var, thr = _root(res["tree"]._tree)
        rows.extend(
            [
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"value_policy_d{depth}",
                    estimate=float(res["value_policy"]), n=N,
                ),
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"fraction_treated_d{depth}",
                    estimate=float(res["fraction_treated"]), n=N,
                ),
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"root_split_variable_d{depth}",
                    estimate=var, n=N,
                ),
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"root_split_value_d{depth}",
                    estimate=thr, n=N,
                ),
            ]
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "covariates": COVARIATES,
            "depths": list(DEPTHS),
            "min_node_size": MIN_NODE_SIZE,
            "search": "exact",
            "split_step": 1,
            "seed": PARITY_SEED,
            "policy": policies,
            "dgp": (
                "randomised d (e=0.5); tau(X)=x1+x2; "
                "Y=0.5*x3+tau*d+N(0,0.5^2); N=1200, K=3. AIPW scores use "
                "the known e=0.5 and per-arm OLS working models."
            ),
            "estimator": "exact depth<=2 welfare-maximising tree search",
            "note": (
                "Shared-nuisance parity: the AIPW score vector Gamma is "
                "computed once on the Python side and shipped to R inside "
                "the CSV, so both engines optimise the identical objective "
                "over the identical data. policytree consumes it as the "
                "two-action reward matrix cbind(0, Gamma), which makes "
                "mean(Gamma[i, pi(X_i)]) equal to StatsPAI's "
                "mean(Gamma * pi). Any residual gap is therefore a "
                "tree-search difference, not nuisance noise."
            ),
        },
    )


if __name__ == "__main__":
    main()
