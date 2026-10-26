"""
Policy Tree: Optimal treatment assignment rules.

Learns a depth-limited decision tree that maximises expected welfare:

    max_{pi}  E[ Gamma_i * pi(X_i) + Gamma_i^0 * (1 - pi(X_i)) ]

where pi(X) in {0, 1} is the treatment policy and Gamma_i are doubly
robust scores (AIPW pseudo-outcomes).

For depth-1 and depth-2 trees the welfare optimum is found by exhaustive
search over the complete grid of distinct covariate values
(:mod:`statspai.policy_learning._exact_tree`), matching
``policytree::policy_tree``. For depth >= 3 exact search is
combinatorially infeasible and a greedy recursive split is used; the
mode actually used is reported as ``result["search_mode"]``.

References
----------
[@athey2021policy], [@zhou2023offline]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from .._result_serialize import ResultProtocolMixin
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches policy_tree.


TreeNode = Dict[str, Any]


# ======================================================================
# Result class — promoted from a plain dict to a rich result object
# (Athey-Wager 2021; Sverdrup-Kanodia-Zhou-Athey-Wager 2020 policytree).
# ======================================================================


class PolicyTreeResult(dict, ResultProtocolMixin):
    """Result of :func:`policy_tree`.

    Inherits from :class:`dict` so the legacy ``result['policy']`` API
    keeps working (and ``isinstance(result, dict)`` is still True), while
    also exposing rich attribute access plus methods:

    * :attr:`value_policy_se` — influence-function SE of the policy value,
      computed from the AIPW scores :math:`\\Gamma_i` and the policy's
      assignments. With ``value_evaluation='cross_fit'`` (default) each
      unit is assigned by a tree grown without it, so the SE describes the
      reported value; with ``'in_sample'`` it ignores the search and the
      value is optimistic.
    * :meth:`summary` / :meth:`plot_tree` / :meth:`to_latex` / :meth:`cite`
      that match the Stata / R reporting idioms.
    * :meth:`to_excel` for publication exports.

    The ``tree`` attribute holds the fitted :class:`PolicyTree` instance
    so :meth:`PolicyTree.predict` is reachable downstream.

    Examples
    --------
    Produced by :func:`policy_tree`; the legacy ``dict`` API still works
    alongside attribute access and the rich reporting methods:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> treat = rng.integers(0, 2, n)
    >>> y = 1.0 + 2.0 * (x1 > 0) * treat + 0.5 * x2 + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})
    >>> res = sp.policy_tree(df, y="y", treat="treat",
    ...                      covariates=["x1", "x2"], max_depth=2,
    ...                      min_leaf_size=30, n_folds=3, random_state=0)
    >>> isinstance(res, sp.PolicyTreeResult)
    True
    >>> isinstance(res, dict)               # legacy result['policy'] still works
    True
    >>> res["n_obs"]
    300
    >>> bool(0.0 <= res.fraction_treated <= 1.0)
    True
    >>> "begin{table}" in res.to_latex()        # LaTeX table output
    True
    """

    def __init__(
        self,
        *,
        tree: "PolicyTree",
        policy: np.ndarray,
        value_treat_all: float,
        value_treat_none: float,
        value_policy: float,
        value_gain: float,
        fraction_treated: float,
        rules: str,
        n_obs: int,
        scores: Optional[np.ndarray] = None,
        value_policy_se: float = float("nan"),
        value_policy_ci: Tuple[float, float] = (float("nan"), float("nan")),
        value_gain_se: float = float("nan"),
        policy_covariates: Tuple[str, ...] = (),
        max_depth: int = 0,
        search_mode: str = "exact",
    ) -> None:
        super().__init__(
            tree=tree,
            policy=policy,
            value_treat_all=value_treat_all,
            value_treat_none=value_treat_none,
            value_policy=value_policy,
            value_gain=value_gain,
            fraction_treated=fraction_treated,
            rules=rules,
            n_obs=n_obs,
            scores=scores,
            value_policy_se=value_policy_se,
            value_policy_ci=value_policy_ci,
            value_gain_se=value_gain_se,
            policy_covariates=policy_covariates,
            max_depth=max_depth,
            search_mode=search_mode,
        )

    # Attribute access mirrors dict keys for ergonomic .field syntax.
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    # ------- reporting -------
    def summary(self) -> str:
        lo, hi = self.value_policy_ci
        rules = str(self["rules"])
        if self.get("multi_action"):
            vals = ", ".join(f"{k}: {v:+.4f}" for k, v in self["action_values"].items())
            shares = ", ".join(
                f"{k}: {v:.3f}" for k, v in self["action_shares"].items()
            )
            return (
                "Multi-action Policy Tree\n"
                "----------------------------------------\n"
                f"  Max depth          : {self.max_depth}\n"
                f"  Policy covariates  : {', '.join(self.policy_covariates)}\n"
                f"  Observations       : {self.n_obs:,}\n"
                f"  V(assign all to a) : {vals}\n"
                f"  Shares assigned    : {shares}\n"
                f"  V(learned policy)  : {self.value_policy:+.4f}"
                f"  (SE {self.value_policy_se:.4f}, "
                f"95% CI [{lo:+.4f}, {hi:+.4f}])\n"
                f"  Gain vs best uniform ({self['best_uniform_action']!r}) : "
                f"{self.value_gain:+.4f} (SE {self.value_gain_se:.4f})\n"
                "\nLearned policy:\n" + rules
            )
        return (
            "Policy Tree (Athey-Wager 2021)\n"
            "----------------------------------------\n"
            f"  Max depth          : {self.max_depth}\n"
            f"  Policy covariates  : {', '.join(self.policy_covariates)}\n"
            f"  Observations       : {self.n_obs:,}\n"
            f"  Fraction treated   : {self.fraction_treated:.3f}\n"
            f"  V(treat all)       : {self.value_treat_all:+.4f}\n"
            f"  V(treat none)      : {self.value_treat_none:+.4f}\n"
            f"  V(learned policy)  : {self.value_policy:+.4f}"
            f"  (SE {self.value_policy_se:.4f}, "
            f"95% CI [{lo:+.4f}, {hi:+.4f}])\n"
            f"  Value gain vs best uniform : {self.value_gain:+.4f}\n"
            "\nLearned policy:\n" + rules
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def plot_tree(
        self,
        ax: Any = None,
        figsize: Tuple[float, float] = (8.0, 5.0),
        node_color: str = "#e8f0fe",
    ) -> Tuple[Any, Any]:
        """Draw the policy tree as a labeled hierarchical diagram.

        Each split node shows ``feature ≤ threshold``; each leaf shows
        ``TREAT`` / ``DON'T TREAT`` plus the leaf value (mean AIPW score).
        Requires matplotlib.
        """
        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for plot_tree()") from e

        # Recursively assign x-positions via leaf count.
        node = self.tree._tree
        var_names = list(self.policy_covariates)

        def _count_leaves(n: TreeNode) -> int:
            if n["type"] == "leaf":
                return 1
            return _count_leaves(n["left"]) + _count_leaves(n["right"])

        total_leaves = _count_leaves(node)
        positions: List[Dict[str, Any]] = []

        def _assign(
            n: TreeNode,
            depth: int,
            x_left: float,
            x_right: float,
        ) -> None:
            x = (x_left + x_right) / 2
            entry = {"node": n, "depth": depth, "x": x}
            if n["type"] == "leaf":
                positions.append(entry)
                return
            left_leaves = _count_leaves(n["left"])
            right_leaves = _count_leaves(n["right"])
            split_x = x_left + (x_right - x_left) * (
                left_leaves / (left_leaves + right_leaves)
            )
            _assign(n["left"], depth + 1, x_left, split_x)
            _assign(n["right"], depth + 1, split_x, x_right)
            positions.append(entry)

        _assign(node, 0, 0.0, float(total_leaves))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        max_depth = max(p["depth"] for p in positions)

        def _draw(n: TreeNode, x: float, depth: int) -> None:
            y = max_depth - depth
            if n["type"] == "leaf":
                action = "TREAT" if n["action"] == 1 else "DON'T TREAT"
                color = "#a3d9a5" if n["action"] == 1 else "#f5b7b1"
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (x - 0.45, y - 0.25),
                        0.9,
                        0.5,
                        boxstyle="round,pad=0.02",
                        fc=color,
                        ec="#555",
                        lw=1.0,
                    )
                )
                ax.text(
                    x,
                    y,
                    f"{action}\nv={n['value']:+.3f}\nn={n['n']}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                return

            # split node
            feat = (
                var_names[n["feature"]]
                if n["feature"] < len(var_names)
                else f"X{n['feature']}"
            )
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - 0.55, y - 0.22),
                    1.1,
                    0.45,
                    boxstyle="round,pad=0.02",
                    fc=node_color,
                    ec="#333",
                    lw=1.0,
                )
            )
            ax.text(
                x,
                y,
                f"{feat}\n≤ {n['threshold']:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )

            # children positions: pull out next-level entries from `positions`
            left_entry = next(p for p in positions if p["node"] is n["left"])
            right_entry = next(p for p in positions if p["node"] is n["right"])
            ax.plot(
                [x, left_entry["x"]],
                [y - 0.22, max_depth - left_entry["depth"] + 0.22],
                color="#555",
                lw=1.0,
            )
            ax.plot(
                [x, right_entry["x"]],
                [y - 0.22, max_depth - right_entry["depth"] + 0.22],
                color="#555",
                lw=1.0,
            )
            _draw(n["left"], left_entry["x"], depth + 1)
            _draw(n["right"], right_entry["x"], depth + 1)

        root_entry = next(p for p in positions if p["node"] is node)
        _draw(node, root_entry["x"], 0)

        ax.set_xlim(-0.5, total_leaves + 0.5)
        ax.set_ylim(-0.5, max_depth + 0.8)
        ax.set_axis_off()
        ax.set_title(f"Policy Tree (depth ≤ {self.max_depth})", fontsize=12)
        return fig, ax

    def to_latex(
        self, caption: Optional[str] = None, label: str = "tab:policy_tree"
    ) -> str:
        """Render a publication-style summary table (LaTeX)."""
        cap = caption or "Policy Learning Results (Athey-Wager 2021)"
        lo, hi = self.value_policy_ci
        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{cap}}}\n"
            f"\\label{{{label}}}\n"
            "\\begin{tabular}{lc}\n"
            "\\hline\\hline\n"
            "Quantity & Estimate \\\\\n"
            "\\hline\n"
            f"V(treat all) & {self.value_treat_all:+.4f} \\\\\n"
            f"V(treat none) & {self.value_treat_none:+.4f} \\\\\n"
            f"V(learned policy) & {self.value_policy:+.4f} "
            f"({self.value_policy_se:.4f}) \\\\\n"
            f"\\hspace{{1em}}95\\% CI & "
            f"[{lo:+.4f}, {hi:+.4f}] \\\\\n"
            f"Value gain & {self.value_gain:+.4f} \\\\\n"
            f"Fraction treated & {self.fraction_treated:.3f} \\\\\n"
            "\\hline\n"
            f"Observations & {self.n_obs:,} \\\\\n"
            "\\hline\\hline\n"
            "\\end{tabular}\n"
            "\\begin{tablenotes}\\footnotesize\n"
            "\\item Standard errors in parentheses (influence-function "
            "SE on AIPW scores).\n"
            "\\end{tablenotes}\n"
            "\\end{table}"
        )

    def to_excel(self, path: str, digits: int = 6) -> str:
        """Write a single-sheet Excel summary.

        Six decimals to match every other ``to_excel`` in the package: a
        spreadsheet is a data-interchange target that gets sorted, charted
        and recomputed, so it keeps numeric headroom rather than the
        display precision used by the presentation exports.
        """
        df = pd.DataFrame(
            {
                "quantity": [
                    "V(treat_all)",
                    "V(treat_none)",
                    "V(policy)",
                    "V(policy)_SE",
                    "V(policy)_CI_lo",
                    "V(policy)_CI_hi",
                    "value_gain",
                    "fraction_treated",
                    "n_obs",
                ],
                "value": [
                    self.value_treat_all,
                    self.value_treat_none,
                    self.value_policy,
                    self.value_policy_se,
                    self.value_policy_ci[0],
                    self.value_policy_ci[1],
                    self.value_gain,
                    self.fraction_treated,
                    self.n_obs,
                ],
            }
        )
        with pd.ExcelWriter(path) as writer:
            df.round(digits).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame({"rules": [self.rules]}).to_excel(
                writer, sheet_name="Rules", index=False
            )
        return path

    def cite(self, format: str = "bibtex") -> str:
        return CausalResult._CITATIONS["policy_tree"]


# ======================================================================
# Public API
# ======================================================================


def policy_tree(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    policy_covariates: Optional[List[str]] = None,
    max_depth: int = 2,
    min_leaf_size: int = 25,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
    scores: Optional[np.ndarray] = None,
    search: str = "auto",
    split_step: int = 1,
    value_evaluation: str = "cross_fit",
) -> Dict[str, Any]:
    """
    Learn an optimal treatment assignment policy.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Treatment variable. A 0/1 column grows the binary treat / don't
        treat tree. Any other column (three or more arms, or two levels
        not coded 0/1) grows a multi-action tree whose leaves each assign
        one level: the reward matrix is the cross-fitted AIPW score of
        every arm (``Gamma[i, a] = mu_a + 1[A = a] (Y - mu_a) / e_a``,
        gradient-boosting nuisances, generalized propensity scores
        clipped below at 0.025 and counted in
        ``n_propensity_clipped``) unless ``scores=`` supplies it. Given
        the reward matrix the depth <= 2 search is exact and assigns every
        row the same action as R ``policytree`` 1.2.4
        (``tests/reference_parity/test_policytree_multi_action_parity.py``).
        Multi-action results add ``actions``, ``levels``,
        ``action_values`` (value of assigning everyone to each level),
        ``action_shares`` and ``best_uniform_action``; ``value_gain`` is
        against the best uniform action and ``predict`` returns levels.
    covariates : list of str
        Covariates for CATE estimation (nuisance).
    policy_covariates : list of str, optional
        Covariates for the policy tree. If None, uses `covariates`.
    max_depth : int, default 2
        Maximum depth of the policy tree.
    min_leaf_size : int, default 25
        Minimum observations in a leaf.
    n_folds : int, default 5
        Cross-fitting folds for doubly robust scores.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42
    scores : np.ndarray (n,) or (n, A), optional
        Pre-computed doubly-robust scores ``Gamma_i`` for treating unit
        ``i``, one per row of ``data``; or an ``n x A`` reward matrix, one
        column per action in the order of the sorted treatment levels,
        which grows a multi-action tree (``policytree::policy_tree(X,
        Gamma)``).  When supplied the internal
        cross-fitted AIPW step is skipped entirely, so the tree search is
        decoupled from nuisance estimation -- this is the injection point
        that makes the search directly comparable with
        ``policytree::policy_tree(X, cbind(0, Gamma))``, and it lets
        callers reuse scores from ``sp.causal_forest`` or ``sp.dml``.
    search : {'auto', 'exact', 'greedy'}, default 'auto'
        ``'exact'`` solves the depth ``<= 2`` welfare optimum by
        exhaustive search over the complete split grid.  ``'greedy'``
        uses the recursive one-step-lookahead split.  ``'auto'`` picks
        ``'exact'`` for depth ``<= 2`` unless the problem exceeds the
        cost budget, in which case it warns and falls back to greedy.
        Depth ``>= 3`` is always greedy.
    split_step : int, default 1
        Consider only every ``split_step``-th distinct value of each
        policy covariate. ``1`` searches the complete grid. An
        approximation in the spirit of ``policytree``'s ``split.step``,
        which thins by sample rather than by distinct value, so the two
        agree only at 1.
    value_evaluation : {'cross_fit', 'in_sample'}, default 'cross_fit'
        How ``value_policy`` / ``value_gain`` and their standard errors are
        computed. ``'cross_fit'`` splits the rows into ``n_folds`` folds,
        grows the tree on the other folds and evaluates its assignments
        (and the best uniform action of those folds) on the held-out rows,
        so every unit is scored by a policy that did not see it: the value
        of the learning procedure, unbiased for it, with an
        influence-function interval. That interval treats the folds as
        independent, while each fold's tree is grown on the other folds'
        scores; with real heterogeneity this is second order (95% intervals
        covered 92% at n = 400 and 97% at n = 1600 in a simulation); at an
        exact null of no heterogeneity it is not (81% at n = 400, 85% at
        n = 1600).
        ``'in_sample'`` scores the full-sample tree on the scores
        it was chosen to maximise (``policytree``'s objective); that value
        is optimistic -- with no heterogeneity at all its gain over the
        best uniform action is positive by construction -- and a warning
        says so. The returned ``policy`` / ``tree`` are always the
        full-sample fit, and ``value_policy_in_sample`` /
        ``value_gain_in_sample`` are always reported. ``'cross_fit'``
        became the default in the Oct 2026 scan; earlier releases
        reported the in-sample value with an interval that ignored the
        search.

    Returns
    -------
    dict
        'tree' : PolicyTree
            The fitted policy tree object.
        'policy' : np.ndarray
            Treatment recommendations (0 or 1) for each observation.
        'value_treat_all' : float
            Expected value if treating everyone.
        'value_treat_none' : float
            Expected value if treating no one.
        'value_policy' : float
            Expected value under the learned policy.
        'value_gain' : float
            Gain from policy vs. best uniform policy.
        'fraction_treated' : float
            Fraction recommended for treatment.
        'rules' : str
            Human-readable policy rules.
        'n_obs' : int
        'search_mode' : str
            ``'exact'`` or ``'greedy'`` -- which search actually ran.
        'value_evaluation' : str
            ``'cross_fit'`` or ``'in_sample'`` (see the parameter).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> treatment = rng.integers(0, 2, n)
    >>> outcome = 1 + 1.5 * (x1 > 0) * treatment + 0.3 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"outcome": outcome, "treatment": treatment,
    ...                    "x1": x1, "x2": x2})
    >>> result = sp.policy_tree(df, y="outcome", treat="treatment",
    ...                         covariates=["x1", "x2"], max_depth=2,
    ...                         min_leaf_size=20, n_folds=3, random_state=0)
    >>> isinstance(result, sp.PolicyTreeResult)
    True
    >>> bool(result["value_gain_in_sample"] >= 0)  # by construction
    True
    """
    est = PolicyTree(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        policy_covariates=policy_covariates,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        n_folds=n_folds,
        alpha=alpha,
        random_state=random_state,
        scores=scores,
        search=search,
        split_step=split_step,
        value_evaluation=value_evaluation,
    )
    return est.fit()


def policy_value(
    scores: np.ndarray,
    policy: np.ndarray,
) -> float:
    """
    Evaluate the value of a treatment policy relative to treating no one.

    Returns ``mean(scores * policy)``. With ``scores`` the doubly robust
    treatment-effect scores ``Gamma_1 - Gamma_0`` (grf ``get_scores``;
    policytree's reward matrix is ``[Gamma_0, Gamma_1]``), this is
    ``mean(Gamma[i, pi_i]) - mean(Gamma[i, 0])``: the AIPW estimate of
    ``E[Y(pi)] - E[Y(0)]``. It equals ``mean(policy)`` times the AIPW ATE
    of the units the policy treats (grf
    ``average_treatment_effect(subset = policy == 1)``). No standard
    error is returned; :func:`policy_tree` reports one.

    Parameters
    ----------
    scores : np.ndarray (n,)
        Doubly robust scores (AIPW pseudo-outcomes for treatment).
        Positive scores indicate the individual benefits from treatment.
        Must be one-dimensional.
    policy : np.ndarray (n,)
        Policy recommendations (0 or 1; a probability of treatment
        gives the value of the stochastic policy). Same length as
        ``scores``, or a scalar for a constant policy.

    Returns
    -------
    float
        Estimated value gain of the policy over treating no one.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> scores = rng.normal(0.3, 1.0, size=n)  # DR gains from treating

    Treat-everyone vs an oracle policy (treat only positive-gain units):

    >>> policy_all = np.ones(n, dtype=int)
    >>> policy_oracle = (scores > 0).astype(int)
    >>> round(float(sp.policy_value(scores, policy_all)), 2)
    0.29
    >>> round(float(sp.policy_value(scores, policy_oracle)), 2)
    0.54
    """
    scores = np.asarray(scores, dtype=float)
    policy = np.asarray(policy, dtype=float)
    if policy.ndim == 0:  # a constant policy (0 = treat no one, 1 = treat all)
        policy = np.full(scores.shape, float(policy))
    # A column vector (n, 1) against an (n,) policy used to broadcast to an
    # (n, n) outer product and return mean(scores) * mean(policy) silently.
    if scores.ndim != 1 or policy.ndim != 1 or scores.shape != policy.shape:
        raise MethodIncompatibility(
            "policy_value: scores and policy must be one-dimensional arrays of "
            f"the same length; got shapes {scores.shape} and {policy.shape}.",
            recovery_hint="Flatten scores and policy to 1-D arrays of equal length.",
        )
    # Scores are the gain from treatment, so the value relative to treating
    # no one is mean(scores * policy).
    return float(np.mean(scores * policy))


def _warn_in_sample_value() -> None:
    import warnings

    warnings.warn(
        "policy_tree(value_evaluation='in_sample'): the policy value and its "
        "gain are computed on the same scores the tree was chosen to "
        "maximise, so they are optimistic and their intervals do not account "
        "for the search. Use value_evaluation='cross_fit' (the default) for "
        "an honest value.",
        UserWarning,
        stacklevel=4,
    )


# ======================================================================
# Policy Tree class
# ======================================================================


class PolicyTree:
    """
    Optimal depth-limited policy tree.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    policy_covariates : list of str, optional
    max_depth : int
    min_leaf_size : int
    n_folds : int
    alpha : float
    random_state : int

    Examples
    --------
    Fit the estimator directly, then route fresh covariates through the
    learned rule with :meth:`predict`:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> treat = rng.integers(0, 2, n)
    >>> y = 1.0 + 2.0 * (x1 > 0) * treat + 0.5 * x2 + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})
    >>> tree = sp.PolicyTree(data=df, y="y", treat="treat",
    ...                      covariates=["x1", "x2"], max_depth=2,
    ...                      min_leaf_size=30, n_folds=3, random_state=0)
    >>> res = tree.fit()
    >>> bool(res["value_gain_in_sample"] >= 0)  # by construction
    True
    >>> rec = tree.predict(np.array([[1.5, 0.0], [-1.5, 0.0]]))
    >>> int(rec.shape[0])
    2
    >>> bool(set(int(v) for v in rec) <= {0, 1})
    True
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        policy_covariates: Optional[List[str]] = None,
        max_depth: int = 2,
        min_leaf_size: int = 25,
        n_folds: int = 5,
        alpha: float = 0.05,
        random_state: int = 42,
        scores: Optional[np.ndarray] = None,
        search: str = "auto",
        split_step: int = 1,
        value_evaluation: str = "cross_fit",
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.policy_covariates = policy_covariates or covariates
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.n_folds = n_folds
        self.alpha = alpha
        self.random_state = random_state
        if search not in ("auto", "exact", "greedy"):
            raise MethodIncompatibility(
                f"policy_tree: unknown search={search!r}.",
                recovery_hint="Use search='auto', 'exact', or 'greedy'.",
            )
        self.search = search
        if int(split_step) < 1:
            raise MethodIncompatibility(
                f"policy_tree: split_step must be >= 1, got {split_step!r}.",
                recovery_hint="Use split_step=1 to search the complete grid.",
            )
        self.split_step = int(split_step)
        if value_evaluation not in ("cross_fit", "in_sample"):
            raise MethodIncompatibility(
                f"policy_tree: unknown value_evaluation={value_evaluation!r}.",
                recovery_hint="Use value_evaluation='cross_fit' (honest) or "
                "'in_sample'.",
            )
        if value_evaluation == "cross_fit" and int(n_folds) < 2:
            raise MethodIncompatibility(
                "policy_tree: value_evaluation='cross_fit' needs n_folds >= 2.",
                recovery_hint="Raise n_folds, or pass value_evaluation='in_sample'.",
            )
        self.value_evaluation = value_evaluation
        self.scores = None if scores is None else np.asarray(scores, dtype=np.float64)

    def fit(self) -> Dict[str, Any]:
        """Learn the optimal policy tree."""
        # Prepare data
        all_cols = list(
            set([self.y, self.treat] + self.covariates + self.policy_covariates)
        )
        check_columns(
            self.data,
            {
                "y": self.y,
                "treat": self.treat,
                "covariates": list(self.covariates),
                "policy_covariates": (
                    None
                    if self.policy_covariates is self.covariates
                    else list(self.policy_covariates)
                ),
            },
            function="policy_tree",
        )

        clean = self.data[all_cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        W = clean[self.covariates].values.astype(np.float64)
        X_pol = clean[self.policy_covariates].values.astype(np.float64)
        n = len(Y)

        t = clean[self.treat]
        levels = sorted(t.unique().tolist(), key=lambda v: (str(type(v)), v))
        binary01 = len(levels) == 2 and set(levels) <= {0, 1, 0.0, 1.0, True, False}
        if not binary01 or (self.scores is not None and self.scores.ndim == 2):
            return self._fit_multi_action(clean, all_cols, Y, t, levels, W, X_pol)
        D = t.values.astype(np.float64)

        # Step 1: doubly robust scores -- caller-supplied, or cross-fitted.
        if self.scores is not None:
            if self.scores.shape[0] != len(self.data):
                raise MethodIncompatibility(
                    "policy_tree: len(scores) must match len(data) "
                    f"({self.scores.shape[0]} vs {len(self.data)}).",
                    recovery_hint=(
                        "Pass one doubly-robust score per input row; rows with "
                        "missing model columns are dropped afterwards."
                    ),
                )
            keep = self.data[all_cols].notna().all(axis=1).to_numpy()
            scores = np.asarray(self.scores[keep], dtype=np.float64)
        else:
            scores = self._compute_dr_scores(Y, D, W, n)

        # Step 2: Grow policy tree on scores. Depth <= 2 admits an exact
        # welfare optimum; deeper trees fall back to the greedy split.
        tree, search_mode = self._build_tree(X_pol, scores)

        # Step 3: Generate policy
        policy_decisions = self._predict_tree(tree, X_pol)

        # Step 4: Evaluate
        value_policy = float(np.mean(scores * policy_decisions))
        value_all = float(np.mean(scores))
        value_none = 0.0

        fraction_treated = float(np.mean(policy_decisions))

        # Generate rules
        rules = self._tree_to_rules(tree, self.policy_covariates)

        self._tree = tree
        self._scores = scores
        self._n_policy_features = X_pol.shape[1]

        # Influence-function SE on the policy value:
        # V̂(π̂) = (1/n) Σ Γ_i π̂(X_i)  →  IF_i = Γ_i π̂(X_i) - V̂.
        # The same construction yields an SE on V_gain by replacing the
        # contrast policy with the best uniform policy.
        contrib_pol = scores * policy_decisions
        v_se = (
            float(np.std(contrib_pol, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        )
        from scipy import stats as _stats

        z = float(_stats.norm.ppf(1 - self.alpha / 2))
        v_ci = (value_policy - z * v_se, value_policy + z * v_se)

        contrast = (
            np.zeros_like(scores) if value_all <= value_none else np.ones_like(scores)
        )
        contrib_gain = scores * (policy_decisions - contrast)
        gain_se = (
            float(np.std(contrib_gain, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        )
        value_gain = value_policy - max(value_all, value_none)
        in_sample = {"value_policy": value_policy, "value_gain": value_gain}
        if self.value_evaluation == "cross_fit":
            from ._multi_action import honest_value_summary

            hv = honest_value_summary(
                lambda Xf, Rf: self._build_tree(Xf, Rf[:, 1])[0],
                self._predict_tree,
                X_pol,
                np.column_stack([np.zeros(n), scores]),
                int(self.n_folds),
                self.random_state,
                self.alpha,
            )
            value_policy, v_se, v_ci = (
                hv["value_policy"],
                hv["value_policy_se"],
                hv["value_policy_ci"],
            )
            value_gain, gain_se = hv["value_gain"], hv["value_gain_se"]
        else:
            _warn_in_sample_value()
        res = PolicyTreeResult(
            tree=self,
            policy=policy_decisions,
            value_treat_all=value_all,
            value_treat_none=value_none,
            value_policy=value_policy,
            value_gain=value_gain,
            fraction_treated=fraction_treated,
            rules=rules,
            n_obs=n,
            scores=scores,
            value_policy_se=v_se,
            value_policy_ci=v_ci,
            value_gain_se=gain_se,
            policy_covariates=tuple(self.policy_covariates),
            max_depth=self.max_depth,
            search_mode=search_mode,
        )
        res.update(
            value_evaluation=self.value_evaluation,
            value_policy_in_sample=in_sample["value_policy"],
            value_gain_in_sample=in_sample["value_gain"],
        )
        return res

    def _fit_multi_action(
        self,
        clean: pd.DataFrame,
        all_cols: List[str],
        Y: np.ndarray,
        t: pd.Series,
        levels: List[Any],
        W: np.ndarray,
        X_pol: np.ndarray,
    ) -> "PolicyTreeResult":
        """Multi-action tree on an ``n x A`` reward matrix (see
        :mod:`._multi_action`)."""
        from ._multi_action import (
            check_reward_matrix,
            fit_multi_action,
            multi_arm_dr_scores,
        )

        if len(levels) < 2:
            raise MethodIncompatibility(
                "policy_tree: the treatment has a single level.",
                recovery_hint="A policy needs at least two actions.",
            )
        code = {lv: k for k, lv in enumerate(levels)}
        G = np.array([code[v] for v in t.tolist()])
        extra: Dict[str, Any] = {}
        if self.scores is not None:
            R_in = np.asarray(self.scores, dtype=np.float64)
            if R_in.ndim != 2:
                raise MethodIncompatibility(
                    "policy_tree: with a treatment of more than two levels (or "
                    "not coded 0/1), scores= must be the n x A reward matrix.",
                    recovery_hint="Pass one column of doubly robust scores per "
                    "action, ordered as the sorted treatment levels.",
                )
            keep = self.data[all_cols].notna().all(axis=1).to_numpy()
            R = check_reward_matrix(R_in, len(self.data), len(levels))[keep]
            extra["score_source"] = "user"
        else:
            R, info = multi_arm_dr_scores(
                Y, G, W, len(levels), self.n_folds, self.random_state
            )
            extra.update(info)
            extra["score_source"] = "cross-fitted AIPW (gradient boosting)"
        out = fit_multi_action(self, X_pol, R, list(levels), extra)
        in_sample = {
            "value_policy": out["value_policy"],
            "value_gain": out["value_gain"],
        }
        if self.value_evaluation == "cross_fit":
            from ._multi_action import (
                grow_multi_action_tree,
                honest_value_summary,
                predict_actions,
            )

            hv = honest_value_summary(
                lambda Xf, Rf: grow_multi_action_tree(self, Xf, Rf)[0],
                predict_actions,
                X_pol,
                R,
                int(self.n_folds),
                self.random_state,
                self.alpha,
            )
            for key in (
                "value_policy",
                "value_policy_se",
                "value_policy_ci",
                "value_gain",
                "value_gain_se",
            ):
                out[key] = hv[key]
        else:
            _warn_in_sample_value()
        self._tree = out["tree"]
        self._scores = R
        self._n_policy_features = X_pol.shape[1]
        self._multi_levels = list(levels)
        av = out["action_values"]
        best = out["best_uniform_action"]
        res = PolicyTreeResult(
            tree=self,
            policy=out["policy"],
            value_treat_all=float(av[best]),
            value_treat_none=float(av[0]),
            value_policy=out["value_policy"],
            value_gain=out["value_gain"],
            fraction_treated=float(np.mean(out["actions"] != 0)),
            rules=out["rules"],
            n_obs=len(Y),
            scores=R,
            value_policy_se=out["value_policy_se"],
            value_policy_ci=out["value_policy_ci"],
            value_gain_se=out["value_gain_se"],
            policy_covariates=tuple(self.policy_covariates),
            max_depth=self.max_depth,
            search_mode=out["search_mode"],
        )
        res.update(
            multi_action=True,
            levels=list(levels),
            actions=out["actions"],
            action_values={str(lv): float(v) for lv, v in zip(levels, av)},
            action_shares=out["action_shares"],
            best_uniform_action=levels[best],
            value_evaluation=self.value_evaluation,
            value_policy_in_sample=in_sample["value_policy"],
            value_gain_in_sample=in_sample["value_gain"],
            **extra,
        )
        return res

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predict treatment assignment for new data.

        Parameters
        ----------
        X_new : np.ndarray (n, p)
            Policy covariates for new observations.

        Returns
        -------
        np.ndarray (n,)
            Binary treatment recommendations (0 or 1).
        """
        if not hasattr(self, "_tree"):
            raise MethodIncompatibility(
                "PolicyTree.predict() requires a fitted policy tree.",
                recovery_hint="Call fit() before predict().",
            )
        try:
            X_new = np.asarray(X_new, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "PolicyTree.predict() requires numeric policy covariates.",
                recovery_hint="Convert prediction covariates to numeric values.",
            ) from exc
        expected = getattr(self, "_n_policy_features", None)
        if X_new.ndim == 1:
            if expected is not None and expected > 1 and X_new.size == expected:
                X_new = X_new.reshape(1, -1)
            elif expected in (None, 1):
                X_new = X_new.reshape(-1, 1)
            else:
                raise MethodIncompatibility(
                    "PolicyTree.predict(): one-dimensional input does not "
                    "match the fitted policy feature count.",
                    recovery_hint="Pass X_new shaped (n_samples, n_policy_features).",
                    diagnostics={
                        "n_values": int(X_new.size),
                        "expected_features": expected,
                    },
                )
        elif X_new.ndim != 2:
            raise MethodIncompatibility(
                "PolicyTree.predict(): X_new must be a 1D or 2D array.",
                recovery_hint="Pass X_new shaped (n_samples, n_policy_features).",
                diagnostics={"x_ndim": int(X_new.ndim)},
            )
        if X_new.shape[0] == 0:
            raise DataInsufficient(
                "PolicyTree.predict() received no rows.",
                recovery_hint="Pass at least one prediction row.",
            )
        if expected is not None and X_new.shape[1] != expected:
            raise MethodIncompatibility(
                "PolicyTree.predict(): feature count does not match fit().",
                recovery_hint=(
                    "Use the same policy covariates and order used when fitting."
                ),
                diagnostics={
                    "expected_features": expected,
                    "observed_features": int(X_new.shape[1]),
                },
            )
        if not np.isfinite(X_new).all():
            raise MethodIncompatibility(
                "PolicyTree.predict(): X_new contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite policy covariate rows.",
            )
        if getattr(self, "_multi_levels", None) is not None:
            from ._multi_action import predict_actions

            labels = np.asarray(self._multi_levels, dtype=object)
            return labels[predict_actions(self._tree, X_new)]
        return self._predict_tree(self._tree, X_new)

    def _compute_dr_scores(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        W: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Compute AIPW doubly robust scores for treatment benefit."""
        from sklearn import ensemble
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        mu1_hat = np.zeros(n)
        mu0_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, test_idx in kf.split(W):
            W_tr, D_tr, Y_tr = W[train_idx], D[train_idx], Y[train_idx]
            W_te = W[test_idx]

            # Outcome models
            mask1 = D_tr == 1
            mask0 = D_tr == 0

            m1 = ensemble.GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )
            m0 = ensemble.GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )

            if mask1.sum() > 0:
                m1.fit(W_tr[mask1], Y_tr[mask1])
                mu1_hat[test_idx] = m1.predict(W_te)
            if mask0.sum() > 0:
                m0.fit(W_tr[mask0], Y_tr[mask0])
                mu0_hat[test_idx] = m0.predict(W_te)

            # Propensity
            prop = ensemble.GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )
            prop.fit(W_tr, D_tr)
            e_hat[test_idx] = np.clip(prop.predict_proba(W_te)[:, 1], 0.025, 0.975)

        # AIPW score for treatment benefit
        scores = (
            (mu1_hat - mu0_hat)
            + D * (Y - mu1_hat) / e_hat
            - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        )

        return np.asarray(scores, dtype=float)

    def _build_tree(
        self,
        X: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[TreeNode, str]:
        """Dispatch to the exact or the greedy search.

        Depth <= 2 has a tractable welfare optimum, so it is solved
        exactly by default (matching ``policytree::policy_tree``).  Depth
        >= 3 is combinatorially infeasible and uses the greedy recursive
        split.  ``search="auto"`` also declines the exact route when the
        problem exceeds the cost budget -- with a warning, never silently.
        """
        from ._exact_tree import (
            exact_policy_tree,
            exact_search_is_affordable,
            warn_greedy_fallback,
        )

        n, p = X.shape
        if self.search == "greedy" or self.max_depth > 2:
            return self._grow_tree(X, scores, depth=0), "greedy"
        if self.search == "auto" and not exact_search_is_affordable(
            n, p, self.max_depth
        ):
            warn_greedy_fallback(n, p, self.max_depth)
            return self._grow_tree(X, scores, depth=0), "greedy"
        tree = exact_policy_tree(
            X,
            scores,
            max_depth=self.max_depth,
            min_leaf_size=self.min_leaf_size,
            split_step=self.split_step,
        )
        return tree, "exact"

    def _grow_tree(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """
        Recursively grow the policy tree.

        Returns a nested dict representing the tree structure.
        """
        n = len(scores)

        # Leaf node: assign treatment if mean score > 0
        if depth >= self.max_depth or n < 2 * self.min_leaf_size:
            return {
                "type": "leaf",
                "action": 1 if np.mean(scores) > 0 else 0,
                "value": float(np.mean(scores)),
                "n": n,
            }

        best_gain = -np.inf
        best_split = None

        n_features = X.shape[1]

        for j in range(n_features):
            # Complete candidate grid: every distinct value that leaves a
            # strictly larger value on the right, thinned only by the
            # explicit ``split_step`` knob. (Before v1.21 this path
            # silently subsampled to <= 50 quantiles per feature, so it
            # did not return the greedy optimum over the full grid.)
            vals = np.unique(X[:, j])
            if len(vals) < 2:
                continue
            thresholds = vals[: len(vals) - 1 : self.split_step]

            for threshold in thresholds:
                left_mask = X[:, j] <= threshold
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_leaf_size or n_right < self.min_leaf_size:
                    continue

                # Value of optimal assignment in each child
                left_scores = scores[left_mask]
                right_scores = scores[right_mask]

                # Best action in each leaf
                left_val = max(np.mean(left_scores), 0)
                right_val = max(np.mean(right_scores), 0)

                # Total gain
                gain = (n_left * left_val + n_right * right_val) / n

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": j,
                        "threshold": float(threshold),
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }

        # If no valid split found, return leaf
        if best_split is None:
            return {
                "type": "leaf",
                "action": 1 if np.mean(scores) > 0 else 0,
                "value": float(np.mean(scores)),
                "n": n,
            }

        # Recurse
        left_child = self._grow_tree(
            X[best_split["left_mask"]],
            scores[best_split["left_mask"]],
            depth + 1,
        )
        right_child = self._grow_tree(
            X[best_split["right_mask"]],
            scores[best_split["right_mask"]],
            depth + 1,
        )

        return {
            "type": "split",
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_child,
            "right": right_child,
            "n": n,
        }

    def _predict_tree(self, tree: TreeNode, X: np.ndarray) -> np.ndarray:
        """Predict treatment assignments from a tree."""
        n = X.shape[0]
        policy = np.zeros(n, dtype=int)

        for i in range(n):
            node = tree
            while node["type"] == "split":
                if X[i, node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            policy[i] = node["action"]

        return policy

    def _tree_to_rules(
        self,
        tree: TreeNode,
        var_names: List[str],
        prefix: str = "",
    ) -> str:
        """Convert tree to human-readable rules."""
        if tree["type"] == "leaf":
            action = "TREAT" if tree["action"] == 1 else "DON'T TREAT"
            return (
                f"{prefix}{action} "
                f"(n={tree['n']}, avg_benefit={tree['value']:.4f})\n"
            )

        feat = var_names[tree["feature"]]
        thresh = tree["threshold"]

        rules = ""
        rules += f"{prefix}IF {feat} <= {thresh:.4f}:\n"
        rules += self._tree_to_rules(tree["left"], var_names, prefix + "  ")
        rules += f"{prefix}ELSE ({feat} > {thresh:.4f}):\n"
        rules += self._tree_to_rules(tree["right"], var_names, prefix + "  ")
        return rules


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS["policy_tree"] = (
    "@article{athey2021policy,\n"
    "  title={Policy Learning with Observational Data},\n"
    "  author={Athey, Susan and Wager, Stefan},\n"
    "  journal={Econometrica},\n"
    "  volume={89},\n"
    "  number={1},\n"
    "  pages={133--161},\n"
    "  year={2021},\n"
    "  publisher={Wiley}\n"
    "}"
)
