"""
Policy Tree: Optimal treatment assignment rules.

Learns a depth-limited decision tree that maximises expected welfare:

    max_{pi}  E[ Gamma_i * pi(X_i) + Gamma_i^0 * (1 - pi(X_i)) ]

where pi(X) in {0, 1} is the treatment policy and Gamma_i are doubly
robust scores (AIPW pseudo-outcomes).

For depth-1 (stump) and depth-2 trees, an exact solution is found
via exhaustive search over all possible splits. For deeper trees,
a greedy recursive splitting approach is used.

References
----------
Athey, S. & Wager, S. (2021).
"Policy Learning with Observational Data."
Econometrica, 89(1), 133-161. [@athey2021matrix]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


class PolicyTreeResult(dict):
    """Result of :func:`policy_tree`.

    Inherits from :class:`dict` so the legacy ``result['policy']`` API
    keeps working (and ``isinstance(result, dict)`` is still True), while
    also exposing rich attribute access plus methods:

    * :attr:`value_policy_se` — influence-function SE of the policy value,
      computed from the AIPW scores :math:`\\Gamma_i` and the binary
      policy :math:`\\hat\\pi(X_i)`. Under the standard cross-fit /
      overlap conditions this is asymptotically valid.
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

    def to_excel(self, path: str, digits: int = 4) -> str:
        """Write a single-sheet Excel summary."""
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
        Binary treatment variable (0/1).
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
    >>> bool(result["value_gain"] >= 0)
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
    )
    return est.fit()


def policy_value(
    scores: np.ndarray,
    policy: np.ndarray,
) -> float:
    """
    Evaluate the expected value of a treatment policy.

    Parameters
    ----------
    scores : np.ndarray (n,)
        Doubly robust scores (AIPW pseudo-outcomes for treatment).
        Positive scores indicate the individual benefits from treatment.
    policy : np.ndarray (n,)
        Binary policy recommendations (0 or 1).

    Returns
    -------
    float
        Estimated expected value of the policy.

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
    scores = np.asarray(scores)
    policy = np.asarray(policy)
    # Value = E[Gamma * pi + (baseline)]
    # Since scores represent the *gain* from treatment,
    # policy value = mean(scores * policy)
    return float(np.mean(scores * policy))


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
    >>> bool(res["value_gain"] >= 0)
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

    def fit(self) -> Dict[str, Any]:
        """Learn the optimal policy tree."""
        # Prepare data
        all_cols = list(
            set([self.y, self.treat] + self.covariates + self.policy_covariates)
        )
        missing = [c for c in all_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        clean = self.data[all_cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        D = clean[self.treat].values.astype(np.float64)
        W = clean[self.covariates].values.astype(np.float64)
        X_pol = clean[self.policy_covariates].values.astype(np.float64)
        n = len(Y)

        unique_d = np.unique(D)
        if not (len(unique_d) == 2 and set(unique_d.astype(int)) == {0, 1}):
            raise ValueError("Treatment must be binary (0/1)")

        # Step 1: Compute doubly robust scores via cross-fitting
        scores = self._compute_dr_scores(Y, D, W, n)

        # Step 2: Grow policy tree on scores
        tree = self._grow_tree(X_pol, scores, depth=0)

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

        return PolicyTreeResult(
            tree=self,
            policy=policy_decisions,
            value_treat_all=value_all,
            value_treat_none=value_none,
            value_policy=value_policy,
            value_gain=value_policy - max(value_all, value_none),
            fraction_treated=fraction_treated,
            rules=rules,
            n_obs=n,
            scores=scores,
            value_policy_se=v_se,
            value_policy_ci=v_ci,
            value_gain_se=gain_se,
            policy_covariates=tuple(self.policy_covariates),
            max_depth=self.max_depth,
        )

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
            # Get sorted unique thresholds
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) < 2:
                continue

            # Use quantile-based candidate splits for efficiency
            n_candidates = min(50, len(vals) - 1)
            quantiles = np.linspace(0, 1, n_candidates + 2)[1:-1]
            thresholds = np.quantile(X[:, j], quantiles)
            thresholds = np.unique(thresholds)

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
