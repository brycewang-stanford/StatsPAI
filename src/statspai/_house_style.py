"""Canonical public-API parameter vocabulary — the StatsPAI house style.

This module is the single source of truth for how inference / data / design
keyword arguments should be *spelled* across the public API.  Two consumers
read it:

* ``scripts/signature_house_style.py`` — the CI lint that flags public
  functions whose parameters use a non-canonical spelling of a known theme.
* ``statspai._aliases.accepts_aliases`` (grammar-convergence step) — the
  decorator that lets estimators accept legacy spellings while the canonical
  one becomes the documented default.

Design notes
------------
* The vocabulary is **additive and reversible** (CLAUDE.md §3 + the JSS-review
  constraint): naming a spelling "legacy" here does *not* change any runtime
  behaviour on its own.  Convergence happens later, opt-in, via the alias
  decorator — with deprecation warnings held off during JSS review.
* ``FALSE_FRIENDS`` is the critical noise-suppression layer.  Several params
  *look* like an inference/weight keyword but mean something orthogonal:

  - ``w`` / ``W`` — spatial weight matrix in spatial econometrics, and the
    EconML-mirrored treatment symbol in the CATE family.
  - ``cov_type`` — the random-effects covariance *structure* in mixed models
    (``'unstructured'`` / ``'independent'``), **not** an SE type.
  - ``group`` — a comparison/cohort group in decompositions and DiD, **not**
    a clustering variable.  (Hence ``group`` is deliberately *not* a cluster
    alias below.)
  - ``Y`` / ``T`` — EconML-mirrored outcome/treatment in ``causal_forest`` &
    friends, kept verbatim so EconML users migrate without relearning.

  Anything listed here is reported separately as "acknowledged" and never
  counted as a violation, or the lint becomes noise and gets ignored.

The canonical choices below were ratified by the maintainer (2026-06-30):
SE keyword ``vce=`` (Stata muscle memory), outcome ``y``, treatment ``treat``.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: SE / variance keyword.  Stata's ``vce()`` is the universality benchmark the
#: reviewer invoked, and it matches the muscle memory of migrating Stata users.
CANONICAL_SE = "vce"

#: Outcome variable.  ``y`` dominates the existing surface (313 uses) and is
#: the terse terminal-friendly default.
CANONICAL_OUTCOME = "y"

#: Treatment variable.  ``treat`` dominates (98 uses) over ``treatment`` (61).
CANONICAL_TREATMENT = "treat"


#: theme -> (canonical spelling, legacy spellings to converge toward it).
#:
#: Only *high-confidence* synonyms live here.  Ambiguous symbols (``group``,
#: ``w``/``W``, ``cov_type``) are intentionally absent — they are handled by
#: ``FALSE_FRIENDS`` so the lint never penalises a semantically-distinct param.
THEMES: Dict[str, Dict[str, object]] = {
    "se": {
        "canonical": CANONICAL_SE,
        "aliases": (
            "robust",
            "vcov",
            "se_type",
            "vcov_type",
            "variance_type",
            "cov_type",  # exempted on mixed-model family via FALSE_FRIENDS
        ),
        "note": (
            "Standard-error / variance type.  `vcov` is the pyfixest/R-fixest "
            "spelling and stays a permanently-accepted alias on the fixest "
            "family; `robust` is additionally type-overloaded (see "
            "ROBUST_BOOL_HINTS); `cov_type` is a false friend on mixed models "
            "(random-effects covariance structure) and is exempted there."
        ),
    },
    "cluster": {
        "canonical": "cluster",
        "aliases": ("cluster_var", "clustervar", "cluster_col"),
        "note": (
            "Clustering variable.  `clusters` (plural) is reserved for the "
            "multiway helpers that take a *list* of cluster keys, so it is not "
            "treated as a violation."
        ),
    },
    "weights": {
        "canonical": "weights",
        "aliases": ("weight", "sample_weight", "obs_weights"),
        "note": (
            "Observation weights.  `w`/`W` are NOT weights here (spatial weight "
            "matrix / EconML treatment) — see FALSE_FRIENDS."
        ),
    },
    "data": {
        "canonical": "data",
        "aliases": ("df", "frame", "dataset"),
        "note": "Input dataframe.  `data` is already near-universal (486 uses).",
    },
    "outcome": {
        "canonical": CANONICAL_OUTCOME,
        "aliases": ("outcome", "depvar", "dependent", "yname", "y_col", "Y"),
        "note": (
            "Outcome / dependent variable.  EconML-mirrored `Y` is exempt on "
            "the CATE / forest family (FALSE_FRIENDS)."
        ),
    },
    "treatment": {
        "canonical": CANONICAL_TREATMENT,
        "aliases": ("treatment", "treatvar", "treat_var", "d_var", "T", "W"),
        "note": (
            "Treatment variable.  EconML-mirrored `T` / `W` is exempt on the "
            "CATE / forest family, and `w`/`W` is exempt on spatial models "
            "(spatial weight matrix) — both via FALSE_FRIENDS."
        ),
    },
}


#: param spelling -> function names (or ``module:`` prefixes) where that
#: spelling is semantically unrelated to any theme and must NOT be flagged.
#: A ``module:`` entry matches when the resolved function's ``__module__``
#: contains that substring (e.g. ``"spatial"`` matches ``statspai.spatial.*``).
FALSE_FRIENDS: Dict[str, Tuple[str, ...]] = {
    # Spatial weight matrix W / w — orthogonal to observation weights.
    "w": ("module:spatial", "moran", "geary", "getis_ord_g", "lisa"),
    "W": (
        "module:spatial",
        # EconML-mirrored treatment symbol on the CATE / forest family.
        "module:metalearners",
        "module:neural_causal",
        "module:forest",
        # W is a *matrix*, not a treatment, in these contexts:
        "module:gmm",  # GMM weighting matrix
        "module:dag",  # do-calculus graph node set
        "module:interference",  # network adjacency matrix
        "rd_flex",  # RD flexible-adjustment covariate vector W
        "causal_forest",
        "average_treatment_effect",
        "cate_eval",
        "rate",
        "policy_tree",
    ),
    # Random-effects covariance *structure*, not an SE type.
    "cov_type": (
        "mixed",
        "meglm",
        "melogit",
        "meologit",
        "menbreg",
        "mepoisson",
        "megamma",
        "module:multilevel",
    ),
    # EconML / array-convention outcome & treatment, kept verbatim for
    # migration parity, plus do-calculus node labels.
    "Y": (
        "module:metalearners",
        "module:neural_causal",
        "module:forest",
        "module:dag",  # P(Y | do(X)) node label
        "module:interference",  # array-style outcome vector
        "causal_forest",
        "cate_eval",
        "rate",
    ),
    "T": (
        "module:metalearners",
        "module:neural_causal",
        "module:forest",
        "causal_forest",
        "average_treatment_effect",
    ),
}


#: Functions where a parameter named ``robust`` carries a *boolean* on/off
#: meaning rather than the HC-type *string* meaning it has in the regression
#: family.  This is the highest-impact hazard from the signature audit: the
#: same keyword name takes incompatible types across estimators.  The lint
#: surfaces the split from live defaults; this set documents the known
#: bool-typed sites for the convergence step (each should additionally accept
#: a `vce=` string alias).
ROBUST_BOOL_HINTS: Tuple[str, ...] = (
    "did",
    "ddd",
    "did_2x2",
    "did_analysis",
    "interactive_fe",
    "mixlogit",
    "xtabond",
)


def is_false_friend(spelling: str, func_name: str, module: str) -> bool:
    """Return True if ``spelling`` is an acknowledged false friend for a call.

    Parameters
    ----------
    spelling : str
        The parameter name as written in the signature.
    func_name : str
        The public function name (e.g. ``"causal_forest"``).
    module : str
        The function's ``__module__`` (e.g. ``"statspai.spatial.sar"``).
    """
    entries = FALSE_FRIENDS.get(spelling)
    if not entries:
        return False
    for entry in entries:
        if entry.startswith("module:"):
            if entry[len("module:") :] in (module or ""):
                return True
        elif entry == func_name:
            return True
    return False


def alias_index() -> Dict[str, str]:
    """Map every legacy spelling to its canonical spelling.

    Returns
    -------
    dict
        e.g. ``{"robust": "vce", "vcov": "vce", "outcome": "y", ...}``.
        Used by both the lint and the alias decorator.
    """
    out: Dict[str, str] = {}
    for theme in THEMES.values():
        canonical = str(theme["canonical"])
        for alias in theme["aliases"]:  # type: ignore[union-attr]
            out[str(alias)] = canonical
    return out


def canonical_for(spelling: str) -> str | None:
    """Return the canonical spelling for ``spelling`` (or None if unknown)."""
    if any(spelling == str(t["canonical"]) for t in THEMES.values()):
        return spelling
    return alias_index().get(spelling)
