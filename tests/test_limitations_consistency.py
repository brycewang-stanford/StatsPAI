"""Runtime consistency tests for ``FunctionSpec.limitations``.

The ``limitations`` field on every :class:`statspai.registry.FunctionSpec`
declares parameter values / variant gaps that are documented as
not-yet-implemented inside an otherwise stable function — see
``docs/guides/stability.md``. Without this test those advertisements
silently rot the moment the underlying code learns a new variant or the
function is renamed: the registry would still claim the gap exists but
the runtime no longer enforces it (or vice versa).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

ALLOWED_PHRASES: Tuple[str, ...] = (
    "not yet implemented",
    "not implemented",
    "not yet supported",
    "is not supported",
    "not currently supported",
    "raises notimplementederror",
    "raises importerror",
    "currently requires",
    "currently falls back",
    "is an mvp",
    "mvp",
    "only",
    "fallback",
    "silently",
    "deterministic fallback",
    # Vetted convention-gap vocabulary: T4-style disclosures that an
    # estimand is identified but the implementation-level object is not
    # unique, or that a default selector/aggregation convention differs
    # from the reference. These describe a documented parity boundary
    # (Section 5.4 validation tiers), not a code path that raises.
    "convention",
    "can differ",
)


LIMITATIONS_DESCRIPTIVE_ONLY: Dict[str, List[str]] = {
    "match": [
        # Both are documented parity boundaries, not code paths that raise:
        # greedy matching without replacement is genuinely order-dependent
        # (every package picks a convention), and the with-replacement tie
        # rule differs from Matching::Match. Pinned in
        # tests/reference_parity/test_matching_r_parity.py.
        "greedy nearest-neighbour matching without replacement is",
        "bias_correction=True follows a different convention",
    ],
    "did_multiplegt_dyn": [
        "switch-off events are handled, but the",
        "se_method='analytic' is available but the paper's own",
        # Documented parity boundary, not a code path that raises: the
        # headline weights differ from DIDmultiplegtDYN's Av_tot_eff and
        # both are reachable. Pinned in Track A module 78.
        "the headline aggregation convention differs",
    ],
    "spillover_did": [
        # Scope statements, not code paths that raise. The first is the
        # honest headline: this estimator has no reference implementation.
        "there is no reference implementation",
        "ring boundaries are the analyst's choice",
        "covariate adjustment is not implemented",
    ],
    "cgs_continuous_did": [
        # All three are scope statements about this implementation, not
        # code paths that raise. The per-cell estimator is pinned; what is
        # not implemented is named.
        "standard errors come from the per-cell influence function",
        "staggered designs aggregate cells with StatsPAI's own",
        "the cck (nonparametric) dose estimator is not implemented",
    ],
    "functional_form_test": [
        # Both are properties of the test, not code paths that raise: it is
        # a moment-inequality test, so failing to reject is not evidence of
        # the null, and its critical value is asymptotic by construction.
        "a large p-value is only a failure to reject",
        "standard errors and the critical value are asymptotic",
    ],
    "ddd_heterogeneous": [
        # Same shape: the (g, t) cells match triplediff::ddd exactly and
        # only the aggregation weights and the SE method differ, both
        # documented and both pinned in Track A module 77.
        "the placebo joint test is only produced on the bootstrap path",
        "control_group='notyettreated' is only partially comparable",
        "the aggregation convention differs",
    ],
    "continuous_did": [
        "method='cgs' is an MVP",
    ],
    "network_exposure": [
        "design='complete' is reserved but not implemented",
    ],
    "text_treatment_effect": [
        "embedder='sbert' requires the optional sentence-transformers",
        "Veitch et al. (2020) full BERT/topic-model recipe",
    ],
    "principal_strat": [
        # The function now implements the basic AIR / Wald LATE under
        # the encouragement-design path (Step G); the remaining gap is
        # always-survivor SACE under Mealli & Pacini (2013) partial
        # identification, which has no hard exception to test.
        "Always-survivor SACE under encouragement design",
    ],
    # ----- T4-style documented convention-gap disclosures -------------
    # These describe a parity boundary (default selector/aggregation
    # convention, or implementation-level non-uniqueness) rather than a
    # code path that raises; they are graded T4 in Section 5.4 and the
    # cross-language row is retained only as a disclosed gap.
    "causal_forest": [
        "validated against grf on clean-overlap designs only",
    ],
    "did_imputation": [
        "untreated-only TWFE",
        "simple ATT convention only",
    ],
    "etwfe": [
        "aggregation-convention sensitive",
        "cohort-share weighting convention",
        # Describes what the nonlinear branch *returns* (a response-scale
        # average marginal effect, matching R etwfe::emfx) rather than a code
        # path that raises, so it is descriptive.  The numeric claim is pinned
        # in tests/reference_parity/test_etwfe_glm_parity.py.
        "reports an average marginal effect",
    ],
    "pretrends_equivalence": [
        # Describes an intentionally absent default (the equivalence bound is
        # a substantive judgement in outcome units), not a code path that
        # raises. Pinned in test_fect_equivalence_parity.py.
        "the TOST is computed only when tost_threshold is supplied",
    ],
    "rddensity": [
        "not a reference-parity guarantee",
    ],
    "rdrobust": [
        "R-parity certification applies",
    ],
    "synth": [
        "documented Kaul-style convention",
        "regularisation or local-optimum convention gaps",
        "documented local-optimum convention",
    ],
}


def _runtime_map() -> (
    Dict[Tuple[str, str], Tuple[Callable[[], Any], type | tuple[type, ...]]]
):
    """Build the (function_name, limitation_substring) -> (call, exc) map."""
    import statspai as sp
    from statspai.exceptions import MethodIncompatibility

    rng = np.random.default_rng(0)
    n = 200

    df_panel = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "i": np.repeat(np.arange(n // 4), 4),
            "t": np.tile(np.arange(4), n // 4),
            "g": np.repeat(rng.choice([0, 2, 3], size=n // 4), 4),
            "d": rng.binomial(1, 0.4, size=n),
            "dose": rng.uniform(0, 1, size=n),
        }
    )

    df_cs = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": rng.binomial(1, 0.5, size=n).astype(float),
            "s": rng.binomial(1, 0.5, size=n).astype(float),
            "x": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.binomial(1, 0.5, size=n).astype(float),
            "w": rng.normal(size=n),
            "score": rng.normal(size=n),
        }
    )

    return {
        ("hal_tmle", "variant='projection'"): (
            lambda: sp.hal_tmle(
                df_cs,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                variant="projection",
            ),
            NotImplementedError,
        ),
        ("callaway_santanna", "clustervars"): (
            # estimator='dr' / control_group='notyettreated' are now supported
            # under panel=False; clustervars is the remaining gap.
            lambda: sp.callaway_santanna(
                df_panel,
                y="y",
                g="g",
                t="t",
                i="i",
                panel=False,
                estimator="dr",
                clustervars="i",
            ),
            NotImplementedError,
        ),
        ("etwfe", "cgroup='nevertreated' combined with panel=False"): (
            lambda: sp.etwfe(
                df_panel,
                y="y",
                group="i",
                time="t",
                first_treat="g",
                cgroup="nevertreated",
                panel=False,
            ),
            NotImplementedError,
        ),
        ("etwfe", "family='poisson'/'logit' with xvar"): (
            lambda: sp.etwfe(
                df_panel,
                y="y",
                group="i",
                time="t",
                first_treat="g",
                family="poisson",
                xvar="dose",
            ),
            MethodIncompatibility,
        ),
        ("rdrobust", "observation-level weights"): (
            lambda: sp.rdrobust(
                df_cs,
                y="y",
                x="x",
                c=0.0,
                weights="w",
            ),
            NotImplementedError,
        ),
        ("llm_annotator_correct", "method='hausman' is the only supported"): (
            lambda: sp.llm_annotator_correct(
                annotations_llm=df_cs["d"],
                outcome=df_cs["y"],
                annotations_human=df_cs["d"][:60],
                method="logistic",
            ),
            (NotImplementedError, ValueError),
        ),
    }


def _matches_allowed_vocabulary(limitation: str) -> bool:
    low = limitation.lower()
    return any(phrase in low for phrase in ALLOWED_PHRASES)


def _all_limitations() -> List[Tuple[str, str]]:
    """List every (function_name, limitation_string) pair in the registry.

    Forces a full submodule import first so the registry is fully
    populated regardless of which other tests have run in the same
    process; otherwise the parametrised set would be import-order
    dependent and the contract could pass standalone but fail in a full
    suite run (or vice versa).
    """
    import importlib
    import pkgutil

    import statspai
    import statspai as sp

    sp.list_functions()
    for _m in pkgutil.walk_packages(statspai.__path__, "statspai."):
        try:
            importlib.import_module(_m.name)
        except Exception:
            # Optional-extra modules (torch/jax/pymc) may be absent; the
            # registry entries we audit do not depend on importing them.
            pass
    from statspai.registry import _REGISTRY

    out: List[Tuple[str, str]] = []
    for name, spec in sorted(_REGISTRY.items()):
        for lim in spec.limitations:
            out.append((name, lim))
    return out


@pytest.mark.parametrize("name,limitation", _all_limitations())
def test_limitation_uses_allowed_vocabulary(name: str, limitation: str) -> None:
    """Every registered limitation must use vetted phrasing."""
    assert _matches_allowed_vocabulary(
        limitation
    ), f"sp.{name} limitation does not use vetted vocabulary: {limitation!r}"


def test_every_limitation_is_classified() -> None:
    """Every limitation must be either runtime-tested or explicitly descriptive."""
    runtime_keys = {key for key in _runtime_map().keys()}
    descriptive_keys = {
        (fn, sub) for fn, subs in LIMITATIONS_DESCRIPTIVE_ONLY.items() for sub in subs
    }

    unclassified: List[Tuple[str, str]] = []
    for name, limitation in _all_limitations():
        runtime_match = any(
            fn == name and sub in limitation for fn, sub in runtime_keys
        )
        descriptive_match = any(
            fn == name and sub in limitation for fn, sub in descriptive_keys
        )
        if not (runtime_match or descriptive_match):
            unclassified.append((name, limitation))

    assert not unclassified, (
        "The following limitations are not classified as either "
        "runtime-testable or descriptive:\n"
        + "\n".join(f"  sp.{n}: {lim!r}" for n, lim in unclassified)
    )


@pytest.mark.parametrize(
    "key",
    list(_runtime_map().keys()),
    ids=lambda k: f"{k[0]}::{k[1][:40]}",
)
def test_limitation_actually_raises(
    key: Tuple[str, str],
) -> None:
    """The documented limitation must trigger the documented exception."""
    call, exc = _runtime_map()[key]
    with pytest.raises(exc):
        call()
