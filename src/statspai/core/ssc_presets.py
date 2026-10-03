"""Small-sample-correction presets that reproduce Stata's commands.

``sp.feols`` (pyfixest backend) takes an ``ssc=`` dictionary whose four
switches decide the finite-sample factor of the variance. Matching a Stata
table meant knowing, per command, which switches Stata implies -- ``areg``
counts fixed effects nested in the cluster, ``reghdfe`` and ``xtreg, fe`` do
not, ``ivregress`` applies no factor unless ``small`` is given. The presets
below were read off Stata 18 on ``tests/reference_parity/_fixtures/
ssc_presets.csv`` (firm effects nested in industry clusters), where each
reproduces the command's standard error to 1e-12.
"""

from __future__ import annotations

from typing import Any, Dict

from ..exceptions import MethodIncompatibility

_PRESETS: Dict[str, Dict[str, Any]] = {
    # regress, vce(robust | cluster): (n-1)/(n-k) and G/(G-1)
    "regress": {"k_adj": True, "k_fixef": "full", "G_adj": True},
    # areg ..., absorb(): absorbed levels count in k even when nested
    "areg": {"k_adj": True, "k_fixef": "full", "G_adj": True},
    # reghdfe / xtreg, fe: levels nested in the cluster are not counted
    "reghdfe": {"k_adj": True, "k_fixef": "nonnested", "G_adj": True},
    "xtreg": {"k_adj": True, "k_fixef": "nonnested", "G_adj": True},
    # ivregress 2sls without `small`: no finite-sample factor at all
    "ivregress": {"k_adj": False, "k_fixef": "nonnested", "G_adj": False},
    # ivregress 2sls, small
    "ivregress_small": {"k_adj": True, "k_fixef": "full", "G_adj": True},
    # pyfixest / fixest default
    "fixest": {"k_adj": True, "k_fixef": "nonnested", "G_adj": True},
}


def ssc(preset: str = "fixest", **overrides: Any) -> Dict[str, Any]:
    """Small-sample correction for ``sp.feols(..., ssc=...)`` by Stata command.

    Parameters
    ----------
    preset : str, default 'fixest'
        ``'regress'``, ``'areg'``, ``'reghdfe'``, ``'xtreg'`` (``xtreg, fe``),
        ``'ivregress'`` (``ivregress 2sls`` without ``small``),
        ``'ivregress_small'`` or ``'fixest'`` (the backend default). Stata
        spellings with a ``stata_`` prefix are accepted.
    **overrides
        Any of pyfixest's switches (``k_adj``, ``k_fixef``, ``G_adj``,
        ``G_df``) to change on top of the preset.

    Returns
    -------
    dict
        Pass as ``sp.feols(..., ssc=sp.ssc('areg'))``.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.ssc("areg")["k_fixef"]
    'full'
    >>> sp.ssc("ivregress")["k_adj"]
    False
    """
    key = str(preset).lower().replace("-", "_")
    if key.startswith("stata_"):
        key = key[len("stata_") :]
    if key not in _PRESETS:
        raise MethodIncompatibility(
            f"Unknown ssc preset {preset!r}.",
            recovery_hint=f"Use one of {sorted(_PRESETS)}.",
            diagnostics={"preset": preset},
        )
    from .._optional_deps import require_optional

    pf = require_optional(
        "pyfixest",
        extra="fixest",
        purpose="small-sample correction presets for sp.feols",
    )

    cfg = dict(_PRESETS[key])
    cfg.update(overrides)
    return dict(pf.ssc(**cfg))
