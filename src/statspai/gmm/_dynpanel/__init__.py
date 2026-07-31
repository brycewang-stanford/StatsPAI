"""Internal machinery for dynamic-panel GMM (Arellano-Bond family).

Split by concern so that new moment sets are additive rather than another
rewrite:

===================  ======================================================
``_spec``            lag-operator parsing, instrument-block declarations
``_data``            long panel -> dense ``(unit, period)`` arrays
``_moments``         design matrix + block-diagonal instrument matrix
``_estimate``        weight matrices and the closed-form GMM solve
``_inference``       classical / robust / Windmeijer variance
``_diagnostics``     AR(1)/AR(2), Sargan, Hansen, instrument-count guard
``_fit``             orchestration
===================  ======================================================

Nothing here is public API; ``sp.xtabond`` is the user-facing entry point.
"""

from ._data import PanelArrays, add_time_dummies, build_panel_arrays
from ._diagnostics import (
    arellano_bond_ar_test,
    check_instrument_count,
    difference_in_hansen,
    overid_test,
)
from ._fit import DynPanelFit, fit_dynamic_panel
from ._moments import Design, build_design, first_difference_H
from ._spec import GMMBlock, IVBlock, Term, parse_terms, term_name

__all__ = [
    "PanelArrays",
    "build_panel_arrays",
    "add_time_dummies",
    "Design",
    "build_design",
    "first_difference_H",
    "GMMBlock",
    "IVBlock",
    "Term",
    "parse_terms",
    "term_name",
    "arellano_bond_ar_test",
    "check_instrument_count",
    "difference_in_hansen",
    "overid_test",
    "fit_dynamic_panel",
    "DynPanelFit",
]
