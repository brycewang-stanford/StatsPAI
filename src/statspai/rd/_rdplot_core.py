"""Numerical core of :func:`statspai.rdplot` and :func:`statspai.rdplotdensity`.

Re-exports the clean-room implementations (2026-10-20):
:func:`statspai.rd._rdplot_numbers.rdplot_numbers` (binned means and the
global polynomial of an RD plot, Calonico, Cattaneo and Titiunik 2015, bib
key ``calonico2015optimal``) and
:func:`statspai.rd._lpdensity.lpdensity_numbers` (local polynomial density,
Cattaneo, Jansson and Ma 2020, bib key ``cattaneo2020simple``). The
references R ``rdrobust::rdplot`` (GPL-3) and ``lpdensity`` (GPL-2) were
used only as black boxes; specifications in ``docs/dev/cleanroom/``.
"""

from __future__ import annotations

from ._lpdensity import lpdensity_numbers
from ._rdplot_numbers import gram_inverse as _xx_inv
from ._rdplot_numbers import kernel_weights as _kweight
from ._rdplot_numbers import rdplot_numbers

__all__ = ["rdplot_numbers", "lpdensity_numbers"]
__all__ += ["_kweight", "_xx_inv"]
