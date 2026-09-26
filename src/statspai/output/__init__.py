"""Output utilities for regression and causal-inference results.

The package is organised by purpose:

**Regression-table renderers** (4 entry points historically — see PR-B
design doc; ``regtable`` is the canonical one):

- :func:`regtable` — canonical multi-model regression table renderer.
  Supports text / HTML / LaTeX / Markdown / Quarto / Excel / Word /
  DataFrame, journal templates, multi-row SE, repro provenance.
- :func:`esttab` — Stata ``estout`` / ``esttab`` clone (use ``eststo``
  to register, then ``esttab`` to print). Thin Stata-flavoured surface.
- :func:`modelsummary` — R ``modelsummary`` clone (functional API).
- :func:`outreg2` / :class:`OutReg2` — Stata ``outreg2`` clone
  (Excel-first surface).

**Single-table helpers**:

- :func:`tab` — Stata-style ``tabulate``.
- :func:`sumstats` — descriptive summary statistics.
- :func:`balance_table` — covariate-balance table.
- :func:`mean_comparison` — two-group mean comparison with t-test /
  ranksum / chi2 (lives in ``mean_comparison.py`` since v1.6.x —
  re-exported from ``regression_table`` for back-compat).

**Multi-table / paper bundles**:

- :func:`paper_tables` — Main / Heterogeneity / Robustness panels.
- :class:`Collection` / :func:`collect` — narrative document builder.

**Plotting**:

- :func:`coefplot` — coefficient plot.

**Provenance / replication / citations**:

- :class:`Provenance`, :func:`attach_provenance`, :func:`get_provenance`,
  :func:`compute_data_hash`, :func:`format_provenance`,
  :func:`lineage_summary`.
- :class:`ReplicationPack`, :func:`replication_pack`.
- :func:`cite`, :data:`CSL_REGISTRY`, :func:`csl_url`, ...

**Adapters**:

- :func:`to_gt`, :func:`is_great_tables_available` — ``great_tables``
  adapter (lazy).
"""

# ── Bibliography / CSL ──────────────────────────────────────────────────
from ._bibliography import (
    CSL_REGISTRY,
    citations_to_bib_entries,
    csl_filename,
    csl_url,
    list_csl_styles,
    make_bib_key,
    parse_citation_to_bib,
    write_bib,
)

# ── great_tables adapter ────────────────────────────────────────────────
from ._gt import is_great_tables_available, to_gt

# ── Inline citation ─────────────────────────────────────────────────────
from ._inline import cite

# ── Journal templates ───────────────────────────────────────────────────
from ._journals import JOURNALS
from ._journals import get_template as get_journal_template
from ._journals import list_templates as list_journal_templates

# ── Provenance / lineage ────────────────────────────────────────────────
from ._lineage import (
    Provenance,
    attach_provenance,
    compute_data_hash,
    format_provenance,
    get_provenance,
    lineage_summary,
)

# ── Replication pack ────────────────────────────────────────────────────
from ._replication_pack import ReplicationPack, replication_pack
from ._replication_verify import ReplicationVerification, verify_replication_pack
from .collection import Collection, CollectionItem, collect
from .estimates import EstimateTableResult, estclear, eststo, esttab
from .mean_comparison import MeanComparisonResult, mean_comparison
from .modelsummary import coefplot, coefplot_tikz, modelsummary  # noqa: F401
from .outreg2 import OutReg2, outreg2  # noqa: F401

# ── Multi-table / paper bundles ─────────────────────────────────────────
from .paper_tables import TEMPLATES as PAPER_TABLE_TEMPLATES
from .paper_tables import PaperTables, paper_tables

# ── Regression-table renderers ──────────────────────────────────────────
from .regression_table import RegtableResult, regtable

# ── Single-table helpers ────────────────────────────────────────────────
from .sumstats import balance_table, sumstats
from .tab import tab

__all__ = [
    # Regression-table renderers (canonical first)
    "regtable",
    "RegtableResult",
    "esttab",
    "eststo",
    "estclear",
    "EstimateTableResult",
    "coefplot",
    "coefplot_tikz",
    # Single-table helpers
    "sumstats",
    "balance_table",
    "tab",
    "mean_comparison",
    "MeanComparisonResult",
    # Multi-table / paper bundles
    "paper_tables",
    "PaperTables",
    "PAPER_TABLE_TEMPLATES",
    "Collection",
    "CollectionItem",
    "collect",
    # Inline citation
    "cite",
    # Journal templates
    "JOURNALS",
    "list_journal_templates",
    "get_journal_template",
    # Provenance / lineage
    "Provenance",
    "attach_provenance",
    "get_provenance",
    "compute_data_hash",
    "format_provenance",
    "lineage_summary",
    # Replication pack
    "ReplicationPack",
    "replication_pack",
    "verify_replication_pack",
    "ReplicationVerification",
    # great_tables adapter
    "to_gt",
    "is_great_tables_available",
    # Bibliography / CSL
    "CSL_REGISTRY",
    "csl_url",
    "csl_filename",
    "list_csl_styles",
    "parse_citation_to_bib",
    "make_bib_key",
    "citations_to_bib_entries",
    "write_bib",
]
