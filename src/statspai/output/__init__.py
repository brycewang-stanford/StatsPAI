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

# ── Regression-table renderers ──────────────────────────────────────────
from .regression_table import regtable, RegtableResult
from .estimates import eststo, estclear, esttab, EstimateTableResult
from .modelsummary import modelsummary, coefplot
from .outreg2 import OutReg2, outreg2

# ── Single-table helpers ────────────────────────────────────────────────
from .sumstats import sumstats, balance_table
from .tab import tab
from .mean_comparison import mean_comparison, MeanComparisonResult

# ── Multi-table / paper bundles ─────────────────────────────────────────
from .paper_tables import paper_tables, PaperTables, TEMPLATES as PAPER_TABLE_TEMPLATES
from .collection import Collection, CollectionItem, collect

# ── Inline citation ─────────────────────────────────────────────────────
from ._inline import cite

# ── Journal templates ───────────────────────────────────────────────────
from ._journals import (
    JOURNALS,
    list_templates as list_journal_templates,
    get_template as get_journal_template,
)

# ── Provenance / lineage ────────────────────────────────────────────────
from ._lineage import (
    Provenance,
    attach_provenance,
    get_provenance,
    compute_data_hash,
    format_provenance,
    lineage_summary,
)

# ── Replication pack ────────────────────────────────────────────────────
from ._replication_pack import ReplicationPack, replication_pack

# ── great_tables adapter ────────────────────────────────────────────────
from ._gt import to_gt, is_great_tables_available

# ── Bibliography / CSL ──────────────────────────────────────────────────
from ._bibliography import (
    CSL_REGISTRY,
    csl_url,
    csl_filename,
    list_csl_styles,
    parse_citation_to_bib,
    make_bib_key,
    citations_to_bib_entries,
    write_bib,
)


__all__ = [
    # Regression-table renderers (canonical first)
    "regtable",
    "RegtableResult",
    "esttab",
    "eststo",
    "estclear",
    "EstimateTableResult",
    "modelsummary",
    "coefplot",
    "outreg2",
    "OutReg2",
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
