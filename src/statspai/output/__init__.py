"""
Output utilities for regression and causal inference results.
"""

from .outreg2 import OutReg2, outreg2
from .modelsummary import modelsummary, coefplot
from .sumstats import sumstats, balance_table
from .tab import tab
from .estimates import eststo, estclear, esttab, EstimateTableResult
from .regression_table import regtable, RegtableResult, mean_comparison, MeanComparisonResult
from .collection import Collection, CollectionItem, collect
from .paper_tables import paper_tables, PaperTables, TEMPLATES as PAPER_TABLE_TEMPLATES
from ._inline import cite
from ._journals import JOURNALS, list_templates as list_journal_templates, get_template as get_journal_template
from ._lineage import (
    Provenance,
    attach_provenance,
    get_provenance,
    compute_data_hash,
    format_provenance,
    lineage_summary,
)
from ._replication_pack import ReplicationPack, replication_pack
from ._gt import to_gt, is_great_tables_available
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
    "OutReg2",
    "outreg2",
    "modelsummary",
    "coefplot",
    "sumstats",
    "balance_table",
    "tab",
    "eststo",
    "estclear",
    "esttab",
    "EstimateTableResult",
    "regtable",
    "RegtableResult",
    "mean_comparison",
    "MeanComparisonResult",
    "Collection",
    "CollectionItem",
    "collect",
    "paper_tables",
    "PaperTables",
    "PAPER_TABLE_TEMPLATES",
    "cite",
    "JOURNALS",
    "list_journal_templates",
    "get_journal_template",
    # Lineage / provenance
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
