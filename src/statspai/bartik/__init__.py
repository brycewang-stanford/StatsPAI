"""
Shift-Share (Bartik) Instrumental Variables for StatsPAI.

Constructs Bartik instruments from industry shares and national shocks,
with diagnostics for instrument validity following Goldsmith-Pinkham,
Sorkin, and Swift (2020) and Borusyak, Hull, and Jaravel (2022).

References
----------
Goldsmith-Pinkham, P., Sorkin, I., and Swift, H. (2020).
"Bartik Instruments: What, When, Why, and How."
*American Economic Review*, 110(8), 2586-2624. [@goldsmithpinkham2020bartik]

Borusyak, K., Hull, P., and Jaravel, X. (2022).
"Quasi-Experimental Shift-Share Research Designs."
*Review of Economic Studies*, 89(1), 181-213. [@borusyak2022quasi]
"""

from .shift_share import bartik, BartikIV
from .adao_correction import ssaggregate, shift_share_se
from .political import (
    shift_share_political, ShiftSharePoliticalResult,
    shift_share_political_panel, ShiftSharePoliticalPanelResult,
)

__all__ = [
    'bartik', 'BartikIV', 'ssaggregate', 'shift_share_se',
    'shift_share_political', 'ShiftSharePoliticalResult',
    'shift_share_political_panel', 'ShiftSharePoliticalPanelResult',
]
