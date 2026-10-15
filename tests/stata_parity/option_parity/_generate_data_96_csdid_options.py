"""Data for 96_csdid_options.do: mpdta with every 37th row removed.

mpdta (``tests/orig_parity/data/02_mpdta_original.csv``) in its stored row
order, dropping rows 37, 74, ... (one-based): 67 county-years go missing, so
the panel is unbalanced and the cells of ``csdid``'s default pairwise rule
(StatsPAI ``balance='pair'``) are built from different units.

    python tests/stata_parity/option_parity/_generate_data_96_csdid_options.py
"""

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
d = pd.read_csv(_ROOT / "tests" / "orig_parity" / "data" / "02_mpdta_original.csv")
d = d.drop(index=d.index[36::37]).reset_index(drop=True)
d.to_csv(Path(__file__).parent / "data_96_csdid_unbalanced.csv", index=False)
print(f"wrote data_96_csdid_unbalanced.csv: {len(d)} rows")
