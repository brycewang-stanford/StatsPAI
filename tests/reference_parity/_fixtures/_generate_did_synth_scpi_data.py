"""Write the California Prop 99 panel used by the scpi parity fixture.

    PYTHONPATH=src python \
        tests/reference_parity/_fixtures/_generate_did_synth_scpi_data.py

Writes ``did_synth_scpi_california.csv`` (state, year, packspercapita) from
``sp.california_prop99()`` with round-trip float formatting, so R
(``_generate_did_synth_scpi_R.R``) and the Python test read identical bytes.
The West Germany panel is written by the R generator from ``scpi::scpi_germany``.
"""

from pathlib import Path

import statspai as sp

out = Path(__file__).resolve().parent / "did_synth_scpi_california.csv"
df = sp.california_prop99()[["state", "year", "packspercapita"]]
df = df.sort_values(["state", "year"]).reset_index(drop=True)
df.to_csv(out, index=False, float_format="%.17g")
print(f"wrote {out} ({len(df)} rows)")
