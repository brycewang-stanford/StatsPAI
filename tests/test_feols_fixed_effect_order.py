"""``sp.feols`` with several fixed effects gives the same numbers in every process.

pyfixest orders the fixed effects through a ``set`` of their names, so the
sweep order of the alternating projections followed ``PYTHONHASHSEED`` and
standard errors differed between runs of the same call (4e-9 relative for a
one-way clustered SE here, up to 1.7e-6 with two-way clustering).
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from statspai.fixest.wrapper import _written_order_encoder  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")
SCRIPT = """
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import statspai as sp
rng = np.random.default_rng(0); n = 20000
df = pd.DataFrame({"w": rng.integers(0, 3000, n), "f": rng.integers(0, 300, n),
                   "g": rng.integers(0, 40, n)})
df["x"] = rng.normal(size=n) + 0.01 * df.w % 7
df["y"] = 0.5 * df.x + 0.01 * (df.w % 13) + 0.1 * (df.f % 5) + rng.normal(size=n)
for fml, v in [("y ~ x | w + f", {"CRV1": "w"}), ("y ~ x | w + f + g", {"CRV1": "w+f"})]:
    r = sp.feols(fml, data=df, vcov=v)
    print(repr(float(r.params["x"])), repr(float(r.std_errors["x"])))
"""


def _run(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_same_numbers_under_every_hash_seed():
    outputs = {_run(seed) for seed in ("0", "1", "2", "3")}
    assert len(outputs) == 1, outputs


def test_encoder_restores_the_written_order():
    def shuffled(fixed_effects, data):
        # what pyfixest returns when the set iterates in another order
        return "__fixed_effect__(b_c) + __fixed_effect__(a) - 1"

    enc = _written_order_encoder(shuffled)
    assert enc("a + b^c", pd.DataFrame()) == (
        "__fixed_effect__(a) + __fixed_effect__(b_c) - 1"
    )


def test_unrecognised_output_passes_through():
    enc = _written_order_encoder(lambda fixed_effects, data: "something else")
    assert enc("a + b", pd.DataFrame()) == "something else"
