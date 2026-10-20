"""Data for the clean-room ``leebounds`` black-box fixture.

Samples (``sample`` column) for ``test_cleanroom_leebounds.py``:

1, 2  continuous outcomes of order 1e-7 (treated / control trimmed): Stata
      holds such a threshold in scientific notation with 12 significant
      digits, so the tie rule rarely applies;
3, 4  continuous outcomes, three covariate cells, no threshold tied in any
      cell (the between-cell divisor is the overall count);
5     integer outcomes, three cells, one cell whose trimmed arm is the less
      retained (negative trimming share);
6     ``(1 - p) n`` a whole number (``_pctile`` averages two order
      statistics).

Also writes ``leebounds_cleanroom_macro.csv``: doubles whose ``local m = v``
text Stata prints (the generator do-file records it).

    python tests/reference_parity/_fixtures/_generate_leebounds_cleanroom_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def _sample(seed, n, scale=1.0, flip=False, cells=False, kind="normal"):
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    x = rng.integers(0, 3, n)
    hi, lo = (0.9, 0.7) if not flip else (0.7, 0.9)
    s = (
        rng.random(n) < np.where(d == 1, hi, lo) + (0.03 * (x - 1) if cells else 0)
    ).astype(int)
    if kind == "normal":
        y = rng.normal(1 + d + (x if cells else 0), 1, n) * scale
    else:
        y = rng.poisson(2 + d + x, n).astype(float)
    return pd.DataFrame(dict(y=np.where(s == 1, y, np.nan), d=d, s=s, x=x))


def main() -> None:
    frames = [
        _sample(8801, 900, scale=1e-7),
        _sample(8802, 800, scale=1e-7, flip=True),
        # seeds chosen so that no cell threshold coincides with an outcome
        _sample(100087, 600, cells=True),
        _sample(100089, 600, cells=True, flip=True),
    ]
    rng = np.random.default_rng(9905)
    n = 1500
    d = rng.integers(0, 2, n)
    x = rng.integers(0, 3, n)
    p = np.where(x == 2, np.where(d == 1, 0.6, 0.8), np.where(d == 1, 0.85, 0.6))
    s = (rng.random(n) < p).astype(int)
    y = rng.poisson(2 + d + x, n).astype(float)
    frames.append(pd.DataFrame(dict(y=np.where(s == 1, y, np.nan), d=d, s=s, x=x)))
    rng = np.random.default_rng(5040)
    d = np.repeat([1, 0], 100)
    s = np.r_[np.ones(100), np.ones(80), np.zeros(20)].astype(int)
    y = np.round(rng.normal(size=200), 3)
    x = rng.integers(0, 3, 200)
    frames.append(pd.DataFrame(dict(y=np.where(s == 1, y, np.nan), d=d, s=s, x=x)))
    out = pd.concat(
        [f.assign(sample=k) for k, f in enumerate(frames, start=1)], ignore_index=True
    )
    out = out[["sample", "y", "d", "s", "x"]]
    out.to_csv(HERE / "leebounds_cleanroom.csv", index=False, float_format="%.17g")

    rng = np.random.default_rng(11)
    v = np.r_[
        rng.normal(size=60) * 10.0 ** rng.integers(-9, 18, 60),
        np.float32(rng.lognormal(3, 1.2, 30)).astype(float),
        [0.1, 1 / 3, 2 / 3, 1e-5, 9.5e-6, 1e15 + 0.25, 1e16 + 2, 17.350250244140625],
    ]
    pd.DataFrame(dict(v=v)).to_csv(
        HERE / "leebounds_cleanroom_macro.csv", index=False, float_format="%.17g"
    )
    print("wrote leebounds_cleanroom.csv and leebounds_cleanroom_macro.csv")


if __name__ == "__main__":
    main()
