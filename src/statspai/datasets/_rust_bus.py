"""Rust (1987) bus engine replacement data, fetched on demand.

The odometer readings and engine replacement records of the 162 Madison
Metro buses analysed by Rust (1987). John Rust distributes them with his
NFXP software (``editorialexpress.com/jrust/nfxp.zip``) without a stated
licence; the OpenSourceEconomics ``zurcher-data`` archive on Zenodo (MIT,
DOI 10.5281/zenodo.3385050) holds the same nine ``.asc`` files, value for
value. StatsPAI fetches that archive (319 kB) on first use, checks its
pinned SHA-256 and caches it (:mod:`statspai.datasets._fetch`); nothing is
redistributed.

The discretisation follows Rust's ``stordat.gpr`` line by line, so the
frame reproduces the counts behind Tables V, VI, IX and X of the paper.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np
import pandas as pd

from ._fetch import RemoteFile, fetch

if TYPE_CHECKING:
    import zipfile

RUST_BUS = RemoteFile(
    filename="zurcher-data-v0.14.zip",
    url=(
        "https://zenodo.org/api/records/3385050/files/"
        "OpenSourceEconomics/zurcher-data-v0.14.zip/content"
    ),
    sha256="8b1c267c0311c036eb957234c08c5c80288504452c16bc7f2899b8f9372cf16f",
    size=318793,
    licence="MIT (OpenSourceEconomics/zurcher-data LICENSE); the data are "
    "John Rust's NFXP files, distributed without a separate licence; "
    "fetched, not redistributed",
    origin="Zenodo record 3385050 (zurcher-data v0.14, DOI "
    "10.5281/zenodo.3385050), data/original_data/*.asc; values equal "
    "editorialexpress.com/jrust/nfxp.zip",
)

#: Rust (1987) group number -> (file stem, number of buses), as in Table I.
_GROUPS = {
    1: ("g870", 15),  # 1983 Grumman model 870
    2: ("rt50", 4),  # 1981 Chance RT-50
    3: ("t8h203", 48),  # 1979 GMC T8H203
    4: ("a530875", 37),  # 1975 GMC A5308
    5: ("a530874", 12),  # 1974 GMC A5308
    6: ("a452374", 10),  # 1974 GMC A4523
    7: ("a530872", 18),  # 1972 GMC A5308
    8: ("a452372", 18),  # 1972 GMC A4523
}

#: Columns of the frame ``rust_bus()`` returns (listed without downloading).
COLUMNS = (
    "group",
    "bus",
    "period",
    "odometer",
    "mileage",
    "state",
    "replace",
    "increment",
    "rust_sample",
)


def _bad(msg: str, hint: str, **diag: object) -> Exception:
    from ..exceptions import MethodIncompatibility

    return MethodIncompatibility(msg, recovery_hint=hint, diagnostics=diag)


def _read_group(archive: "zipfile.ZipFile", group: int) -> np.ndarray:
    """The ``(buses, records)`` matrix of one group's ``.asc`` file."""
    from ..exceptions import DataInsufficient

    stem, n_bus = _GROUPS[group]
    member = [n for n in archive.namelist() if n.endswith(f"original_data/{stem}.asc")]
    if len(member) != 1:
        raise DataInsufficient(
            f"{stem}.asc is missing from {RUST_BUS.filename}",
            recovery_hint="Delete the cached archive and download it again.",
            diagnostics={"member": stem, "found": member},
        )
    values = np.array(archive.read(member[0]).decode("ascii").split(), dtype=float)
    # Rust's GAUSS reshape(dt, buses, rows/buses): each bus is one block of
    # consecutive values (11 header records, then monthly odometer readings).
    return values.reshape(n_bus, -1)


def _discretise(
    dt: np.ndarray, group: int, n_states: int, max_mileage: float
) -> pd.DataFrame:
    """Rust's ``stordat.gpr`` for one group, one row per bus-month."""
    ov1, ov2 = dt[:, 5:6], dt[:, 8:9]
    odo = dt[:, 11:]
    # cumulative replacements; a replacement counts from the first reading
    # at or above the odometer value recorded for it
    cum = ((odo >= ov1) & (ov1 > 0)).astype(int) + ((odo >= ov2) & (ov2 > 0)).astype(
        int
    )
    mileage = odo + ov1 * cum * (cum - 2) - 0.5 * ov2 * cum * (cum - 1)
    dtx = np.ceil(n_states * mileage / max_mileage)  # Rust's 1-based bin
    if dtx.max() > n_states:
        raise _bad(
            f"mileage {mileage.max():.0f} exceeds max_mileage={max_mileage:g}",
            "Raise max_mileage (Rust uses 450,000) or n_states.",
            max_mileage=max_mileage,
            largest_mileage=float(mileage.max()),
        )
    replace = np.zeros_like(cum)
    replace[:, :-1] = cum[:, 1:] - cum[:, :-1]
    # Rust's monthly mileage: the bin step, measured from bin index 0 (one
    # below the 1-based bin of zero mileage) after a replacement
    increment = np.full(dtx.shape, np.nan)
    increment[:, 1:] = dtx[:, 1:] - dtx[:, :-1] + dtx[:, :-1] * replace[:, :-1]
    n_bus, T = odo.shape
    return pd.DataFrame(
        {
            "group": np.full(n_bus * T, group, dtype=np.int64),
            "bus": np.repeat(dt[:, 0].astype(np.int64), T),
            "period": np.tile(np.arange(T, dtype=np.int64), n_bus),
            "odometer": odo.ravel(),
            "mileage": mileage.ravel(),
            "state": np.maximum(dtx - 1, 0).astype(np.int64).ravel(),
            "replace": replace.astype(np.int64).ravel(),
            "increment": increment.ravel(),
            "rust_sample": np.tile(np.arange(T) > 0, n_bus),
        }
    )


def rust_bus(
    groups: Union[Sequence[int], str] = (1, 2, 3, 4),
    n_states: int = 90,
    max_mileage: float = 450_000.0,
    download: bool = True,
) -> pd.DataFrame:
    """Rust (1987) Madison Metro bus engine replacement panel — real data.

    Monthly odometer readings and engine replacements of the buses in
    Rust's "Optimal Replacement of GMC Bus Engines" [@rust1987optimal],
    discretised exactly as Rust's NFXP pre-processor ``stordat.gpr`` does.
    The raw files are fetched on first use from the Zenodo archive of
    OpenSourceEconomics' ``zurcher-data`` (MIT; the same values as the
    files in Rust's ``nfxp.zip``) and cached; see
    :mod:`statspai.datasets._fetch`.

    Parameters
    ----------
    groups : sequence of int or 'all', default (1, 2, 3, 4)
        Rust's bus groups (Table I): 1 Grumman 870 (1983), 2 Chance RT-50
        (1981), 3 GMC T8H203 (1979), 4 GMC A5308 (1975), 5 GMC A5308
        (1974), 6 GMC A4523 (1974), 7 GMC A5308 (1972), 8 GMC A4523 (1972).
        Groups 1-4 are the estimation sample of Tables IX and X.
    n_states : int, default 90
        Fixed-point dimension: number of mileage bins (Table IX uses 90,
        Table X 175).
    max_mileage : float, default 450000
        Upper bound of the mileage grid; bins are ``max_mileage / n_states``
        miles wide (5,000 at the defaults).
    download : bool, default True
        Download the archive when it is not cached. With ``False`` a
        missing file raises instead of touching the network.

    Returns
    -------
    pd.DataFrame
        One row per bus and month:

        - ``group``, ``bus`` (Rust's four-digit bus number), ``period``
          (month index from the bus's first reading);
        - ``odometer`` (miles) and ``mileage`` (miles since the last engine
          replacement);
        - ``state``: mileage bin ``ceil(n_states * mileage / max_mileage) - 1``,
          i.e. Rust's 1-based bin shifted to start at 0;
        - ``replace``: 1 if the engine is replaced between this reading and
          the next (Rust's ``i_t``); 0 in a bus's last month;
        - ``increment``: Rust's discretised monthly mileage into this month
          (NaN in the first month). After a replacement it is the next bin
          on Rust's 1-based scale, one more than the step from state 0;
          this is the convention behind the published transition estimates;
        - ``rust_sample``: False for each bus's first month, which Rust
          drops (8,156 rows for groups 1-4 at the defaults).

        ``attrs`` carries the source URL, the pinned SHA-256 and the licence.

    Raises
    ------
    DataInsufficient
        Offline and not cached, or the download does not match the pinned
        SHA-256. The message gives the URL and the path to save it at.
    MethodIncompatibility
        Unknown group, or a grid that does not cover the observed mileage.

    Notes
    -----
    With ``beta=0.9999`` and the default grid, ``sp.ddc`` on the
    ``rust_sample`` rows reproduces Table IX (groups 1-4: ``RC`` 9.7558,
    ``theta_11`` 2.6275) and the partial log-likelihoods of Table VIII,
    given Rust's increment frequencies and ``vce='opg'`` (his BHHH
    standard errors). Table X (175 bins) prints ``RC = 10.896`` for group
    4; the data give 10.0896 together with the printed log-likelihood
    -4495.135, so the printed value has lost a digit
    (``tests/external_parity/test_rust1987_bus.py``).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> df = sp.datasets.rust_bus()  # doctest: +SKIP
    >>> s = df[df["rust_sample"]]  # doctest: +SKIP
    >>> p = np.bincount(s["increment"].astype(int)) / len(s)  # doctest: +SKIP
    >>> fit = sp.ddc(s, "replace", "state", beta=0.9999, n_states=90,
    ...              transition=p, vce="opg")  # doctest: +SKIP
    >>> fit.params.round(4).tolist()  # doctest: +SKIP
    [9.7557, 2.6276]

    References
    ----------
    [@rust1987optimal]
    """
    if isinstance(groups, str):
        if groups != "all":
            raise _bad(
                f"rust_bus: groups must be a sequence of 1..8 or 'all', got {groups!r}",
                "Use groups=(1, 2, 3, 4) for Rust's estimation sample.",
                groups=groups,
            )
        chosen = list(_GROUPS)
    else:
        chosen = [int(g) for g in groups]
        unknown = [g for g in chosen if g not in _GROUPS]
        if unknown or not chosen or len(set(chosen)) != len(chosen):
            raise _bad(
                f"rust_bus: groups must be distinct values in 1..8, got {list(groups)}",
                "Use groups=(1, 2, 3, 4) for Rust's estimation sample, or 'all'.",
                groups=list(groups),
            )
    if int(n_states) != n_states or n_states < 2:
        raise _bad(
            f"rust_bus: n_states must be an integer >= 2, got {n_states!r}",
            "Rust uses 90 (Table IX) or 175 (Table X).",
            n_states=n_states,
        )
    if not np.isfinite(max_mileage) or max_mileage <= 0:
        raise _bad(
            f"rust_bus: max_mileage must be positive, got {max_mileage!r}",
            "Rust uses 450,000.",
            max_mileage=max_mileage,
        )
    import zipfile

    path = fetch(RUST_BUS, download=download)
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        frames = [
            _discretise(_read_group(archive, g), g, int(n_states), float(max_mileage))
            for g in chosen
        ]
    df = pd.concat(frames, ignore_index=True)
    df.attrs["paper"] = (
        "Rust (1987), Optimal Replacement of GMC Bus Engines: An Empirical "
        "Model of Harold Zurcher, Econometrica 55(5)"
    )
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_url"] = RUST_BUS.url
    df.attrs["sha256"] = RUST_BUS.sha256
    df.attrs["licence"] = RUST_BUS.licence
    df.attrs["source_origin"] = RUST_BUS.origin
    df.attrs["n_states"] = int(n_states)
    df.attrs["max_mileage"] = float(max_mileage)
    df.attrs["groups"] = chosen
    return df
