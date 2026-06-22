"""Normalise data-source payloads into StatsPAI-ready tidy frames.

The 2026 econometrics workflow is *data MCP → estimator*: an agent pulls a
series from the World Bank, FRED or OECD MCP server, then hands it to a model.
StatsPAI's job is to be the **best consumer** of those payloads, not to
re-implement the (excellent, official) data connectors. These helpers take the
JSON/dict/DataFrame a data MCP already returned and reshape it into the tidy
long (or wide) panel / time-series frame that ``sp.detect_design`` →
``sp.recommend`` → ``sp.feols`` / ``sp.cross_validate`` expect.

No network calls happen here — payloads come in already fetched, so these
functions are deterministic and offline-testable.

Functions
---------
- :func:`from_worldbank` — World Bank Indicators API v2 / Data360 rows.
- :func:`from_fred`       — FRED series observations (one or many series).
- :func:`from_sdmx`       — SDMX-JSON (OECD, Eurostat, IMF SDMX endpoints).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

JSONLike = Union[Mapping[str, Any], Sequence[Any], pd.DataFrame]


def _payload_kind(payload: Any) -> str:
    if isinstance(payload, pd.DataFrame):
        return "dataframe"
    if isinstance(payload, Mapping):
        return "mapping"
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return "sequence"
    return type(payload).__name__


def _stamp_attrs(
    df: pd.DataFrame,
    *,
    source: str,
    normalizer: str,
    payload_kind: str,
    shape: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Attach deterministic source metadata to a normalized data-MCP frame."""
    provenance: Dict[str, Any] = {
        "source": source,
        "source_type": "data_mcp_payload",
        "normalizer": normalizer,
        "payload_kind": payload_kind,
        "shape": shape,
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }
    if extra:
        provenance.update({k: v for k, v in extra.items() if v is not None})
    df.attrs["source"] = source
    df.attrs["provenance"] = provenance
    return df


# --------------------------------------------------------------------------- #
# World Bank
# --------------------------------------------------------------------------- #


def from_worldbank(
    payload: JSONLike,
    *,
    wide: bool = False,
    value_name: str = "value",
) -> pd.DataFrame:
    """Normalise a World Bank Indicators API payload to a tidy panel.

    Accepts any of the shapes a World Bank MCP / the v2 REST API returns:

    * the raw ``[metadata, rows]`` two-element list,
    * just the ``rows`` list of observation dicts,
    * a ``DataFrame`` already close to tidy.

    Each observation dict looks like::

        {"indicator": {"id": "NY.GDP.PCAP.KD", "value": "GDP per capita"},
         "country":   {"id": "US", "value": "United States"},
         "countryiso3code": "USA", "date": "2020", "value": 63027.7}

    Parameters
    ----------
    payload : list or dict or DataFrame
    wide : bool, default False
        If True, pivot indicators to columns indexed by (iso3, year) — handy
        when several indicators were fetched and you want one regression frame.
    value_name : str, default "value"
        Name of the value column in long form.

    Returns
    -------
    pandas.DataFrame
        Long form columns: ``country``, ``iso3``, ``indicator``,
        ``indicator_id``, ``year``, ``<value_name>``. ``df.attrs['source']``
        records the provenance. Wide form: one row per (iso3, year), one column
        per indicator.

    Examples
    --------
    >>> rows = [{"indicator": {"id": "NY.GDP", "value": "GDP"},
    ...          "country": {"id": "US", "value": "United States"},
    ...          "countryiso3code": "USA", "date": "2020", "value": 1.0}]
    >>> import statspai as sp
    >>> df = sp.from_worldbank(rows)
    >>> list(df.columns)
    ['country', 'iso3', 'indicator', 'indicator_id', 'year', 'value']
    >>> int(df.loc[0, 'year'])
    2020
    """
    kind = _payload_kind(payload)
    rows = _worldbank_rows(payload)
    if isinstance(rows, pd.DataFrame):
        long = rows.copy()
    else:
        recs: List[Dict[str, Any]] = []
        for r in rows:
            ind = r.get("indicator") or {}
            ctry = r.get("country") or {}
            recs.append(
                {
                    "country": ctry.get("value"),
                    "iso3": r.get("countryiso3code") or ctry.get("id"),
                    "indicator": ind.get("value"),
                    "indicator_id": ind.get("id"),
                    "year": _to_year(r.get("date")),
                    value_name: _to_float(r.get("value")),
                }
            )
        long = pd.DataFrame.from_records(recs)
    long = long.dropna(subset=["year"]).reset_index(drop=True)
    long["year"] = long["year"].astype("Int64").astype("int64")
    if not wide:
        return _stamp_attrs(
            long,
            source="worldbank",
            normalizer="from_worldbank",
            payload_kind=kind,
            shape="long",
            extra={
                "value_name": value_name,
                "indicator_ids": sorted(
                    str(x)
                    for x in long.get("indicator_id", pd.Series()).dropna().unique()
                ),
            },
        )
    wide_df = long.pivot_table(
        index=["iso3", "year"],
        columns="indicator_id",
        values=value_name,
        aggfunc="first",
    ).reset_index()
    wide_df.columns.name = None
    return _stamp_attrs(
        wide_df,
        source="worldbank",
        normalizer="from_worldbank",
        payload_kind=kind,
        shape="wide",
        extra={
            "value_name": value_name,
            "indicator_ids": sorted(
                str(x) for x in long.get("indicator_id", pd.Series()).dropna().unique()
            ),
        },
    )


def _worldbank_rows(payload: JSONLike) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, Mapping):
        # Some MCPs wrap rows in {"data": [...]} or {"rows": [...]}.
        for key in ("data", "rows", "observations", "results"):
            if key in payload and isinstance(payload[key], Sequence):
                return list(payload[key])  # type: ignore[arg-type]
        raise ValueError(
            "Unrecognised World Bank payload mapping; expected a 'data'/'rows' "
            f"key, got keys {list(payload.keys())}."
        )
    seq = list(payload)
    # Raw API returns [metadata_dict, rows_list]; unwrap to the rows list.
    if (
        len(seq) == 2
        and isinstance(seq[0], Mapping)
        and isinstance(seq[1], Sequence)
        and not isinstance(seq[1], (str, bytes))
    ):
        return list(seq[1])
    return seq  # already a list of row dicts


# --------------------------------------------------------------------------- #
# FRED
# --------------------------------------------------------------------------- #


def from_fred(
    payload: Union[JSONLike, Mapping[str, JSONLike]],
    *,
    series_id: Optional[str] = None,
) -> pd.DataFrame:
    """Normalise FRED series observations to a tidy time series.

    Accepts:

    * a single series' ``{"observations": [{"date", "value"}, ...]}`` dict,
    * just the observations list,
    * a mapping ``{series_id: observations}`` for several series, merged on
      ``date`` into a wide frame (one column per series).

    FRED's missing-value sentinel ``"."`` becomes ``NaN``; dates are parsed to
    ``datetime64``.

    Parameters
    ----------
    payload : dict or list or mapping of series_id -> observations
    series_id : str, optional
        Column name for the value series when a single series is passed
        (default ``"value"``).

    Returns
    -------
    pandas.DataFrame
        Columns ``date`` + one value column per series, sorted by date.

    Examples
    --------
    >>> obs = {"observations": [{"date": "2020-01-01", "value": "1.5"},
    ...                         {"date": "2020-02-01", "value": "."}]}
    >>> import statspai as sp
    >>> df = sp.from_fred(obs, series_id="cpi")
    >>> list(df.columns)
    ['date', 'cpi']
    >>> bool(df['cpi'].isna().iloc[1])
    True
    """
    kind = _payload_kind(payload)
    # Mapping of several series → merge on date.
    if (
        isinstance(payload, Mapping)
        and "observations" not in payload
        and not ({"date", "value"} & set(payload.keys()))
    ):
        merged: Optional[pd.DataFrame] = None
        for sid, obs in payload.items():
            one = _fred_one(obs, str(sid))
            merged = (
                one if merged is None else merged.merge(one, on="date", how="outer")
            )
        out = (
            merged.sort_values("date").reset_index(drop=True)
            if merged is not None
            else pd.DataFrame(columns=["date"])
        )
        return _stamp_attrs(
            out,
            source="fred",
            normalizer="from_fred",
            payload_kind="multi_series_mapping",
            shape="wide",
            extra={"series_ids": [str(sid) for sid in payload.keys()]},
        )

    out = _fred_one(payload, series_id or "value")
    return _stamp_attrs(
        out,
        source="fred",
        normalizer="from_fred",
        payload_kind=kind,
        shape="long",
        extra={"series_ids": [series_id or "value"]},
    )


def _fred_one(payload: JSONLike, col: str) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        obs: Sequence[Any] = payload.to_dict("records")
    elif isinstance(payload, Mapping):
        obs = list(payload.get("observations", []))  # type: ignore[arg-type]
    else:
        obs = list(payload)
    dates = [o.get("date") for o in obs]
    vals = [_to_float(o.get("value")) for o in obs]
    df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), col: vals})
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SDMX-JSON (OECD / Eurostat / IMF)
# --------------------------------------------------------------------------- #


def from_sdmx(payload: JSONLike, *, value_name: str = "value") -> pd.DataFrame:
    """Normalise an SDMX-JSON payload (OECD / Eurostat / IMF) to long form.

    SDMX-JSON encodes each observation by integer indices into per-dimension
    code lists. This expands those indices back into human-readable dimension
    columns plus a value column — the shape ``sp.detect_design`` can read.

    Accepts the SDMX-JSON 1.0 structure (``{"dataSets": [...],
    "structure": {"dimensions": {...}}}``) or a pre-tidied list of record
    dicts (returned as a DataFrame unchanged).

    Parameters
    ----------
    payload : dict or list or DataFrame
    value_name : str, default "value"

    Returns
    -------
    pandas.DataFrame
        One column per SDMX dimension (e.g. ``LOCATION``, ``TIME_PERIOD``) plus
        ``<value_name>``. ``df.attrs['source'] == 'sdmx'``.

    Examples
    --------
    >>> payload = {
    ...   "dataSets": [{"series": {"0:0": {"observations": {"0": [3.2]}}}}],
    ...   "structure": {"dimensions": {
    ...      "series": [
    ...        {"id": "LOCATION", "values": [{"id": "USA", "name": "USA"}]},
    ...        {"id": "SUBJECT", "values": [{"id": "UNR", "name": "Unemp"}]}],
    ...      "observation": [
    ...        {"id": "TIME_PERIOD", "values": [{"id": "2020", "name": "2020"}]}]
    ...   }}}
    >>> import statspai as sp
    >>> df = sp.from_sdmx(payload)
    >>> df.loc[0, "LOCATION"], df.loc[0, "TIME_PERIOD"], df.loc[0, "value"]
    ('USA', '2020', 3.2)
    """
    kind = _payload_kind(payload)
    if isinstance(payload, pd.DataFrame):
        out = payload.copy()
        return _stamp_attrs(
            out,
            source="sdmx",
            normalizer="from_sdmx",
            payload_kind=kind,
            shape="long",
        )
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        out = pd.DataFrame.from_records(list(payload))
        return _stamp_attrs(
            out,
            source="sdmx",
            normalizer="from_sdmx",
            payload_kind=kind,
            shape="long",
        )
    if not isinstance(payload, Mapping) or "dataSets" not in payload:
        raise ValueError(
            "Unrecognised SDMX payload; expected SDMX-JSON with 'dataSets' / "
            "'structure', a record list, or a DataFrame."
        )

    structure = payload.get("structure", {})
    dims = structure.get("dimensions", {})
    series_dims = dims.get("series", [])
    obs_dims = dims.get("observation", [])
    series_codes = [[v.get("id") for v in d.get("values", [])] for d in series_dims]
    series_names = [d.get("id") for d in series_dims]
    obs_codes = [[v.get("id") for v in d.get("values", [])] for d in obs_dims]
    obs_names = [d.get("id") for d in obs_dims]

    recs: List[Dict[str, Any]] = []
    for ds in payload.get("dataSets", []):
        for skey, sval in ds.get("series", {}).items():
            s_idx = [int(i) for i in skey.split(":")]
            base = {
                series_names[i]: series_codes[i][s_idx[i]]
                for i in range(len(s_idx))
                if i < len(series_names) and s_idx[i] < len(series_codes[i])
            }
            for okey, ovals in sval.get("observations", {}).items():
                rec = dict(base)
                o_idx = [int(i) for i in okey.split(":")]
                for i, oi in enumerate(o_idx):
                    if i < len(obs_names) and oi < len(obs_codes[i]):
                        rec[obs_names[i]] = obs_codes[i][oi]
                rec[value_name] = _to_float(ovals[0] if ovals else None)
                recs.append(rec)
    out = pd.DataFrame.from_records(recs)
    return _stamp_attrs(
        out,
        source="sdmx",
        normalizer="from_sdmx",
        payload_kind=kind,
        shape="long",
        extra={
            "value_name": value_name,
            "dimensions": [str(x) for x in series_names + obs_names if x],
        },
    )


# --------------------------------------------------------------------------- #
# Shared coercion helpers
# --------------------------------------------------------------------------- #


def _to_float(x: Any) -> float:
    if x is None or x == "." or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _to_year(x: Any) -> Any:
    if x is None:
        return np.nan
    s = str(x)
    # World Bank dates are usually "2020"; quarterly/monthly come as "2020Q1".
    head = s[:4]
    try:
        return int(head)
    except ValueError:
        return np.nan
