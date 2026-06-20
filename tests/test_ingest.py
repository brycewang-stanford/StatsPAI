"""Tests for the data-source ingestion normalisers.

Synthetic payloads mirror the real shapes returned by the World Bank, FRED and
OECD/Eurostat (SDMX-JSON) MCP servers / REST APIs. No network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.datasets.ingest import from_fred, from_sdmx, from_worldbank

# --------------------------------------------------------------------------- #
# World Bank
# --------------------------------------------------------------------------- #


def _wb_rows():
    mk = lambda iso, cname, yr, val: {  # noqa: E731
        "indicator": {"id": "NY.GDP.PCAP.KD", "value": "GDP per capita"},
        "country": {"id": iso[:2], "value": cname},
        "countryiso3code": iso,
        "date": str(yr),
        "value": val,
    }
    return [
        mk("USA", "United States", 2020, 63027.7),
        mk("USA", "United States", 2019, 62823.3),
        mk("CHN", "China", 2020, 10408.7),
    ]


class TestWorldBank:
    def test_raw_meta_rows_unwrapped(self):
        df = from_worldbank([{"page": 1}, _wb_rows()])
        assert list(df.columns) == [
            "country",
            "iso3",
            "indicator",
            "indicator_id",
            "year",
            "value",
        ]
        assert len(df) == 3
        assert df["year"].dtype.kind in "iu"

    def test_plain_rows_list(self):
        df = from_worldbank(_wb_rows())
        assert df.loc[df["iso3"] == "CHN", "value"].iloc[0] == pytest.approx(10408.7)

    def test_mapping_with_data_key(self):
        df = from_worldbank({"data": _wb_rows()})
        assert len(df) == 3

    def test_wide_pivot(self):
        df = from_worldbank([{"page": 1}, _wb_rows()], wide=True)
        assert "NY.GDP.PCAP.KD" in df.columns
        # USA appears in two years, China in one → 3 (iso3, year) rows.
        assert len(df) == 3
        assert {"iso3", "year"} <= set(df.columns)
        assert df.attrs["provenance"]["shape"] == "wide"

    def test_missing_value_becomes_nan(self):
        rows = _wb_rows()
        rows[0]["value"] = None
        df = from_worldbank(rows)
        assert df["value"].isna().any()

    def test_attrs_source(self):
        df = from_worldbank(_wb_rows())
        assert df.attrs.get("source") == "worldbank"
        prov = df.attrs["provenance"]
        assert prov["source_type"] == "data_mcp_payload"
        assert prov["normalizer"] == "from_worldbank"
        assert prov["shape"] == "long"
        assert prov["indicator_ids"] == ["NY.GDP.PCAP.KD"]

    def test_bad_mapping_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            from_worldbank({"unexpected": 1})


# --------------------------------------------------------------------------- #
# FRED
# --------------------------------------------------------------------------- #


class TestFred:
    def test_observations_dict(self):
        obs = {
            "observations": [
                {"date": "2020-01-01", "value": "1.5"},
                {"date": "2020-02-01", "value": "."},
            ]
        }
        df = from_fred(obs, series_id="cpi")
        assert list(df.columns) == ["date", "cpi"]
        assert df["cpi"].isna().iloc[1]  # "." -> NaN
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_plain_list(self):
        df = from_fred([{"date": "2020-01-01", "value": "2.0"}])
        assert df["value"].iloc[0] == pytest.approx(2.0)

    def test_multi_series_merge(self):
        payload = {
            "CPI": [
                {"date": "2020-01-01", "value": "1.5"},
                {"date": "2020-02-01", "value": "1.6"},
            ],
            "UNRATE": [
                {"date": "2020-01-01", "value": "3.6"},
                {"date": "2020-03-01", "value": "4.4"},
            ],
        }
        df = from_fred(payload)
        assert {"date", "CPI", "UNRATE"} == set(df.columns)
        # Outer merge → 3 distinct dates, with NaNs where a series is missing.
        assert len(df) == 3
        assert df["UNRATE"].isna().sum() == 1
        assert df.attrs["provenance"]["series_ids"] == ["CPI", "UNRATE"]
        assert df.attrs["provenance"]["shape"] == "wide"

    def test_sorted_by_date(self):
        obs = [
            {"date": "2020-03-01", "value": "3"},
            {"date": "2020-01-01", "value": "1"},
        ]
        df = from_fred(obs)
        assert df["date"].is_monotonic_increasing

    def test_attrs_source(self):
        df = from_fred([{"date": "2020-01-01", "value": "1"}])
        assert df.attrs.get("source") == "fred"
        assert df.attrs["provenance"]["series_ids"] == ["value"]
        assert df.attrs["provenance"]["normalizer"] == "from_fred"


# --------------------------------------------------------------------------- #
# SDMX-JSON
# --------------------------------------------------------------------------- #


def _sdmx_payload():
    return {
        "dataSets": [
            {
                "series": {
                    "0:0": {"observations": {"0": [3.2], "1": [3.4]}},
                    "1:0": {"observations": {"0": [5.0]}},
                }
            }
        ],
        "structure": {
            "dimensions": {
                "series": [
                    {
                        "id": "LOCATION",
                        "values": [{"id": "USA"}, {"id": "DEU"}],
                    },
                    {"id": "SUBJECT", "values": [{"id": "UNR"}]},
                ],
                "observation": [
                    {
                        "id": "TIME_PERIOD",
                        "values": [{"id": "2019"}, {"id": "2020"}],
                    }
                ],
            }
        },
    }


class TestSdmx:
    def test_index_expansion(self):
        df = from_sdmx(_sdmx_payload())
        assert {"LOCATION", "SUBJECT", "TIME_PERIOD", "value"} == set(df.columns)
        usa20 = df[(df["LOCATION"] == "USA") & (df["TIME_PERIOD"] == "2020")]
        assert usa20["value"].iloc[0] == pytest.approx(3.4)
        assert df.attrs["provenance"]["dimensions"] == [
            "LOCATION",
            "SUBJECT",
            "TIME_PERIOD",
        ]

    def test_ragged_series(self):
        # USA has 2 periods, DEU has 1 → 3 observations total.
        df = from_sdmx(_sdmx_payload())
        assert len(df) == 3

    def test_record_list_passthrough(self):
        recs = [{"LOCATION": "USA", "value": 1.0}]
        df = from_sdmx(recs)
        assert df.loc[0, "LOCATION"] == "USA"

    def test_dataframe_passthrough(self):
        src = pd.DataFrame({"LOCATION": ["USA"], "value": [1.0]})
        df = from_sdmx(src)
        assert df.attrs.get("source") == "sdmx"

    def test_bad_payload_raises(self):
        with pytest.raises(ValueError, match="Unrecognised SDMX"):
            from_sdmx({"not": "sdmx"})


# --------------------------------------------------------------------------- #
# Wiring + downstream integration
# --------------------------------------------------------------------------- #


class TestWiringAndIntegration:
    def test_top_level_exports(self):
        assert callable(sp.from_worldbank)
        assert callable(sp.from_fred)
        assert callable(sp.from_sdmx)

    def test_registered(self):
        fns = sp.list_functions()
        assert {"from_worldbank", "from_fred", "from_sdmx"} <= set(fns)

    def test_normalised_frame_feeds_a_regression(self):
        # End-to-end: a wide World Bank frame regresses without further fuss.
        rng = np.random.default_rng(0)
        rows = []
        for i in range(40):
            gdp = 10000 + 500 * i + rng.normal(0, 200)
            life = 60 + 0.0005 * gdp + rng.normal(0, 1)
            for ind, val in (("gdp", gdp), ("life", life)):
                rows.append(
                    {
                        "indicator": {"id": ind, "value": ind},
                        "country": {"id": f"C{i}", "value": f"Country {i}"},
                        "countryiso3code": f"C{i:02d}",
                        "date": "2020",
                        "value": val,
                    }
                )
        wide = sp.from_worldbank(rows, wide=True)
        res = sp.regress("life ~ gdp", data=wide)
        assert "gdp" in list(res.tidy()["term"])
