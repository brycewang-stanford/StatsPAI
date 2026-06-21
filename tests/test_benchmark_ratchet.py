"""Tests for the benchmark regression ratchet."""

from __future__ import annotations

import json

import pytest

from scripts import benchmark_ratchet as br


def _bench_doc(sp_time: float):
    return {
        "meta": {"statspai_version": "test", "mode": "quick", "platform": "unit"},
        "regression": {
            "rows": [
                {
                    "n": 1000,
                    "sp_regress_s": sp_time,
                    "statsmodels_s": 0.001,
                }
            ]
        },
    }


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_compare_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        br.compare(
            _bench_doc(0.01),
            _bench_doc(0.01),
            threshold=threshold,
            min_seconds=0.0,
        )


@pytest.mark.parametrize("min_seconds", [-1.0, float("nan"), float("inf")])
def test_compare_rejects_invalid_min_seconds(min_seconds):
    with pytest.raises(ValueError, match="min_seconds"):
        br.compare(
            _bench_doc(0.01),
            _bench_doc(0.01),
            threshold=1.5,
            min_seconds=min_seconds,
        )


def test_compare_flags_stats_pai_regression():
    failures, lines = br.compare(
        _bench_doc(0.01),
        _bench_doc(0.03),
        threshold=1.5,
        min_seconds=0.0,
    )
    assert failures
    assert any("3.00x" in line for line in lines)


def test_main_rejects_invalid_cli_threshold(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_bench_doc(0.01)), encoding="utf-8")
    current.write_text(json.dumps(_bench_doc(0.01)), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_ratchet.py",
            "--check",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--threshold",
            "nan",
        ],
    )

    assert br.main() == 2
    assert "threshold" in capsys.readouterr().out
