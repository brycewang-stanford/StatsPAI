"""Static guards for optional exact-parity dependencies in CI."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_canonical_ci_installs_all_exact_parity_extras():
    workflow = _read(".github/workflows/ci-cd.yml")

    assert "Install exact-parity extras" in workflow
    assert "matrix.os == 'ubuntu-latest' && matrix.python-version == '3.10'" in workflow
    assert 'pip install -e ".[parity,rd-cct]"' in workflow
    assert "test_dml_python_parity.py run its 4 sp.dml-vs-doubleml pins" in workflow
    assert "test_published_replications.py run the bwselect='cct' RD pins" in workflow


def test_optional_dependency_names_match_pyproject_extras_and_skips():
    pyproject = _read("pyproject.toml")
    dml_test = _read("tests/external_parity/test_dml_python_parity.py")
    published_test = _read("tests/external_parity/test_published_replications.py")

    assert "parity = [" in pyproject
    assert '"doubleml>=0.7"' in pyproject
    assert "rd-cct = [" in pyproject
    assert '"rdrobust>=2.0"' in pyproject

    assert 'pytest.importorskip("doubleml")' in dml_test
    assert "all four DoubleML model classes" in dml_test

    assert "pytest.importorskip(\n            'rdrobust'," in published_test
    assert "extras [rd-cct] are not installed" in published_test
    assert r"pip install statspai\[rd-cct\]" in published_test
