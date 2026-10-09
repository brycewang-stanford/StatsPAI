"""Coverage gaps (batch B) for statspai.did.report (``sp.cs_report``).

Optional-dependency fallbacks of the exporters, the evidence lines of the
text report, the ignored-argument warning for a pre-fitted result, and the
degradation contract: a best-effort step that fails is warned about and
listed in ``report.degradations`` while the estimate itself is unchanged.
"""

import dataclasses
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
import statspai.did.report  # noqa: E402,F401
from statspai.workflow._degradation import WorkflowDegradedWarning  # noqa: E402

R = sys.modules["statspai.did.report"]
KW = dict(y="y", g="g", t="t", i="i", n_boot=30, random_state=1, verbose=False)


def make_panel(seed=0, cohorts=(4, 6, 0), n_per=25, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            ufe = rng.normal()
            xv = rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                yv = ufe + 0.3 * t + te + rng.normal() * 0.5
                rows.append({"i": uid, "t": t, "y": yv, "g": g, "x1": xv})
            uid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def report(panel):
    return sp.cs_report(panel, x=["x1"], **KW)


# ----------------------------------------------------------------------
# text report: evidence lines
# ----------------------------------------------------------------------
def test_text_reports_balanced_covariates(report):
    # x1 is drawn independently of the cohort, so nothing is flagged.
    assert report.balance is not None and report.balance.flagged == []
    txt = report.to_text()
    assert "Covariate balance: none over |norm.diff| >" in txt
    assert "DEGRADED STEPS" not in txt


def test_text_flags_disagreeing_covariate_strategies(report):
    cmp_df = report.estimator_comparison
    assert cmp_df["estimator"].tolist() == ["reg", "ipw", "dr"]
    assert "! spread across strategies" not in report.to_text()
    apart = cmp_df.copy()
    apart.loc[apart["estimator"] == "ipw", "att"] += 10 * cmp_df["se"].median()
    txt = dataclasses.replace(report, estimator_comparison=apart).to_text()
    spread = float(apart["att"].max() - apart["att"].min())
    assert f"! spread across strategies ({spread:.4f}) exceeds" in txt
    assert "Prefer DR" in txt


# ----------------------------------------------------------------------
# exporters without their optional dependency
# ----------------------------------------------------------------------
def test_markdown_falls_back_to_plain_tables_without_tabulate(report, monkeypatch):
    with_tab = report.to_markdown()
    assert "|" in with_tab
    monkeypatch.setitem(sys.modules, "tabulate", None)
    plain = report.to_markdown()
    assert "### Event study (dynamic aggregation)" in plain
    assert "relative_time" in plain and "|" not in plain


def test_excel_uses_xlsxwriter_when_openpyxl_is_missing(report, monkeypatch, tmp_path):
    pytest.importorskip("xlsxwriter")
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    path = report.to_excel(tmp_path / "r.xlsx", engine="xlsxwriter")
    assert os.path.getsize(path) > 0
    monkeypatch.undo()
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"Summary", "Dynamic", "Group", "Calendar"} <= set(sheets)
    assert len(sheets["Dynamic"]) == len(report.dynamic)


def test_bundle_skips_formats_whose_dependency_is_missing(
    report, monkeypatch, tmp_path
):
    for mod in ("openpyxl", "xlsxwriter", "matplotlib"):
        monkeypatch.setitem(sys.modules, mod, None)
    written = R._save_report_bundle(report, str(tmp_path / "out" / "cs"))
    assert set(written) == {"txt", "md", "tex"}
    assert all(os.path.getsize(p) > 0 for p in written.values())
    assert not (tmp_path / "out" / "cs.xlsx").exists()
    assert not (tmp_path / "out" / "cs.png").exists()


def test_bundle_selects_agg_backend_when_pyplot_not_yet_imported(
    report, monkeypatch, tmp_path
):
    import matplotlib.pyplot as plt

    # keep the package attribute and sys.modules pointing at one module
    monkeypatch.setattr(matplotlib, "pyplot", plt)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot")
    written = R._save_report_bundle(report, str(tmp_path / "cs"))
    monkeypatch.undo()
    plt.close("all")
    assert matplotlib.get_backend().lower() == "agg"
    assert set(written) == {"txt", "md", "tex", "xlsx", "png"}
    assert os.path.getsize(written["png"]) > 0


# ----------------------------------------------------------------------
# pre-fitted result + estimation-time arguments
# ----------------------------------------------------------------------
def test_prefitted_result_warns_about_every_ignored_argument(panel):
    cs = sp.callaway_santanna(panel, y="y", g="g", t="t", i="i")
    with pytest.warns(UserWarning, match="those arguments are ignored") as rec:
        rpt = sp.cs_report(
            cs, y="y", g="g", t="t", i="i", x=["x1"], n_boot=30, verbose=False
        )
    msg = str(rec[0].message)
    for piece in ("y='y'", "g='g'", "t='t'", "i='i'", "x=['x1']"):
        assert piece in msg
    # the pre-fitted (no-covariate) estimate is used as-is
    plain = sp.cs_report(cs, n_boot=30, verbose=False)
    assert rpt.overall["estimate"] == plain.overall["estimate"]
    assert rpt.balance is None and rpt.estimator_comparison.empty


# ----------------------------------------------------------------------
# degradation contract
# ----------------------------------------------------------------------
def test_failed_evidence_steps_are_recorded_not_swallowed(panel, report, monkeypatch):
    import statspai.did.balance  # noqa: F401
    import statspai.did.functional_form  # noqa: F401

    def boom(*args, **kwargs):
        raise RuntimeError("evidence step exploded")

    monkeypatch.setattr(sys.modules["statspai.did.balance"], "did_balance", boom)
    monkeypatch.setattr(
        sys.modules["statspai.did.functional_form"], "functional_form_test", boom
    )
    real_cs = R.callaway_santanna

    def cs_without_ipw(*args, **kwargs):
        if kwargs.get("estimator") == "ipw":
            raise RuntimeError("ipw exploded")
        return real_cs(*args, **kwargs)

    monkeypatch.setattr(R, "callaway_santanna", cs_without_ipw)

    with pytest.warns(WorkflowDegradedWarning) as rec:
        rpt = sp.cs_report(panel, x=["x1"], **KW)
    assert len([w for w in rec if w.category is WorkflowDegradedWarning]) == 3

    sections = [d["section"] for d in rpt.degradations]
    assert sections == [
        "covariate balance (step 2)",
        "functional-form test (step 2)",
        "estimator triangulation (ipw)",
    ]
    assert {d["error_type"] for d in rpt.degradations} == {"RuntimeError"}
    assert rpt.balance is None and rpt.functional_form == {}
    assert rpt.estimator_comparison["estimator"].tolist() == ["reg", "dr"]
    # the headline estimate does not depend on the failed evidence steps
    assert rpt.overall["estimate"] == pytest.approx(report.overall["estimate"])

    txt = rpt.to_text()
    assert "DEGRADED STEPS (ran but did not complete):" in txt
    assert "- covariate balance (step 2): RuntimeError evidence step exploded" in txt
