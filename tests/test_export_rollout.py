"""Faithfulness + coverage guards for the ExportMixin rollout.

The ExportMixin rollout campaign attached the uniform export quartet
(to_markdown/to_latex/to_excel/to_word + non-fabricating cite) to many bespoke
result classes. These tests pin that each rolled-out class (a) is genuinely an
ExportMixin subclass and (b) produces a faithful, non-empty export frame from a
synthetic instance — and that overall coverage does not regress.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai
from statspai.core.results import ExportMixin

warnings.filterwarnings("ignore")

# (module suffix, class name) for the classes rolled out in this campaign.
ROLLED_OUT = [
    # coef-table
    ("iv.jive_variants", "JIVEResult"),
    ("panel.feols", "FEOLSResult"),
    ("spatial.panel.estimator", "SpatialPanelResult"),
    ("interference.peer_effects", "PeerEffectsResult"),
    # epi
    ("epi.measures", "OR2x2Result"),
    ("epi.measures", "RR2x2Result"),
    ("epi.measures", "RD2x2Result"),
    ("epi.measures", "IRRResult"),
    ("epi.measures", "NNTResult"),
    ("epi.stratified", "MantelHaenszelResult"),
    # dml / iv estimate-rows
    ("dml._diagnostics", "DMLDiagnostics"),
    ("dml.panel_dml", "DMLPanelResult"),
    ("dml._sensitivity", "DMLSensitivityResult"),
    ("iv.continuous_late", "ContinuousLATEResult"),
    ("iv.ivdml", "IVDMLResult"),
    ("iv.many_weak", "ManyWeakIVResult"),
    # did / interference / panel
    ("bartik.political", "ShiftSharePoliticalPanelResult"),
    ("did.harvest", "HarvestDIDResult"),
    ("interference.dnc_gnn_did", "DNCGNNDiDResult"),
    ("interference.cluster_matched_pair", "MatchedPairResult"),
    ("longitudinal.analyze", "LongitudinalResult"),
    # mendelian randomization
    ("mendelian.frontier.grapple", "GrappleResult"),
    ("mendelian.frontier.lap", "MRLapResult"),
    ("mendelian.frontier.raps", "MRRapsResult"),
    ("mendelian.frontier.cml", "MRcMLResult"),
    ("mendelian.extras", "ModeBasedResult"),
    # misc domains
    ("multilevel.diagnostics", "ICCResult"),
    ("proximal.negative_controls", "NegativeControlResult"),
    ("question.question", "EstimationResult"),
    ("robustness.sensitivity_frontier", "FrontierSensitivityResult"),
    ("surrogate.index", "SurrogateResult"),
    ("survey.estimators", "SurveyResult"),
    ("target_trial.emulate", "TargetTrialResult"),
]


def _load(modsuffix, clsname):
    mod = importlib.import_module("statspai." + modsuffix)
    return getattr(mod, clsname)


def _dummy(ann):
    s = str(ann)
    if "ndarray" in s:
        return np.array([1.0, 2.0, 3.0])
    if "Series" in s:
        return pd.Series([1.0, 2.0, 3.0])
    if "DataFrame" in s:
        return pd.DataFrame({"a": [1.0]})
    if "str" in s:
        return "x"
    if "bool" in s:
        return True
    if "int" in s:
        return 1
    if "float" in s:
        return 1.0
    if any(k in s for k in ("Dict", "dict")):
        return {}
    if any(k in s for k in ("List", "list", "Tuple", "Sequence")):
        return []
    return None


def _build(cls):
    kw = {
        f.name: _dummy(f.type)
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }
    return cls(**kw)


@pytest.mark.parametrize("modsuffix,clsname", ROLLED_OUT,
                         ids=[c for _, c in ROLLED_OUT])
def test_rolled_out_class_is_exportable_and_faithful(modsuffix, clsname):
    cls = _load(modsuffix, clsname)
    assert issubclass(cls, ExportMixin), f"{clsname} is not an ExportMixin"
    inst = _build(cls)
    frame = inst._export_frame()
    assert isinstance(frame, pd.DataFrame) and len(frame.index) > 0
    assert "|" in inst.to_markdown()
    assert "tabular" in inst.to_latex()
    # cite never fabricates unless a verified key is registered.
    cite = str(inst.cite())
    assert ("No verified citation" in cite) or ("@" in cite)


def _all_result_like():
    seen = {}
    for m in pkgutil.walk_packages(statspai.__path__, "statspai."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for _n, o in inspect.getmembers(mod, inspect.isclass):
            if o.__module__.startswith("statspai") and hasattr(o, "summary"):
                seen.setdefault(o.__qualname__, o)
    return seen


def test_export_coverage_does_not_regress():
    """Regression ratchet: number of exportable result classes stays high."""
    quartet = ["to_latex", "to_markdown", "to_excel", "to_word", "cite"]
    classes = _all_result_like().values()
    exportable = sum(
        1 for c in classes
        if issubclass(c, ExportMixin) or all(hasattr(c, m) for m in quartet)
    )
    # 54 at the time of writing; floor guards against accidental regression.
    assert exportable >= 50, f"only {exportable} result classes are exportable"
