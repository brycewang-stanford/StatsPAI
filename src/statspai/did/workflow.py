"""``sp.did_workflow``: a difference-in-differences study, step by step.

A modern DiD analysis is a sequence, not a regression: declare the target
and the comparison group, audit the design, diagnose two-way fixed
effects, estimate with a heterogeneity-robust estimator, report the
event-study path, test pre-trends *and the power of that test*, bound
violations of parallel trends, vary the choices that were choices, and
write down what was estimated (Baker et al. 2026, section 6; Roth et al.
2023).  :func:`did_workflow` runs that sequence on one panel and returns a
:class:`DiDWorkflow` whose nine steps each carry a status, a finding with
its numbers, and what to do next.

It adds no estimator.  Every number is the output of the public function
that step names (``sp.callaway_santanna``, ``sp.aggte``,
``sp.uniform_bands``, ``sp.pretrends_test``, ``sp.honest_did``, ...), all
of them fitted on the *same* estimator, so the pre-trend test and the
sensitivity analysis describe the estimate that is reported.

References
----------
Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026)
[@baker2026difference]; Roth, Sant'Anna, Bilinski and Poe (2023)
[@roth2023whats]; Roth (2022) [@roth2022pretest]; Rambachan and Roth
(2023) [@rambachan2023more].
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .._export_path import checks_export_path
from .._result_serialize import ResultProtocolMixin
from ..core._call_record import records_call
from ..core._result_access import ResultAccessorUnavailable
from ..exceptions import MethodIncompatibility
from . import _workflow_render as _render
from ._workflow_design import run_declare, run_design, run_twfe
from ._workflow_estimate import run_estimate, run_event_study
from ._workflow_evidence import run_pretrends, run_sensitivity
from ._workflow_options import check_options
from ._workflow_robustness import run_report, run_robustness
from ._workflow_spec import (
    STEP_KEYS,
    DiDStep,
    WorkflowInputs,
    blank_steps,
    resolve_inputs,
    worst_status,
)

__all__ = ["DiDWorkflow", "did_workflow"]
_FN = "sp.did_workflow"

RUNNERS: Dict[str, Callable[["DiDWorkflow", DiDStep], None]] = {
    "declare": run_declare,
    "design": run_design,
    "twfe": run_twfe,
    "estimate": run_estimate,
    "event_study": run_event_study,
    "pretrends": run_pretrends,
    "sensitivity": run_sensitivity,
    "robustness": run_robustness,
    "report": run_report,
}
#: Steps that fit or read the estimate; not run when the design audit fails.
_NEEDS_DESIGN = (
    "twfe",
    "estimate",
    "event_study",
    "pretrends",
    "sensitivity",
    "robustness",
)


class DiDWorkflow(ResultProtocolMixin):
    """A DiD study as nine graded steps.

    Returned by :func:`did_workflow`.  ``wf.steps`` is the protocol as a
    table (status, finding, action); ``wf.summary()`` prints it;
    ``wf.step('pretrends')`` gives one step with its evidence; and the
    results of the steps are available directly as ``wf.estimate`` /
    ``wf.se`` / ``wf.ci``, ``wf.event_study``, ``wf.sensitivity``,
    ``wf.robustness`` and ``wf.report()``.

    The workflow can be advanced one step at a time: build it with
    ``through=None``, then call ``wf.advance()`` (the next step) or
    ``wf.run('design')`` (up to and including a step).

    Attributes
    ----------
    verdict : str
        ``fail`` if any step failed, else ``caution`` if any step left a
        caveat, else ``pass``.
    degradations : list of dict
        Optional checks that could not be computed, with the reason.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=0)
    >>> wf = sp.did_workflow(df, y="y", id="unit", time="time",
    ...                      cohort="first_treat", through="design")
    >>> wf.steps["status"].tolist()[:2] == [wf.step("declare").status,
    ...                                    wf.step("design").status]
    True
    >>> wf.pending[0]
    'twfe'
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "baker2026difference",
        "roth2023whats",
    )

    def __init__(
        self,
        inputs: WorkflowInputs,
        options: Dict[str, Any],
        explicit: Set[str],
    ) -> None:
        self._inputs = inputs
        self._options = options
        self._explicit = explicit
        self._steps: List[DiDStep] = blank_steps()
        self._facts: Optional[Dict[str, Any]] = None
        self._blocked: Optional[str] = None
        self._twfe: Optional[float] = None
        self._fit: Any = None
        self._es_result: Any = None
        self._headline: Optional[Dict[str, Any]] = None
        self._path: Optional[pd.DataFrame] = None
        self._sensitivity: Optional[Dict[str, Any]] = None
        self._robustness: Optional[pd.DataFrame] = None
        self._emitted: Set[Tuple[Any, str]] = set()
        self.degradations: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #
    @property
    def pending(self) -> List[str]:
        """Keys of the steps not run yet, in order."""
        return [s.key for s in self._steps if s.status == "pending"]

    def step(self, key: str) -> DiDStep:
        """One step of the protocol, by key (``'pretrends'``) or number."""
        for s in self._steps:
            if s.key == key or s.number == key:
                return s
        raise MethodIncompatibility(
            f"{_FN}: no step {key!r}.",
            recovery_hint=f"Steps are {', '.join(STEP_KEYS)} (or 1-9).",
            diagnostics={"step": repr(key), "valid": list(STEP_KEYS)},
        )

    def __getitem__(self, key: str) -> DiDStep:
        return self.step(key)

    def _optional(self, step: DiDStep, label: str, fn: Callable[[], Any]) -> Any:
        """Run a supporting check; record (do not hide) a failure."""
        from ..workflow._degradation import record_degradation

        try:
            return fn()
        except Exception as exc:  # recorded and warned, never silent
            record_degradation(
                self,
                section=f"did_workflow.{step.key}.{label}",
                exc=exc,
                detail=f"step {step.number} ({step.key}): {label} was not computed",
            )
            return None

    def _run_step(self, step: DiDStep) -> None:
        skip = self._options["skip"]
        if self._blocked and step.key in _NEEDS_DESIGN:
            step.status = "skipped"
            step.finding = f"Not run: {self._blocked}."
            return
        if step.key in skip:
            step.status = "skipped"
            step.finding = f"Skipped on request (skip={step.key!r})."
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RUNNERS[step.key](self, step)
        for w in caught:
            message = str(w.message)
            if issubclass(w.category, UserWarning) and message not in step.warnings:
                step.warnings.append(message)
            seen = (w.category, message)
            if seen not in self._emitted:
                self._emitted.add(seen)
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    def advance(self) -> DiDStep:
        """Run the next pending step and return it."""
        for s in self._steps:
            if s.status == "pending":
                self._run_step(s)
                return s
        raise MethodIncompatibility(
            f"{_FN}: every step has run.",
            recovery_hint="Read wf.steps or wf.summary().",
        )

    def run(self, through: str = "report") -> "DiDWorkflow":
        """Run the pending steps up to and including ``through``."""
        last = self.step(through).number
        for s in self._steps:
            if s.number > last:
                break
            if s.status == "pending":
                self._run_step(s)
        return self

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> pd.DataFrame:
        """The protocol as a table: one row per step."""
        return pd.DataFrame(
            [
                {
                    "step": s.number,
                    "key": s.key,
                    "title": s.title,
                    "status": s.status,
                    "finding": s.finding,
                    "action": s.action,
                }
                for s in self._steps
            ]
        )

    @property
    def verdict(self) -> str:
        """Worst status across the steps that have run (``pending`` if none)."""
        ran = [s.status for s in self._steps if s.status != "pending"]
        return worst_status(ran) if ran else "pending"

    def _need(self, value: Any, step_key: str, what: str) -> Any:
        if value is None:
            status = self.step(step_key).status
            raise ResultAccessorUnavailable(
                f"{_FN}: {what} is not available because step "
                f"{step_key!r} is {status}.",
                recovery_hint=(
                    f"wf.run({step_key!r})"
                    if status == "pending"
                    else f"See wf.step({step_key!r}).finding for the reason."
                ),
                diagnostics={"step": step_key, "status": status},
            )
        return value

    @property
    def headline(self) -> Dict[str, Any]:
        """The declared target: estimate, se, interval, p-value, estimand."""
        return dict(self._need(self._headline, "estimate", "the estimate"))

    @property
    def estimate(self) -> float:
        return float(self.headline["estimate"])

    @property
    def se(self) -> float:
        return float(self.headline["se"])

    @property
    def ci(self) -> Tuple[float, float]:
        h = self.headline
        return (float(h["ci_lower"]), float(h["ci_upper"]))

    @property
    def pvalue(self) -> float:
        return float(self.headline["pvalue"])

    @property
    def estimand(self) -> str:
        return str(self.headline["estimand"])

    @property
    def fit(self) -> Any:
        """The estimator's own result (group-time effects for ``'cs'``)."""
        return self._need(self._fit, "estimate", "the fitted result")

    @property
    def event_study_result(self) -> Any:
        """The result object that carries the event-study path and its
        joint covariance; pass it to ``sp.honest_did`` / ``sp.uniform_bands``."""
        return self._need(self._es_result, "estimate", "the event-study result")

    @property
    def event_study(self) -> pd.DataFrame:
        """The path with pointwise and simultaneous intervals."""
        return self._need(self._path, "event_study", "the event-study path").copy()

    @property
    def sensitivity(self) -> Dict[str, Any]:
        """Honest-DiD breakdown values and the relative-magnitude grid."""
        return dict(self._need(self._sensitivity, "sensitivity", "the sensitivity"))

    @property
    def robustness(self) -> pd.DataFrame:
        """The headline next to every alternative specification."""
        return self._need(self._robustness, "robustness", "the robustness table").copy()

    @property
    def design(self) -> Dict[str, Any]:
        """Counts from the design audit (units, cohorts, clusters, ...)."""
        step = self.step("design")
        return dict(self._need(step.evidence or None, "design", "the design audit"))

    @property
    def checklist(self) -> pd.DataFrame:
        """What the write-up has to state, item by item."""
        step = self.step("report")
        return self._need(
            step.evidence.get("checklist"), "report", "the reporting checklist"
        ).copy()

    def report(self) -> str:
        """The reporting statement: what was estimated, in one paragraph."""
        step = self.step("report")
        return str(
            self._need(step.evidence.get("statement"), "report", "the statement")
        )

    def cite(self, format: str = "keys") -> Any:
        """``paper.bib`` keys of the methods the executed steps used.

        Resolve them with ``sp.bibtex(keys)``.
        """
        keys: List[str] = []
        for s in self._steps:
            if s.status in ("pending", "skipped"):
                continue
            for key in s.references:
                if key not in keys:
                    keys.append(key)
        if format == "json":
            return {
                "citation_keys": keys,
                "source": "paper.bib",
                "resolve_with": "sp.bibtex(keys=[...])",
            }
        if format != "keys":
            raise ValueError(f"format must be 'keys' or 'json'; got {format!r}")
        return "\n".join(keys)

    def violations(self) -> List[Dict[str, Any]]:
        """Steps that failed or left a caveat, in the result-contract shape."""
        out: List[Dict[str, Any]] = []
        for s in self._steps:
            if s.status not in ("caution", "fail"):
                continue
            out.append(
                {
                    "kind": "did_protocol",
                    "severity": "error" if s.status == "fail" else "warning",
                    "test": s.key,
                    "value": None,
                    "threshold": None,
                    "message": s.finding,
                    "recovery_hint": s.action,
                    "alternatives": [],
                }
            )
        return out

    def next_steps(self, print_result: bool = False) -> List[Dict[str, str]]:
        """What the protocol still owes: pending steps and open actions."""
        out: List[Dict[str, str]] = []
        for s in self._steps:
            if s.status == "pending":
                out.append(
                    {
                        "action": f"wf.run({s.key!r})",
                        "reason": s.question,
                        "priority": "high",
                        "category": "protocol",
                    }
                )
                break
            if s.action and s.status in ("caution", "fail"):
                out.append(
                    {
                        "action": s.action,
                        "reason": s.finding,
                        "priority": "high" if s.status == "fail" else "medium",
                        "category": s.key,
                    }
                )
        if print_result:
            for item in out:
                print(f"-> {item['action']}\n   {item['reason']}")
        return out

    # ------------------------------------------------------------------ #
    # Rendering (implemented in _workflow_render)
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """The protocol with each step's status, finding and action."""
        return _render.summary(self)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self, *, detail: str = "standard") -> Dict[str, Any]:
        """JSON-safe dictionary of the workflow.

        ``detail='minimal'`` keeps the verdict, the estimate and each
        step's status; ``'standard'`` adds findings and actions;
        ``'agent'`` adds each step's evidence tables.
        """
        return _render.to_dict(self, detail=detail)

    def to_markdown(self, digits: Optional[int] = None, fmt: Any = None) -> str:
        """The protocol as a GitHub-flavoured Markdown section."""
        return _render.to_markdown(self)

    def to_latex(
        self,
        *,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        digits: Optional[int] = None,
        fmt: Any = None,
    ) -> str:
        """The protocol as a booktabs table (status and finding per step)."""
        return _render.to_latex(self, caption=caption, label=label)

    @checks_export_path("excel")
    def to_excel(self, path: str, **kwargs: Any) -> str:
        """Write every table of the workflow to one workbook, a sheet each."""
        return _render.to_excel(self, path, **kwargs)

    @checks_export_path("word")
    def to_word(
        self,
        filename: str,
        title: Optional[str] = None,
        digits: Optional[int] = None,
        fmt: Any = None,
    ) -> str:
        """Write the protocol table and the reporting checklist to .docx."""
        return _render.to_word(self, filename, title=title)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """The event-study path with pointwise and simultaneous intervals."""
        return _render.plot(self, ax=ax, **kwargs)


@records_call("did_workflow")
def did_workflow(
    data: pd.DataFrame,
    y: str,
    *,
    id: str,
    time: str,
    cohort: Optional[str] = None,
    treat: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    target: Optional[str] = None,
    control_group: Optional[str] = None,
    estimator: Optional[str] = None,
    anticipation: Optional[int] = None,
    event_window: Optional[Tuple[int, int]] = None,
    alpha: float = 0.05,
    through: Optional[str] = "report",
    skip: Optional[Sequence[str]] = None,
    jackknife: Optional[bool] = None,
    estimator_options: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> DiDWorkflow:
    """Run a difference-in-differences study as a nine-step protocol.

    The steps, in order (``sp.did_protocol()`` lists them with the
    functions and references behind each):

    1. **declare** -- the target parameter, the comparison group (which
       *is* the parallel-trends assumption), the window, the weights and
       the clustering level. Choices left to defaults are reported as
       such.
    2. **design** -- absorbing adoption, cohorts and their sizes,
       never-treated and always-treated units, balance, cluster counts.
       A design that cannot identify the declared target stops here.
    3. **twfe** -- the Goodman-Bacon share of the two-way fixed effects
       coefficient that comes from already-treated controls.
    4. **estimate** -- the declared target with a heterogeneity-robust
       estimator.
    5. **event_study** -- the path by time since treatment, with
       pointwise and simultaneous (sup-t) intervals and the number of
       cohorts behind each horizon.
    6. **pretrends** -- the joint test of the pre-treatment coefficients
       and the linear trend that test would miss.
    7. **sensitivity** -- Rambachan-Roth bounds: the breakdown value of
       the average post-treatment effect.
    8. **robustness** -- the other estimators of the family, the other
       comparison group, the estimate without covariates, a
       cluster-jackknife standard error and the functional-form test.
    9. **report** -- the statement a paper owes its reader.

    Every step returns a status (``pass`` / ``caution`` / ``fail`` /
    ``info`` / ``skipped``), a finding and an action. The event study,
    the pre-trend test and the sensitivity analysis are all computed from
    the estimator that produced the headline number.

    Parameters
    ----------
    data : pandas.DataFrame
        Long panel, one row per unit and period.
    y : str
        Outcome column.
    id : str
        Unit identifier.
    time : str
        Period column (numeric or date-typed).
    cohort : str, optional
        Each unit's first treated period; ``0`` or ``NaN`` for units never
        treated. Pass this or ``treat``.
    treat : str, optional
        A 0/1 indicator of being treated in that row. The cohort is taken
        as the first treated period, and a path that switches back off is
        reported by the design step.
    covariates : list of str, optional
        Covariates for conditional parallel trends.
    weights : str, optional
        Unit weights. They change the target, not only the precision.
    cluster : str, optional
        The level treatment is assigned at. Defaults to ``id``, and the
        declare step says so.
    target : {'overall', 'event', 'group', 'calendar'}, optional
        The summary of the group-time effects to report: the average over
        treated unit-periods (default), over event times, over cohorts, or
        over calendar periods. ``'group'`` and ``'calendar'`` need
        ``estimator='cs'``.
    control_group : {'nevertreated', 'notyettreated', 'lastcohort'}, optional
        Comparison group. ``'nevertreated'`` (default) or
        ``'notyettreated'`` with ``estimator='cs'``; ``'nevertreated'`` or
        ``'lastcohort'`` with ``'sa'``; ``'bjs'`` uses every untreated
        observation (``'notyettreated'``).
    estimator : {'cs', 'sa', 'bjs'}, optional
        Callaway-Sant'Anna (default), Sun-Abraham, or Borusyak-Jaravel-
        Spiess imputation. The rest of the family is fitted in the
        robustness step.
    anticipation : int, optional
        Periods before treatment in which units may already respond
        (default 0).
    event_window : (int, int), optional
        First and last event time to report and, for ``target='event'``,
        to average over. Default: every event time the data identify.
    alpha : float, default 0.05
        Significance level.
    through : str or None, default 'report'
        Last step to run. ``None`` returns the plan with every step
        pending; continue with ``wf.advance()`` or ``wf.run(step)``.
    skip : sequence of str, optional
        Steps to leave out, among ``'twfe'``, ``'pretrends'``,
        ``'sensitivity'`` and ``'robustness'``.
    jackknife : bool, optional
        Cluster-jackknife standard error in the robustness step
        (``estimator='cs'``). ``None`` runs it when there are at most 100
        clusters; it refits the estimator once per cluster.
    estimator_options : dict, optional
        Extra keyword arguments for the estimator, e.g.
        ``{'allow_unbalanced_panel': True}`` for ``'cs'``.
    seed : int, default 0
        Seed for the simulated critical values (simultaneous bands,
        honest-DiD) and the functional-form test.

    Returns
    -------
    DiDWorkflow
        ``.steps`` (table), ``.summary()``, ``.step(key)``, ``.verdict``,
        ``.estimate`` / ``.se`` / ``.ci`` / ``.pvalue``, ``.event_study``,
        ``.sensitivity``, ``.robustness``, ``.report()``, ``.checklist``,
        ``.plot()``, ``.to_markdown()`` / ``.to_latex()`` /
        ``.to_excel()`` / ``.to_word()`` / ``.to_dict()``, ``.cite()``.

    Notes
    -----
    The statuses are graded against benchmarks the cited papers state:
    the sensitivity step passes when the sign survives ``Mbar = 1``
    (Rambachan and Roth 2023); the pre-trend step cautions when a linear
    trend the test detects with 80 percent power would account for the
    whole estimate (Roth 2022); the design step flags fewer than ten
    treated or comparison clusters. A status is a reading aid for the
    evidence in ``wf.step(key).evidence``, not a substitute for it.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=0)
    >>> wf = sp.did_workflow(
    ...     df, y="y", id="unit", time="time", cohort="first_treat",
    ...     target="overall", control_group="nevertreated", cluster="unit",
    ...     skip=["robustness"],
    ... )
    >>> wf.steps["key"].tolist()[:3]
    ['declare', 'design', 'twfe']
    >>> wf.step("declare").status
    'pass'
    >>> bool(wf.ci[0] < wf.estimate < wf.ci[1])
    True

    Step by step:

    >>> wf = sp.did_workflow(df, y="y", id="unit", time="time",
    ...                      cohort="first_treat", through=None)
    >>> wf.advance().key
    'declare'
    >>> wf.run("design").pending[0]
    'twfe'

    References
    ----------
    Baker et al. (2026) [@baker2026difference]; Roth et al. (2023)
    [@roth2023whats]; Callaway and Sant'Anna (2021)
    [@callaway2021difference]; Goodman-Bacon (2021)
    [@goodmanbacon2021difference]; Roth (2022) [@roth2022pretest];
    Rambachan and Roth (2023) [@rambachan2023more].
    """
    options, explicit = check_options(
        target=target,
        estimator=estimator,
        control_group=control_group,
        anticipation=anticipation,
        event_window=event_window,
        alpha=alpha,
        through=through,
        skip=skip,
        jackknife=jackknife,
        estimator_options=estimator_options,
        seed=seed,
    )
    inputs = resolve_inputs(
        data,
        y=y,
        id=id,
        time=time,
        cohort=cohort,
        treat=treat,
        covariates=covariates,
        weights=weights,
        cluster=cluster,
    )
    if cluster is not None:
        explicit.add("cluster")
    if weights is not None:
        explicit.add("weights")
    wf = DiDWorkflow(inputs, options, explicit)
    if through is not None:
        wf.run(through)
    return wf
