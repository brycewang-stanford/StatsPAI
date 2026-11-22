"""The DiD protocol as data, and the inputs it runs on.

:func:`statspai.did.workflow.did_workflow` executes the steps listed in
:data:`PROTOCOL`.  The list itself is public through :func:`did_protocol`,
so a reader (or an agent) can see the order of a modern
difference-in-differences study, the question each step answers and the
function that answers it, without holding any data.

The order follows the forward-engineering recipe of Baker et al. (2026,
section 6) and the practitioner checklist of Roth et al. (2023): the
target and the comparison group are stated before a coefficient is read,
the design is audited before it is estimated, and a pre-trend test is
followed by its power and by a sensitivity analysis rather than being
read as a verdict.

References
----------
Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026)
[@baker2026difference]; Roth, Sant'Anna, Bilinski and Poe (2023)
[@roth2023whats].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..exceptions import MethodIncompatibility
from ._validate import normalized_cohort, require_constant_cohort

__all__ = [
    "PROTOCOL",
    "STATUSES",
    "DiDStep",
    "WorkflowInputs",
    "did_protocol",
    "resolve_inputs",
    "worst_status",
]

_FN = "sp.did_workflow"

# --------------------------------------------------------------------- #
# Statuses
# --------------------------------------------------------------------- #
#: ``pass``    the step found nothing that qualifies the result;
#: ``caution`` the result stands but the write-up owes the reader a caveat;
#: ``fail``    the step produced evidence against the design (a ``fail`` in
#:             the design audit also blocks estimation);
#: ``info``    the step reports, it does not grade;
#: ``skipped`` not run, with the reason in ``finding``;
#: ``pending`` not reached yet.
STATUSES: Tuple[str, ...] = ("pass", "info", "caution", "fail", "skipped", "pending")

#: Severity order for the overall verdict. ``skipped`` and ``pending``
#: carry no verdict of their own.
_SEVERITY = {"pass": 0, "info": 0, "skipped": 0, "pending": 0, "caution": 1, "fail": 2}


def worst_status(statuses: List[str]) -> str:
    """``fail`` if any step failed, else ``caution`` if any cautions, else ``pass``."""
    worst = max((_SEVERITY[s] for s in statuses), default=0)
    return {0: "pass", 1: "caution", 2: "fail"}[worst]


# --------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------- #
#: (key, title, question, functions that answer it, paper.bib keys).
PROTOCOL: Tuple[Tuple[str, str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "declare",
        "Declare the target and the assumption",
        "Which effect, for whom, against which comparison group, over "
        "which window, and what is treated as random?",
        ("did_workflow",),
        ("baker2026difference",),
    ),
    (
        "design",
        "Audit the design",
        "Is adoption absorbing, who is never treated, how many cohorts and "
        "clusters are there, and does every cohort have a comparison?",
        ("did_cluster_diagnostics",),
        ("roth2023whats", "ulloaperez2025comparative"),
    ),
    (
        "twfe",
        "Diagnose two-way fixed effects",
        "How much of the TWFE coefficient comes from comparisons that use "
        "already-treated units as controls?",
        ("bacon_decomposition",),
        ("goodmanbacon2021difference", "dechaisemartin2020two"),
    ),
    (
        "estimate",
        "Estimate with a heterogeneity-robust estimator",
        "What is the declared target under the declared comparison group?",
        ("callaway_santanna", "aggte", "sun_abraham", "did_imputation"),
        ("callaway2021difference", "sun2021estimating", "borusyak2024revisiting"),
    ),
    (
        "event_study",
        "Report the event-study path",
        "What does the effect look like by time since treatment, with "
        "simultaneous bands, and which cohorts identify each horizon?",
        ("aggte", "uniform_bands"),
        ("callaway2021difference", "freyaldenhoven2019event"),
    ),
    (
        "pretrends",
        "Test pre-trends and the power of that test",
        "Do the pre-treatment coefficients reject, and how large a trend "
        "could the test have missed?",
        ("pretrends_test", "pretrends_slope_for_power"),
        ("roth2022pretest",),
    ),
    (
        "sensitivity",
        "Bound violations of parallel trends",
        "How large a post-treatment violation, relative to the pre-period, "
        "does the conclusion survive?",
        ("honest_did", "breakdown_m"),
        ("rambachan2023more",),
    ),
    (
        "robustness",
        "Vary the choices that were choices",
        "Does the answer move with the estimator, the comparison group, "
        "the outcome scale or the standard-error construction?",
        ("did_summary", "functional_form_test", "cs_jackknife"),
        ("roth2023when", "baker2026difference"),
    ),
    (
        "report",
        "Write down what was estimated",
        "What must the paper state so that a reader can tell which number " "this is?",
        ("did_workflow",),
        ("baker2026difference", "roth2023whats"),
    ),
)

STEP_KEYS: Tuple[str, ...] = tuple(row[0] for row in PROTOCOL)


def did_protocol() -> pd.DataFrame:
    """The steps of a modern difference-in-differences study, as a table.

    This is the order :func:`did_workflow` runs, returned without touching
    any data: one row per step with the question the step answers, the
    ``sp`` functions that answer it and the ``paper.bib`` keys behind it.

    Returns
    -------
    pandas.DataFrame
        Columns ``step`` (1-9), ``key``, ``title``, ``question``,
        ``functions`` (tuple of ``sp`` names) and ``references`` (tuple of
        bib keys, resolvable with ``sp.bibtex``).

    Examples
    --------
    >>> import statspai as sp
    >>> protocol = sp.did_protocol()
    >>> protocol["key"].tolist()[:4]
    ['declare', 'design', 'twfe', 'estimate']
    >>> len(protocol)
    9

    References
    ----------
    Baker et al. (2026) [@baker2026difference]; Roth et al. (2023)
    [@roth2023whats].
    """
    return pd.DataFrame(
        [
            {
                "step": number,
                "key": key,
                "title": title,
                "question": question,
                "functions": functions,
                "references": refs,
            }
            for number, (key, title, question, functions, refs) in enumerate(
                PROTOCOL, start=1
            )
        ]
    )


# --------------------------------------------------------------------- #
# One executed step
# --------------------------------------------------------------------- #
@dataclass
class DiDStep:
    """One step of the protocol: what was asked, what was found, what next.

    Attributes
    ----------
    number, key, title, question
        The step's place in :func:`did_protocol`.
    status : str
        One of ``pass`` / ``caution`` / ``fail`` / ``info`` / ``skipped``
        / ``pending``.
    finding : str
        The result of the step in a sentence or two, with its numbers.
    action : str
        What to do about it; empty when nothing is owed.
    evidence : dict
        The numbers and tables behind ``finding``.
    objects : dict
        The fitted results the step produced (not exported).
    warnings : list of str
        Warnings the underlying functions raised while the step ran.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=0)
    >>> wf = sp.did_workflow(df, y="y", id="unit", time="time",
    ...                      cohort="first_treat", through="design")
    >>> step = wf.step("design")
    >>> isinstance(step, sp.DiDStep)
    True
    >>> step.number, step.key
    (2, 'design')
    >>> sorted(step.to_dict())[:4]
    ['action', 'evidence', 'finding', 'key']
    """

    number: int
    key: str
    title: str
    question: str
    status: str = "pending"
    finding: str = ""
    action: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    objects: Dict[str, Any] = field(default_factory=dict, repr=False)
    warnings: List[str] = field(default_factory=list)
    references: Tuple[str, ...] = ()

    def summary(self) -> str:
        """The step as indented text."""
        lines = [f"[{self.status.upper():^7}] {self.number}. {self.title}"]
        if self.finding:
            lines.append(f"    {self.finding}")
        if self.action:
            lines.append(f"    -> {self.action}")
        for w in self.warnings:
            lines.append(f"    ! {w}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary of the step (fitted objects left out)."""
        from ..core.results import _to_jsonable

        return {
            "step": self.number,
            "key": self.key,
            "title": self.title,
            "question": self.question,
            "status": self.status,
            "finding": self.finding,
            "action": self.action,
            "evidence": _to_jsonable(self.evidence),
            "warnings": list(self.warnings),
            "references": list(self.references),
        }


def blank_steps() -> List[DiDStep]:
    """The protocol as pending steps."""
    return [
        DiDStep(number=n, key=key, title=title, question=question, references=refs)
        for n, (key, title, question, _fns, refs) in enumerate(PROTOCOL, start=1)
    ]


# --------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------- #
@dataclass
class WorkflowInputs:
    """The panel and the column roles, after the cohort column is settled."""

    data: pd.DataFrame
    y: str
    id: str
    time: str
    cohort: str
    covariates: Optional[List[str]]
    weights: Optional[str]
    cluster: Optional[str]
    cohort_source: str  # "column" | "derived from <treat>"
    n_reversal_units: int = 0
    reversal_examples: List[str] = field(default_factory=list)


def _derived_name(data: pd.DataFrame) -> str:
    name = "_did_workflow_cohort"
    while name in data.columns:
        name += "_"
    return name


def _cohort_from_indicator(
    data: pd.DataFrame, *, id: str, time: str, treat: str
) -> Tuple[pd.Series, int, List[str]]:
    """First treated period per unit from a 0/1 path; count reversals.

    Returns the cohort (NaN for units never treated), the number of units
    whose treatment switches back off, and a few of their ids.
    """
    raw = data[treat]
    values = set(pd.unique(raw.dropna()))
    if not values <= {0, 1, True, False}:
        raise MethodIncompatibility(
            f"{_FN}: treat={treat!r} must be a 0/1 indicator of being "
            f"treated in that row; it takes the values "
            f"{sorted(values, key=str)[:6]}.",
            recovery_hint=(
                "If the column holds each unit's first treated period, pass "
                "it as cohort= instead. For a dose or a multi-valued "
                "treatment use sp.did_multiplegt_dyn or sp.continuous_did."
            ),
            diagnostics={"treat": treat, "n_values": len(values)},
            alternative_functions=["sp.did_multiplegt_dyn", "sp.continuous_did"],
        )
    d = raw.fillna(0).astype(int)
    varies = d.groupby(data[id].to_numpy()).nunique() > 1
    periods = np.sort(pd.unique(data[time].dropna()))
    if not bool(varies.any()):
        # A group dummy: constant within every unit. Timing is only
        # recoverable when there are two periods (the second is "post").
        if len(periods) != 2:
            raise MethodIncompatibility(
                f"{_FN}: treat={treat!r} is constant within every unit (a "
                f"treated-group dummy) and the panel has {len(periods)} "
                "periods, so the treatment date cannot be read from it.",
                recovery_hint=(
                    "Pass cohort=<column with each unit's first treated "
                    "period, 0 or NaN for never treated>, or a 0/1 treat "
                    "column that switches on when treatment starts."
                ),
                diagnostics={"treat": treat, "n_periods": int(len(periods))},
            )
        on = d.groupby(data[id].to_numpy()).transform("max").to_numpy() == 1
        cohort = pd.Series(np.where(on, periods[1], np.nan), index=data.index)
        return cohort, 0, []

    frame = pd.DataFrame(
        {"_i": data[id].to_numpy(), "_t": data[time].to_numpy(), "_d": d.to_numpy()}
    ).sort_values(["_i", "_t"], kind="mergesort")
    treated_rows = frame[frame["_d"] == 1]
    first = treated_rows.groupby("_i")["_t"].min()
    cohort = pd.Series(data[id].map(first).to_numpy(), index=data.index)
    # Absorbing means the path never goes 1 -> 0 inside a unit.
    drop = frame.groupby("_i")["_d"].diff() < 0
    bad = pd.unique(frame.loc[drop, "_i"])
    return cohort, int(len(bad)), [str(u) for u in bad[:5]]


def resolve_inputs(
    data: pd.DataFrame,
    *,
    y: str,
    id: str,
    time: str,
    cohort: Optional[str],
    treat: Optional[str],
    covariates: Optional[List[str]],
    weights: Optional[str],
    cluster: Optional[str],
) -> WorkflowInputs:
    """Validate the columns and settle on one cohort column.

    Treatment timing can be given either as ``cohort`` (each unit's first
    treated period) or as ``treat`` (a 0/1 path). With ``treat`` the cohort
    is derived as the first treated period and stored in a new column of a
    copy; the caller's frame is never modified.
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{_FN}: data must be a DataFrame, got {type(data).__name__}.",
            recovery_hint="Pass a long panel: one row per unit and period.",
        )
    if (cohort is None) == (treat is None):
        raise MethodIncompatibility(
            f"{_FN}: pass exactly one of cohort= (each unit's first treated "
            "period) or treat= (a 0/1 indicator of being treated in that "
            f"row); got cohort={cohort!r}, treat={treat!r}.",
            recovery_hint=(
                "cohort='first_treat' for a first-treatment-period column, "
                "or treat='d' for a 0/1 treatment path."
            ),
            diagnostics={"cohort": cohort, "treat": treat},
        )
    if isinstance(covariates, str):
        covariates = [covariates]
    covariates = list(covariates) if covariates else None
    roles: Dict[str, Any] = {
        "y": y,
        "id": id,
        "time": time,
        "cohort": cohort,
        "treat": treat,
        "weights": weights,
        "cluster": cluster,
    }
    if covariates:
        roles["covariates"] = covariates
    check_columns(data, {k: v for k, v in roles.items() if v is not None}, function=_FN)

    if cohort is not None:
        require_constant_cohort(data, unit=id, cohort=cohort, function=_FN)
        return WorkflowInputs(
            data=data,
            y=y,
            id=id,
            time=time,
            cohort=cohort,
            covariates=covariates,
            weights=weights,
            cluster=cluster,
            cohort_source="column",
        )

    assert treat is not None
    derived, n_bad, examples = _cohort_from_indicator(
        data, id=id, time=time, treat=treat
    )
    name = _derived_name(data)
    frame = data.copy()
    frame[name] = derived
    return WorkflowInputs(
        data=frame,
        y=y,
        id=id,
        time=time,
        cohort=name,
        covariates=covariates,
        weights=weights,
        cluster=cluster,
        cohort_source=f"derived from {treat!r} (first treated period)",
        n_reversal_units=n_bad,
        reversal_examples=examples,
    )


# --------------------------------------------------------------------- #
# Design facts shared by several steps
# --------------------------------------------------------------------- #
def design_facts(inputs: WorkflowInputs, *, anticipation: int) -> Dict[str, Any]:
    """Counts read off the panel: units, periods, cohorts, comparisons.

    Periods are handled by rank (position among the observed periods), so
    date-typed and unevenly spaced time columns give the same counts as
    their integer encoding.
    """
    data = inputs.data
    periods = np.sort(pd.unique(data[inputs.time].dropna()))
    n_periods = int(len(periods))

    per_unit = data.groupby(inputs.id, sort=False)
    rows_per_unit = per_unit.size()
    n_units = int(len(rows_per_unit))
    n_dup = int(data.duplicated([inputs.id, inputs.time]).sum())
    n_incomplete = int((rows_per_unit < n_periods).sum())

    raw = per_unit[inputs.cohort].first()
    if pd.api.types.is_numeric_dtype(raw):
        g = normalized_cohort(raw)
        never = (g == 0).to_numpy()
        g_values = g.to_numpy(dtype=float)
    else:
        never = raw.isna().to_numpy()
        g_values = raw.to_numpy()
    # Rank of the first treated period among observed periods; a cohort
    # dated after the sample ends is never treated inside the sample.
    rank = np.full(n_units, -1, dtype=int)
    treated_idx = np.flatnonzero(~never)
    if len(treated_idx):
        rank[treated_idx] = np.searchsorted(
            periods, g_values[treated_idx].astype(periods.dtype), side="left"
        )
    late = rank >= n_periods
    never = never | late
    rank[never] = -1
    always = (~never) & (rank <= anticipation)
    in_sample = (~never) & (~always)

    cohort_rows = []
    for r in sorted(set(rank[in_sample].tolist())):
        members = in_sample & (rank == r)
        label = periods[r]
        cohort_rows.append(
            {
                "cohort": label.item() if hasattr(label, "item") else label,
                "n_units": int(members.sum()),
                "n_pre_periods": int(r),
                "n_post_periods": int(n_periods - r),
            }
        )
    cohorts = pd.DataFrame(
        cohort_rows, columns=["cohort", "n_units", "n_pre_periods", "n_post_periods"]
    )
    if len(cohorts):
        cohorts["share_of_treated"] = cohorts["n_units"] / cohorts["n_units"].sum()

    # Event time is a difference of period ranks whenever the estimators
    # recode the time column (date-typed columns) or the numeric periods are
    # evenly spaced; with unevenly spaced numbers it is not.
    if pd.api.types.is_numeric_dtype(periods):
        spacing = np.diff(periods.astype("float64"))
        evenly_spaced = bool(len(spacing) == 0 or np.allclose(spacing, spacing[0]))
    else:
        evenly_spaced = True

    return {
        "n_obs": int(len(data)),
        "n_units": n_units,
        "n_periods": n_periods,
        "periods": periods,
        "balanced": n_incomplete == 0 and n_dup == 0,
        "n_incomplete_units": n_incomplete,
        "n_duplicate_rows": n_dup,
        "n_never_treated": int(never.sum() - late.sum()),
        "n_treated_after_sample": int(late.sum()),
        "n_comparison_never": int(never.sum()),
        "n_always_treated": int(always.sum()),
        "n_treated_units": int(in_sample.sum()),
        "n_cohorts": int(len(cohorts)),
        "cohorts": cohorts,
        "cohort_ranks": sorted(set(rank[in_sample].tolist())),
        "evenly_spaced": evenly_spaced,
    }
