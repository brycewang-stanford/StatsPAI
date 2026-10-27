"""Code open-ended text responses into codebook categories with an LLM.

Coding free-text survey answers (reasons, mental models, explanations)
into a fixed set of categories is a *measurement* step.  When an LLM does
the coding, the codes are only as good as their reliability, and that
reliability has to be measured and reported like any human coder's:
repeated independent LLM codings give intra-model agreement, and a
human-coded validation subset gives LLM-human agreement (Krippendorff's
alpha, Cohen's kappa).  Downstream estimates that use LLM codes as a
regressor or treatment should correct for the remaining misclassification
(see :func:`statspai.llm_annotator_correct`).

Design rules this module keeps:

* **No client, no codes.**  There is no heuristic fallback.  Without a
  client the call raises :class:`LLMClientRequired` (a
  :class:`~statspai.exceptions.MethodIncompatibility` and a
  :class:`~statspai.causal_llm.LLMConfigurationError`).
* **Strict validation.**  A reply label that is not in the codebook,
  spelled exactly, is invalid.  Invalid items are re-asked once; if the
  second answer is invalid too they are flagged (code ``None``, status
  ``'invalid'``) and counted.  Nothing is coerced to a nearby label.
* **Provenance.**  The result records the model id, the client, the
  temperature, the seed, SHA-256 hashes of the codebook, the prompt
  template and every prompt sent.

The client protocol is the one used across ``statspai.causal_llm``: an
object with ``chat(role, prompt) -> str`` (as built by
``sp.causal_llm.anthropic_client`` / ``openai_client`` / ``echo_client``),
or any callable ``prompt -> str``.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..causal_llm._resolver import LLMConfigurationError
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["llm_code_responses", "LLMCodingResult", "LLMClientRequired"]


class LLMClientRequired(MethodIncompatibility, LLMConfigurationError):
    """No LLM client was given (or none could be resolved).

    Subclasses both the StatsPAI taxonomy (``MethodIncompatibility``,
    with ``recovery_hint``) and ``sp.causal_llm.LLMConfigurationError``,
    the error the package's client resolver raises.

    Examples
    --------
    >>> import statspai as sp
    >>> try:
    ...     sp.llm_code_responses(["sold after the news"], ["news", "price"])
    ... except sp.LLMClientRequired as e:
    ...     print(type(e).__name__)
    LLMClientRequired
    """


PROMPT_TEMPLATE = """You are coding open-ended survey responses into the categories
of a codebook.
{instructions}
Codebook (use the labels exactly as written, including case):
{codebook}

Rules:
- Assign {cardinality} to each response.
- Use only labels from the codebook, spelled exactly as above.
- Code what the response says; do not guess beyond it.
- Return only JSON, with no other text, in the form
  {{"codes": [{{"id": 1, "labels": ["<label>"]}}, ...]}}
  with exactly one entry for every response id below.
{retry_note}
Responses:
{responses}
"""

_RETRY_NOTE = (
    "- Your previous answer for these responses was not valid. Allowed labels "
    "are exactly: {labels}.\n"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class LLMCodingResult(ResultProtocolMixin):
    """Output of :func:`llm_code_responses`.

    Attributes
    ----------
    codes : pandas.DataFrame
        One row per response, one column per LLM coding pass
        (``coder_1``, ``coder_2``, ...).  Single-label codes are strings,
        multi-label codes are tuples of labels; flagged items are ``None``.
    log : pandas.DataFrame
        One row per (response, coder): ``status`` (``'valid'`` /
        ``'invalid'``), ``attempts``, ``reason`` of the last failure and
        the ``raw`` labels the model returned last, and ``first_reason``
        (why the first answer was rejected, ``''`` if it was accepted).
    n_invalid : int
        Flagged (still invalid after the retry) response-coder pairs.
    n_retried : int
        Response-coder pairs that needed the retry.
    agreement : dict
        Reliability of the codes.  ``'llm_alpha'`` (Krippendorff's alpha
        among the LLM passes, ``n_coders > 1``), ``'human_kappa'`` and
        ``'human_alpha'`` (coder 1 vs the human codes) and
        ``'human_accuracy'``; for multi-label coding the same quantities
        per category under ``'per_category'``.  A quantity that cannot
        be computed holds ``{'error': ...}``.
    provenance : dict
        ``model``, ``client``, ``temperature``, ``seed``,
        ``codebook_sha256``, ``prompt_template_sha256``,
        ``prompts_sha256`` (hash of every prompt sent, in order),
        ``n_calls``, ``statspai_version`` and ``created_utc``.
    codebook : dict
        ``{label: description}`` as used.

    Examples
    --------
    >>> import json, re
    >>> import statspai as sp
    >>> def stub(prompt):
    ...     ids = [int(i) for i in re.findall(r'<response id="(\\d+)">', prompt)]
    ...     return json.dumps({"codes": [{"id": i, "labels": ["risk"]} for i in ids]})
    >>> res = sp.llm_code_responses(["too risky", "fees"], {"risk": "", "cost": ""},
    ...                             client=stub, model="stub-1")
    >>> list(res.codes.coder_1)
    ['risk', 'risk']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    codes: pd.DataFrame
    log: pd.DataFrame
    n_invalid: int
    n_retried: int
    agreement: Dict[str, Any]
    provenance: Dict[str, Any]
    codebook: Dict[str, str]
    multi_label: bool = False
    n_coders: int = 1
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> pd.Series:
        """Codes of the first coding pass."""
        return self.codes["coder_1"]

    def to_frame(self) -> pd.DataFrame:
        return self.codes.copy()

    def to_indicators(self, coder: int = 1) -> pd.DataFrame:
        """0/1 matrix (responses x categories) of one pass; NaN if flagged."""
        col = self.codes[f"coder_{coder}"]
        out = pd.DataFrame(
            index=self.codes.index, columns=list(self.codebook), dtype=float
        )
        for i, v in col.items():
            if v is None:
                continue
            chosen = set(v) if isinstance(v, tuple) else {v}
            out.loc[i] = [1.0 if lab in chosen else 0.0 for lab in self.codebook]
        return out

    def summary(self) -> str:
        p = self.provenance
        lines = [
            "LLM coding of text responses",
            "=" * 60,
            f"  Responses        : {len(self.codes)}",
            f"  Categories       : {len(self.codebook)}"
            f" ({'multi-label' if self.multi_label else 'single-label'})",
            f"  Coding passes    : {self.n_coders}",
            f"  Model            : {p.get('model')}  (client: {p.get('client')})",
            f"  Temperature/seed : {p.get('temperature')} / {p.get('seed')}",
            f"  Retried / flagged: {self.n_retried} / {self.n_invalid}",
            f"  Codebook sha256  : {str(p.get('codebook_sha256'))[:16]}...",
        ]
        ag = self.agreement
        for key in ("llm_alpha", "human_alpha", "human_kappa", "human_accuracy"):
            if key in ag:
                v = ag[key]
                lines.append(
                    f"  {key:<17}: "
                    + (f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
                )
        if not ag:
            lines.append(
                "  Reliability      : not measured "
                "(use n_coders > 1 and/or human_codes)"
            )
        lines.append(
            "  LLM codes are measurements: report their reliability, and correct "
            "downstream estimates for misclassification."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LLMCodingResult(n={len(self.codes)}, n_coders={self.n_coders}, "
            f"n_invalid={self.n_invalid}, model={self.provenance.get('model')!r})"
        )


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _normalise_codebook(codebook: Any) -> Dict[str, str]:
    if isinstance(codebook, pd.DataFrame):
        if "label" not in codebook.columns:
            raise MethodIncompatibility(
                "a DataFrame codebook needs a 'label' column.",
                recovery_hint="Columns 'label' and (optional) 'description'.",
            )
        desc = (
            codebook["description"]
            if "description" in codebook.columns
            else pd.Series([""] * len(codebook))
        )
        items = list(zip(codebook["label"].tolist(), desc.tolist()))
    elif isinstance(codebook, Mapping):
        items = list(codebook.items())
    elif isinstance(codebook, (list, tuple)):
        items = [(lab, "") for lab in codebook]
    else:
        raise MethodIncompatibility(
            "codebook must be a dict {label: description}, a list of labels or "
            "a DataFrame with a 'label' column.",
            recovery_hint="e.g. codebook={'risk': 'mentions risk or losses', ...}",
            diagnostics={"type": type(codebook).__name__},
        )
    out: Dict[str, str] = {}
    for lab, desc in items:
        if not isinstance(lab, str) or not lab.strip() or lab != lab.strip():
            raise MethodIncompatibility(
                f"codebook label {lab!r} must be a non-empty string without "
                "leading or trailing spaces.",
                recovery_hint="Use short, distinct string labels.",
            )
        if lab in out:
            raise MethodIncompatibility(
                f"codebook label {lab!r} appears twice.",
                recovery_hint="Labels must be unique.",
            )
        out[lab] = (
            ""
            if desc is None or (isinstance(desc, float) and np.isnan(desc))
            else str(desc)
        )
    if len(out) < 2:
        raise MethodIncompatibility(
            "the codebook needs at least two categories.",
            recovery_hint="Add a residual category such as 'other'.",
        )
    return out


def _resolve_client(
    client: Any, model: Optional[str], temperature: float
) -> Tuple[Callable[[str], str], Dict[str, Any]]:
    if client is None:
        from ..causal_llm._resolver import list_available_providers

        raise LLMClientRequired(
            "llm_code_responses needs an LLM client; there is no offline "
            "fallback for coding text.",
            recovery_hint=(
                "Pass client=statspai.causal_llm.anthropic_client(model=...) "
                "or openai_client(...), any callable prompt -> str, or "
                "client='auto' to resolve one from ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY."
            ),
            diagnostics={"available_providers": list_available_providers()},
            alternative_functions=[
                "sp.causal_llm.anthropic_client",
                "sp.causal_llm.openai_client",
            ],
        )
    if isinstance(client, str):
        from ..causal_llm._resolver import get_llm_client

        if client not in ("auto", "anthropic", "openai"):
            raise MethodIncompatibility(
                f"client={client!r}: a string client must be 'auto', "
                "'anthropic' or 'openai'.",
                recovery_hint="Or pass a client object / callable.",
            )
        try:
            client = get_llm_client(
                provider=None if client == "auto" else client,
                model=model,
                allow_interactive=False,
                temperature=temperature,
            )
        except LLMConfigurationError as exc:
            raise LLMClientRequired(
                f"could not resolve an LLM client: {exc}",
                recovery_hint=(
                    "Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass a client object."
                ),
            ) from exc
    info: Dict[str, Any] = {}
    cmodel = getattr(client, "model", None)
    if model is not None and cmodel is not None and str(cmodel) != str(model):
        raise MethodIncompatibility(
            f"model={model!r} but the client is configured for {cmodel!r}.",
            recovery_hint="Drop model= or build the client with that model.",
        )
    info["model"] = str(cmodel) if cmodel is not None else model
    ctemp = getattr(client, "temperature", None)
    if ctemp is not None and float(ctemp) != float(temperature):
        raise MethodIncompatibility(
            f"temperature={temperature!r} but the client is configured with {ctemp!r}.",
            recovery_hint=(
                "Build the client with the temperature you want, or match it here."
            ),
        )
    info["temperature_enforced"] = ctemp is not None
    info["client"] = str(getattr(client, "name", None) or type(client).__name__)
    if hasattr(client, "chat") and callable(client.chat):
        return (lambda prompt: client.chat("coder", prompt)), info
    if callable(client):
        return client, info
    raise MethodIncompatibility(
        f"client of type {type(client).__name__} has no chat(role, prompt) and is "
        "not callable.",
        recovery_hint="Pass a statspai.causal_llm client or a function prompt -> str.",
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_reply(text: Any) -> Optional[List[Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except ValueError:
        m = _JSON_RE.search(s)
        if m is None:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
    if isinstance(obj, Mapping):
        obj = obj.get("codes")
    return obj if isinstance(obj, list) else None


def _render(text: Any) -> str:
    s = "" if text is None else str(text)
    # a response cannot open or close a response tag of its own
    return s.replace("</response", "</ response").replace("<response", "< response")


def _build_prompt(
    items: List[Tuple[int, str]],
    codebook: Dict[str, str],
    multi_label: bool,
    instructions: str,
    retry: bool,
) -> str:
    cb = "\n".join(
        f'- "{lab}"' + (f": {desc}" if desc else "") for lab, desc in codebook.items()
    )
    resp = "\n".join(f'<response id="{i}">{_render(t)}</response>' for i, t in items)
    return PROMPT_TEMPLATE.format(
        instructions=(f"Context: {instructions}\n" if instructions else ""),
        codebook=cb,
        cardinality=(
            "one or more labels (every label that applies)"
            if multi_label
            else "exactly one label"
        ),
        retry_note=(
            _RETRY_NOTE.format(labels=", ".join(json.dumps(k) for k in codebook))
            if retry
            else ""
        ),
        responses=resp,
    )


def _validate(
    reply: Optional[List[Any]],
    ids: List[int],
    codebook: Dict[str, str],
    multi_label: bool,
) -> Dict[int, Tuple[Optional[Any], str, Any]]:
    """Return {local_id: (code or None, reason, raw)}; reason '' when valid."""
    out: Dict[int, Tuple[Optional[Any], str, Any]] = {}
    if reply is None:
        return {i: (None, "unparseable_reply", None) for i in ids}
    seen: Dict[int, List[Any]] = {}
    for entry in reply:
        if not isinstance(entry, Mapping) or "id" not in entry:
            continue
        try:
            lid = int(entry["id"])
        except (TypeError, ValueError):
            continue
        seen.setdefault(lid, []).append(entry.get("labels", entry.get("label")))
    for i in ids:
        raws = seen.get(i)
        if not raws:
            out[i] = (None, "missing_id", None)
            continue
        if len(raws) > 1:
            out[i] = (None, "duplicate_id", raws)
            continue
        raw = raws[0]
        labs = [raw] if isinstance(raw, str) else raw
        if not isinstance(labs, list) or not all(isinstance(v, str) for v in labs):
            out[i] = (None, "malformed_labels", raw)
            continue
        unknown = [v for v in labs if v not in codebook]
        if unknown:
            out[i] = (None, "unknown_label", raw)
            continue
        if len(set(labs)) != len(labs):
            out[i] = (None, "repeated_label", raw)
            continue
        if not labs:
            out[i] = (None, "no_label", raw)
            continue
        if not multi_label and len(labs) != 1:
            out[i] = (None, "not_exactly_one_label", raw)
            continue
        code: Any = (
            tuple(lab for lab in codebook if lab in labs) if multi_label else labs[0]
        )
        out[i] = (code, "", raw)
    return out


def _safe(fn: Callable[[], Any], what: str, sink: Dict[str, Any], key: str) -> None:
    try:
        sink[key] = fn()
    except (DataInsufficient, MethodIncompatibility) as exc:
        sink[key] = {"error": f"{type(exc).__name__}: {exc}"}
        warnings.warn(
            f"llm_code_responses: {what} not computed ({exc}).",
            UserWarning,
            stacklevel=3,
        )


def _agreement(
    codes: pd.DataFrame,
    human: Optional[List[Any]],
    codebook: Dict[str, str],
    multi_label: bool,
) -> Dict[str, Any]:
    from ..epi.agreement import krippendorff_alpha
    from ..epi.diagnostic import cohen_kappa

    n_coders = codes.shape[1]
    out: Dict[str, Any] = {}
    if not multi_label:
        llm = codes.to_numpy(dtype=object)
        if n_coders > 1:
            _safe(
                lambda: krippendorff_alpha(llm, level="nominal").estimate,
                "Krippendorff's alpha among LLM passes",
                out,
                "llm_alpha",
            )
        if human is not None:
            h = np.array(human, dtype=object)
            mat = np.column_stack([h, llm[:, 0]])
            both = np.array([a is not None and b is not None for a, b in mat])
            out["n_human_compared"] = int(both.sum())
            _safe(
                lambda: krippendorff_alpha(mat, level="nominal").estimate,
                "Krippendorff's alpha vs human codes",
                out,
                "human_alpha",
            )
            if both.sum() >= 2:
                out["human_accuracy"] = float(np.mean(mat[both, 0] == mat[both, 1]))

                def _k() -> float:
                    k = cohen_kappa(list(mat[both, 0]), list(mat[both, 1]))
                    if not np.isfinite(k.kappa):
                        raise DataInsufficient(
                            "kappa undefined (expected agreement is 1).",
                            recovery_hint=(
                                "Validate on a sample with more than one category."
                            ),
                        )
                    return float(k.kappa)

                _safe(_k, "Cohen's kappa vs human codes", out, "human_kappa")
        return out
    # multi-label: per-category binary agreement
    per: Dict[str, Dict[str, Any]] = {}
    for lab in codebook:
        ind = np.array(
            [
                [None if v is None else int(lab in v) for v in codes[c]]
                for c in codes.columns
            ],
            dtype=object,
        ).T
        cat: Dict[str, Any] = {}
        if n_coders > 1:
            _safe(
                lambda: krippendorff_alpha(ind, level="nominal").estimate,
                f"alpha among LLM passes for {lab!r}",
                cat,
                "llm_alpha",
            )
        if human is not None:
            hv = np.array(
                [None if v is None else int(lab in v) for v in human], dtype=object
            )
            mat = np.column_stack([hv, ind[:, 0]])
            _safe(
                lambda: krippendorff_alpha(mat, level="nominal").estimate,
                f"alpha vs human codes for {lab!r}",
                cat,
                "human_alpha",
            )
            both = np.array([a is not None and b is not None for a, b in mat])
            if both.sum() >= 2:
                cat["human_accuracy"] = float(np.mean(mat[both, 0] == mat[both, 1]))
        per[lab] = cat
    out["per_category"] = per
    return out


def _normalise_human(
    human_codes: Any, n: int, codebook: Dict[str, str], multi_label: bool
) -> Optional[List[Any]]:
    if human_codes is None:
        return None
    vals = list(human_codes)
    if len(vals) != n:
        raise MethodIncompatibility(
            f"human_codes has {len(vals)} entries for {n} texts.",
            recovery_hint=(
                "Align human_codes with texts; use None where not human-coded."
            ),
        )
    out: List[Any] = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
            continue
        labs = [v] if isinstance(v, str) else list(v)
        bad = [x for x in labs if x not in codebook]
        if bad:
            raise MethodIncompatibility(
                f"human code(s) {bad} are not codebook labels.",
                recovery_hint="Human codes must use the codebook's labels exactly.",
            )
        if not multi_label:
            if len(labs) != 1:
                raise MethodIncompatibility(
                    "single-label coding needs one human label per response.",
                    recovery_hint="Pass multi_label=True for multi-label codes.",
                )
            out.append(labs[0])
        else:
            out.append(tuple(lab for lab in codebook if lab in labs))
    if all(v is None for v in out):
        raise MethodIncompatibility(
            "human_codes has no coded entry.",
            recovery_hint="Pass human codes for a validation subset, None elsewhere.",
        )
    return out


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def llm_code_responses(
    texts: Sequence[str],
    codebook: Union[Mapping[str, str], Sequence[str], pd.DataFrame],
    *,
    client: Any = None,
    model: Optional[str] = None,
    n_coders: int = 1,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    human_codes: Optional[Sequence[Any]] = None,
    batch_size: int = 20,
    multi_label: bool = False,
    instructions: str = "",
) -> LLMCodingResult:
    """Code free-text responses into codebook categories with an LLM.

    Each response is coded into the codebook's categories by an LLM, in
    batches, with every reply checked strictly against the codebook.
    LLM coding is a measurement step: its reliability must be measured
    and reported, and downstream estimates should allow for its error.
    Set ``n_coders > 1`` for repeated independent codings (intra-model
    agreement) and give ``human_codes`` on a validation subset for
    LLM-human agreement; the result then carries Krippendorff's alpha
    [@hayes2007answering] and Cohen's kappa [@cohen1960coefficient].

    Parameters
    ----------
    texts : sequence of str
        The responses to code.
    codebook : dict, list or DataFrame
        ``{label: description}``, a list of labels, or a DataFrame with
        columns ``label`` (and optionally ``description``).  At least two
        unique labels.  Include a residual category (e.g. ``'other'``)
        if a response may fit none.
    client : object, callable or str
        An ``sp.causal_llm`` client (``anthropic_client``,
        ``openai_client``, ``echo_client``; anything with
        ``chat(role, prompt) -> str``), any callable ``prompt -> str``, or
        ``'auto'`` / ``'anthropic'`` / ``'openai'`` to build one through
        ``sp.causal_llm.get_llm_client`` (non-interactive).  Required:
        without a client the call raises :class:`LLMClientRequired`.
    model : str, optional
        Model id.  Recorded in the provenance; used to build the client
        when ``client`` is a string; must equal the client's own
        ``model`` when it has one.
    n_coders : int, default 1
        Independent coding passes over all responses.
    temperature : float, default 0
        Sampling temperature.  Used to build a client from a string; an
        explicit client must be configured with the same value when it
        exposes one (a plain callable cannot be checked, which the
        provenance records).
    seed : int, optional
        Seeds the order in which responses are batched in each pass
        (``numpy.random.default_rng``), so repeated passes see different
        batch compositions.  Without a seed every pass sends identical
        prompts.  The client protocol has no sampling seed, so the seed
        does not make provider sampling reproducible.
    human_codes : sequence, optional
        Human codes aligned with ``texts`` (``None`` where a response was
        not human-coded); labels must be codebook labels.
    batch_size : int, default 20
        Responses per prompt.
    multi_label : bool, default False
        Allow one or more labels per response (codes become tuples, and
        agreement is reported per category).
    instructions : str, default ''
        Context added to the prompt (e.g. the survey question).  Part of
        the hashed prompt.

    Returns
    -------
    LLMCodingResult
        ``codes`` (one column per pass), ``log``, ``n_invalid``,
        ``n_retried``, ``agreement`` and ``provenance``.

    Raises
    ------
    LLMClientRequired
        No client given, or ``client='auto'`` found no provider.
    MethodIncompatibility
        Bad codebook, model / temperature conflict with the client,
        human codes outside the codebook.

    Notes
    -----
    A reply label that is not a codebook label spelled exactly is
    invalid; so is a missing or duplicated response id, an unparseable
    reply, and (single-label) anything but one label.  Invalid items are
    re-asked once in a new prompt that lists the allowed labels; if
    still invalid they are flagged (``None``) and counted in
    ``n_invalid``, with a warning.  Exceptions raised by the client
    propagate unchanged.

    Repeated passes with identical prompts at temperature 0 mostly
    measure the provider's nondeterminism; a warning says so.  Their
    agreement is an upper bound on reliability, not a substitute for a
    human validation subset.

    Examples
    --------
    A deterministic stub client (any callable ``prompt -> str``):

    >>> import json, re
    >>> import statspai as sp
    >>> def stub(prompt):
    ...     out = []
    ...     for i, t in re.findall(r'<response id="(\\d+)">(.*?)</response>', prompt):
    ...         lab = "risk" if "risk" in t else "cost"
    ...         out.append({"id": int(i), "labels": [lab]})
    ...     return json.dumps({"codes": out})
    >>> res = sp.llm_code_responses(
    ...     ["stocks are too risky", "the fees are high", "risk of losing it all"],
    ...     {"risk": "mentions risk or losses", "cost": "mentions fees or costs"},
    ...     client=stub, model="stub-v1", human_codes=["risk", "cost", None])
    >>> list(res.codes.coder_1)
    ['risk', 'cost', 'risk']
    >>> res.n_invalid, res.agreement["human_accuracy"]
    (0, 1.0)
    >>> sp.llm_code_responses(["x"], ["a", "b"])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    LLMClientRequired: llm_code_responses needs an LLM client

    References
    ----------
    [@hayes2007answering; @cohen1960coefficient]
    """
    cb = _normalise_codebook(codebook)
    if isinstance(texts, str):
        raise MethodIncompatibility(
            "texts must be a sequence of strings, not one string.",
            recovery_hint="Wrap a single response in a list.",
        )
    texts = list(texts)
    if not texts:
        raise DataInsufficient(
            "texts is empty.", recovery_hint="Pass at least one response."
        )
    bad = [i for i, t in enumerate(texts) if not isinstance(t, str)]
    if bad:
        raise MethodIncompatibility(
            f"texts entries {bad[:5]} are not strings.",
            recovery_hint="Drop or convert missing responses before coding.",
        )
    if int(n_coders) < 1 or int(batch_size) < 1:
        raise MethodIncompatibility(
            "n_coders and batch_size must be positive integers.",
            recovery_hint="e.g. n_coders=2, batch_size=20.",
        )
    n_coders, batch_size = int(n_coders), int(batch_size)
    human = _normalise_human(human_codes, len(texts), cb, multi_label)
    call, info = _resolve_client(client, model, temperature)
    if info["model"] is None:
        warnings.warn(
            "llm_code_responses: model id unknown (pass model=); the provenance "
            "will not identify the model.",
            UserWarning,
            stacklevel=2,
        )
    if n_coders > 1 and seed is None and float(temperature) == 0.0:
        warnings.warn(
            "n_coders > 1 with identical prompts at temperature 0: agreement among "
            "the passes measures provider nondeterminism, not coding reliability. "
            "Pass seed= to vary batch composition, and validate against human codes.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed) if seed is not None else None
    n = len(texts)
    codes = np.empty((n, n_coders), dtype=object)
    codes[:] = None
    log_rows: List[Dict[str, Any]] = []
    prompt_hashes: List[str] = []
    n_retried = 0

    def _ask(
        items: List[Tuple[int, str]], retry: bool
    ) -> Dict[int, Tuple[Any, str, Any]]:
        local = [(j + 1, t) for j, (_, t) in enumerate(items)]
        prompt = _build_prompt(local, cb, multi_label, instructions, retry)
        prompt_hashes.append(_sha256(prompt))
        reply = call(prompt)
        res = _validate(_parse_reply(reply), [j for j, _ in local], cb, multi_label)
        return {items[j - 1][0]: v for j, v in res.items()}

    for c in range(n_coders):
        order = rng.permutation(n) if rng is not None else np.arange(n)
        results: Dict[int, Tuple[Any, str, Any]] = {}
        attempts: Dict[int, int] = {}
        for start in range(0, n, batch_size):
            batch = [(int(i), texts[int(i)]) for i in order[start : start + batch_size]]
            results.update(_ask(batch, retry=False))
            for i, _ in batch:
                attempts[i] = 1
        failed = [i for i in order.tolist() if results[i][1]]
        first_reason = {i: results[i][1] for i in failed}
        if failed:
            n_retried += len(failed)
            for start in range(0, len(failed), batch_size):
                batch = [(i, texts[i]) for i in failed[start : start + batch_size]]
                results.update(_ask(batch, retry=True))
                for i, _ in batch:
                    attempts[i] = 2
        for i in range(n):
            code, reason, raw = results[i]
            codes[i, c] = code
            log_rows.append(
                {
                    "response": i,
                    "coder": c + 1,
                    "status": "valid" if not reason else "invalid",
                    "attempts": attempts[i],
                    "reason": reason,
                    "first_reason": first_reason.get(i, ""),
                    "raw": raw,
                }
            )

    log = pd.DataFrame(log_rows)
    n_invalid = int((log["status"] == "invalid").sum())
    if n_invalid:
        reasons = log.loc[log.status == "invalid", "reason"].value_counts().to_dict()
        warnings.warn(
            f"llm_code_responses: {n_invalid} response-coder pair(s) still invalid "
            f"after one retry and flagged as None ({reasons}).",
            UserWarning,
            stacklevel=2,
        )
    codes_df = pd.DataFrame(codes, columns=[f"coder_{c + 1}" for c in range(n_coders)])
    agreement = (
        _agreement(codes_df, human, cb, multi_label) if (n_coders > 1 or human) else {}
    )

    from .. import __version__

    cb_canon = json.dumps([[k, v] for k, v in cb.items()], ensure_ascii=True)
    provenance = {
        "model": info["model"],
        "client": info["client"],
        "temperature": float(temperature),
        "temperature_enforced": info["temperature_enforced"],
        "seed": seed,
        "batch_size": batch_size,
        "multi_label": bool(multi_label),
        "codebook_sha256": _sha256(cb_canon),
        "prompt_template_sha256": _sha256(PROMPT_TEMPLATE + _RETRY_NOTE),
        "instructions_sha256": _sha256(instructions),
        "prompts_sha256": _sha256("\n".join(prompt_hashes)),
        "n_calls": len(prompt_hashes),
        "statspai_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return LLMCodingResult(
        codes=codes_df,
        log=log,
        n_invalid=n_invalid,
        n_retried=n_retried,
        agreement=agreement,
        provenance=provenance,
        codebook=cb,
        multi_label=bool(multi_label),
        n_coders=n_coders,
        diagnostics={"prompt_hashes": prompt_hashes},
    )
