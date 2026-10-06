"""``jackknife`` and ``bootstrap`` in an ``sp.stata`` session.

Both prefixes, and the ``vce(jackknife)`` / ``vce(bootstrap)`` options that
stand for them, rerun a command on altered samples. Here the command is run
once to find its estimation sample, then once per replication in a session
of its own; what is collected is the coefficient vector or the expression
written before the colon (``jackknife (_b[x] / _b[z]): regress y x z``).

The jackknife has no random part and reproduces Stata's numbers. The
bootstrap draws from numpy's generator: same procedure, other replications.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ._stata_expr import StataExprError
from ._stata_options import _top_level_colon

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["resample_line"]

_PREFIX = re.compile(r"\s*(jackknife|jknife|bootstrap|bs)\b(.*)\Z", re.I | re.S)
_VCE = re.compile(
    r"\bvce\(\s*(jack(?:k(?:n(?:i(?:fe?)?)?)?)?|boot(?:s(?:t(?:r(?:ap?)?)?)?)?)\s*"
    r"(?:,\s*((?:[^()]|\([^()]*\))*))?\)",
    re.I,
)
_CLUSTER = re.compile(
    r"\b(?:cl(?:u(?:s(?:t(?:er?)?)?)?)?\(\s*(\w+)\s*\)" r"|vce\(\s*cl\w*\s+(\w+)\s*\))"
)
_OPTION = re.compile(r"\b(\w+)\s*\(([^()]*)\)")


def _split_options(text: str) -> Tuple[str, str]:
    """``(before, after)`` the first comma outside parentheses."""
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            return text[:i].strip(), text[i + 1 :].strip()
    return text.strip(), ""


def _expression(text: str) -> Optional[str]:
    """The statistic of ``jackknife <text>:``; ``None`` for the coefficients."""
    from ._stata import _top_level_groups

    text = text.strip()
    if not text or text in ("_b", "_b _se"):
        return None
    groups = _top_level_groups(text)
    if len(groups) > 1:
        raise StataExprError(
            "one expression per jackknife / bootstrap call is read; run the "
            "others in calls of their own"
        )
    if len(groups) == 1:
        text = text[1:-1].strip()
    label, colon, body = text.partition(":")
    if colon and re.fullmatch(r"[A-Za-z_]\w*", label.strip()):
        text = body.strip()
    name, eq, body = text.partition("=")
    if eq and re.fullmatch(r"[A-Za-z_]\w*", name.strip()) and not body.startswith("="):
        text = body.strip()  # `ratio = _b[x]/_b[z]`
    return text


def _parse(line: str) -> Optional[Tuple[str, Optional[str], Dict[str, str], str]]:
    """(kind, expression, options, command) of a resampling line."""
    m = _PREFIX.match(line)
    colon = _top_level_colon(line) if m else -1
    if m and colon > 0:
        kind = "jackknife" if m.group(1).lower().startswith("j") else "bootstrap"
        head = line[m.start(2) : colon]
        exp_text, opt_text = _split_options(head)
        options = {k.lower(): v.strip() for k, v in _OPTION.findall(opt_text)}
        for flag in re.sub(r"\w+\s*\([^()]*\)", " ", opt_text).split():
            options[flag.lower()] = ""
        return kind, _expression(exp_text), options, line[colon + 1 :].strip()
    v = _VCE.search(line)
    if v is None or line.find(",") < 0 or v.start() < line.find(","):
        return None
    from ._stata import from_stata

    # a command with a jackknife or bootstrap variance of its own (sdid,
    # didregress ...) keeps it: only a vce() its translation cannot carry
    # is run by resampling the command
    translated = from_stata(line)
    if translated.get("ok") and "vce" not in (
        translated.get("untranslated_options") or []
    ):
        return None
    kind = "jackknife" if v.group(1).lower().startswith("j") else "bootstrap"
    inner = v.group(2) or ""
    options = {k.lower(): val.strip() for k, val in _OPTION.findall(inner)}
    for flag in re.sub(r"\w+\s*\([^()]*\)", " ", inner).split():
        options[flag.lower()] = ""
    command = (line[: v.start()] + line[v.end() :]).strip()
    command = re.sub(r",\s*$", "", command)
    return kind, None, options, command


def resample_line(session: "StataSession", line: str) -> Optional[bool]:
    """Run a ``jackknife`` / ``bootstrap`` line. ``None`` for any other."""
    import statspai as sp

    from ._stata_run import StataSession

    parsed = _parse(line)
    if parsed is None:
        return None
    kind, expr, options, command = parsed
    allowed = {"reps", "cluster", "bca", "nodots", "dots", "mse", "seed", "nodrop"}
    unread = sorted(set(options) - allowed)
    if unread:
        raise StataExprError(f"{kind}: option(s) {unread} are not implemented")
    cluster = options.get("cluster") or None
    if cluster is None:
        hit = _CLUSTER.search(command.partition(",")[2])
        cluster = (hit.group(1) or hit.group(2)) if hit else None

    # once on the data in memory: the estimate and the estimation sample
    session.run(command)
    fit, used = session.last, session.last_data
    if fit is None or used is None:
        raise StataExprError(f"{kind}: {command!r} is not an estimation command")
    index = (getattr(fit, "data_info", None) or {}).get("sample_index")
    sample = used.loc[index] if index is not None else used
    panel, survival = session.panel, session.survival
    scalars = dict(session.stored.get("scalars") or {})

    def statistic(frame: pd.DataFrame) -> Any:
        sub = StataSession(frame)
        sub.panel, sub.survival = panel, survival
        sub.stored["scalars"].update(scalars)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sub.run(command)
        if expr is None:
            return sub.last.params
        return sub.value(expr)

    if kind == "jackknife":
        result = sp.jackknife(sample, statistic, cluster=cluster, mse="mse" in options)
    else:
        try:
            reps = int(options.get("reps") or 50)
        except ValueError:
            raise StataExprError(
                f"bootstrap: reps({options.get('reps')}) is not an integer"
            ) from None
        rng = session.stored.get("rng")
        seed = int(rng.integers(2**31 - 1)) if rng is not None else None
        result = sp.bootstrap(
            sample, statistic, n_boot=reps, cluster=cluster, ci_method="normal",
            seed=seed,
        )  # fmt: skip
        warnings.warn(
            "sp.stata: the bootstrap replications come from numpy's random "
            "numbers, not Stata's; the standard errors agree with Stata's up "
            "to simulation error only.",
            UserWarning,
            stacklevel=4,
        )
    if kind == "bootstrap":
        session.stored["bootstrap"] = {
            "result": result, "statistic": statistic, "sample": sample,
            "cluster": cluster, "name": "_bs_1" if expr is not None else None,
        }  # fmt: skip
    session.output = result
    if expr is None:
        # the command's coefficients with resampling standard errors: what
        # `test` and `lincom` after it should use
        session.last = result
        session._store_estimates(result)
    if kind == "bootstrap":
        session.stored["bootstrap"]["last"] = session.last
    return True


_ESTAT_BOOT = re.compile(r"\s*estat\s+boot(?:strap)?\s*(?:,\s*(.*))?$", re.I)


def estat_bootstrap_line(session: "StataSession", line: str) -> Optional[bool]:
    """``estat bootstrap [, normal percentile bc bca all]`` after a
    bootstrap: bias, standard error and the requested intervals for each
    statistic. ``None`` for any other line.

    ``bc`` (Stata's default) shifts the percentiles by ``z0``, the normal
    quantile of the share of replicates at or below the estimate. ``bca``
    also uses the acceleration ``a`` from the jackknife:
    ``sum(d^3) / (6 sum(d^2)^1.5)`` with ``d`` the deviations of the
    leave-one-out estimates from their mean.
    """
    from scipy import stats

    import statspai as sp

    m = _ESTAT_BOOT.match(line)
    if m is None:
        return None
    kept = session.stored.get("bootstrap")
    if not kept or session.last is not kept.get("last"):
        raise StataExprError("estat bootstrap has to follow a bootstrap")
    wanted = [w for w in re.split(r"[\s,]+", (m.group(1) or "").lower()) if w]
    known = {"normal": "N", "percentile": "P", "bc": "BC", "bca": "BCa"}
    unknown = [w for w in wanted if w not in known and w not in ("all", "noheader")]
    if unknown:
        raise StataExprError(
            f"estat bootstrap: option(s) {unknown} are not implemented"
        )
    kinds = (
        list(known.values())
        if "all" in wanted
        else [known[w] for w in wanted if w in known] or ["BC"]
    )
    result = kept["result"]
    info = getattr(result, "model_info", None) or {}
    if "boot_distribution" in info:
        draws = np.asarray(info["boot_distribution"], dtype=float)
        names = [str(c) for c in info["boot_distribution"].columns]
        observed = np.asarray(result.params, dtype=float)
    else:
        draws = np.asarray(result.boot_distribution, dtype=float).reshape(-1, 1)
        names = [kept["name"] or "_bs_1"]
        observed = np.array([float(result.estimate)])
    draws = draws[np.all(np.isfinite(draws), axis=1)]
    level = float(session.stored.get("level", 95.0)) / 100.0
    lo_p, hi_p = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    z = stats.norm.ppf(hi_p)
    accel = np.zeros(len(names))
    if "BCa" in kinds:
        jack = sp.jackknife(kept["sample"], kept["statistic"], cluster=kept["cluster"])
        held = (getattr(jack, "model_info", None) or {}).get("replicates")
        if held is None:
            held = jack.replicates
        values = np.asarray(held, dtype=float).reshape(-1, len(names))
        dev = values.mean(axis=0) - values
        accel = (dev**3).sum(axis=0) / (6.0 * (dev**2).sum(axis=0) ** 1.5)
    rows = []
    for j, name in enumerate(names):
        column = draws[:, j]
        se = float(column.std(ddof=1))
        z0 = float(stats.norm.ppf(np.mean(column <= observed[j])))
        for kind in kinds:
            if kind == "N":
                lo, hi = observed[j] - z * se, observed[j] + z * se
            elif kind == "P":
                lo, hi = np.quantile(column, [lo_p, hi_p])
            else:
                a = accel[j] if kind == "BCa" else 0.0
                shifted = [
                    stats.norm.cdf(z0 + (z0 + q) / (1.0 - a * (z0 + q)))
                    for q in (-z, z)
                ]
                lo, hi = np.quantile(column, shifted)
            rows.append(
                {
                    "statistic": name, "observed": float(observed[j]),
                    "bias": float(column.mean() - observed[j]), "se": se,
                    "ci_lower": float(lo), "ci_upper": float(hi), "type": kind,
                }
            )  # fmt: skip
    session.output = pd.DataFrame(rows)
    return True
