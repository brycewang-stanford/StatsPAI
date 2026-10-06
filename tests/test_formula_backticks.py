"""Column names that are not identifiers, written the way R writes them.

`` `In-game Purchases` ~ `Side-quest Engagement` `` used to fail in every
formula entry point (the one exception was ``sp.feols``), with messages
such as "Variable(s) not found: ['quest']". Backticks are now read as
``Q("...")`` in one place, ``core.utils.r_formula_idioms``, and the
parsers that read names on their own (instrumental variables, count
models, fixed effects, smooths) accept the quoted form. Found while
replaying Ness, *Causal AI*, whose data have such names throughout.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.utils import backticks_to_q, unquote_name


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    d = 0.6 * z + rng.normal(size=n)
    y = 1 + 2 * x + 1.5 * d + rng.normal(size=n)
    plain = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "d": d,
            "z": z,
            "g": rng.integers(0, 20, n),
            "yb": (y > y.mean()).astype(int),
            "yc": np.abs(y).round().astype(int),
        }
    )
    odd = plain.rename(
        columns={"y": "my y", "x": "x-1", "d": "the d", "z": "z z", "g": "g id"}
    )
    return plain, odd


NAMES = {"y": "my y", "x": "x-1", "d": "the d", "z": "z z", "g": "g id"}
CALLS = {
    "regress": ("{y} ~ {x} + {d}", lambda f, D: sp.regress(f, data=D)),
    "iv": ("{y} ~ {x} + ({d} ~ {z})", lambda f, D: sp.iv(f, data=D)),
    "ivreg": ("{y} ~ {x} + ({d} ~ {z})", lambda f, D: sp.ivreg(f, data=D)),
    "feols": ("{y} ~ {x} + {d} | {g}", lambda f, D: sp.feols(f, data=D)),
    "logit": ("yb ~ {x} + {d}", lambda f, D: sp.logit(f, data=D)),
    "probit": ("yb ~ {x} + {d}", lambda f, D: sp.probit(f, data=D)),
    "glm": ("yb ~ {x} + {d}", lambda f, D: sp.glm(f, data=D, family="binomial")),
    "poisson": ("yc ~ {x} + {d}", lambda f, D: sp.poisson(f, data=D)),
    "nbreg": ("yc ~ {x} + {d}", lambda f, D: sp.nbreg(f, data=D)),
    "ppmlhdfe": ("yc ~ {x} + {d} | {g}", lambda f, D: sp.ppmlhdfe(f, data=D)),
    "qreg": ("{y} ~ {x} + {d}", lambda f, D: sp.qreg(f, data=D)),
    "tobit": ("{y} ~ {x} + {d}", lambda f, D: sp.tobit(f, data=D, ll=-5)),
    "gam": ("{y} ~ s({x}) + {d}", lambda f, D: sp.gam(f, data=D)),
}
STYLES = {
    "backticks": lambda name: f"`{name}`",
    "Q": lambda name: f'Q("{name}")',
}


@pytest.mark.parametrize("style", sorted(STYLES))
@pytest.mark.parametrize("entry", sorted(CALLS))
def test_quoted_names_give_the_same_fit(frames, entry, style):
    plain, odd = frames
    template, fit = CALLS[entry]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = fit(template.format(**{k: k for k in NAMES}), plain)
        quoted = fit(
            template.format(**{k: STYLES[style](v) for k, v in NAMES.items()}), odd
        )
    assert np.allclose(
        np.asarray(quoted.params, dtype=float),
        np.asarray(base.params, dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_terms_are_named_as_q(frames):
    _, odd = frames
    fit = sp.regress("`my y` ~ `x-1` + `the d`", data=odd)
    assert list(fit.params.index) == ["Intercept", 'Q("x-1")', 'Q("the d")']
    iv = sp.iv("`my y` ~ `x-1` + (`the d` ~ `z z`)", data=odd)
    assert 'Q("the d")' in iv.params.index and 'Q("x-1")' in iv.params.index


def test_a_hyphen_in_a_quoted_name_is_not_an_intercept_switch(frames):
    # The IV parser rewrote "- 1" anywhere, so Q("x-1") became Q("x+ -1").
    plain, odd = frames
    quoted = sp.iv('Q("my y") ~ Q("x-1") + (Q("the d") ~ Q("z z"))', data=odd)
    base = sp.iv("y ~ x + (d ~ z)", data=plain)
    assert np.allclose(quoted.params.to_numpy(), base.params.to_numpy(), rtol=1e-10)
    no_const = sp.iv("y ~ x - 1 + (d ~ z)", data=plain)
    assert "Intercept" not in no_const.params.index


def test_translation_and_unquoting():
    assert backticks_to_q("`a b` ~ `c-d` + x") == 'Q("a b") ~ Q("c-d") + x'
    assert backticks_to_q('y ~ Q("a`b") + `c`') == 'y ~ Q("a`b") + Q("c")'
    assert backticks_to_q('`say "hi"` ~ x') == 'Q("say \\"hi\\"") ~ x'
    assert backticks_to_q("y ~ x") == "y ~ x"
    assert unquote_name('Q("g id")') == "g id"
    assert unquote_name("`g id`") == "g id"
    assert unquote_name("C(g)") == "C(g)"
    assert unquote_name('Q("a") + Q("b")') == 'Q("a") + Q("b")'
    with pytest.raises(sp.MethodIncompatibility, match="unmatched backtick"):
        backticks_to_q("`a b ~ x")


def test_a_missing_quoted_column_is_named(frames):
    _, odd = frames
    with pytest.raises(Exception) as err:
        sp.regress("`my y` ~ `no such column`", data=odd)
    assert "no such column" in str(err.value)
