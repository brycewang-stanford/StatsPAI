"""Table functions must accept a list, and must never fail silently.

``sp.etable([m1, m2])`` used to return an empty DataFrame: no exception,
no warning, just a finished-looking table with nothing in it. That is the
cheapest way to hide a mistake, and CLAUDE.md §3.7 forbids it. The list
spelling is also what R's ``modelsummary`` takes and what this package's
own registry example showed, so it is accepted rather than rejected.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.output._format import unwrap_single_sequence

TABLE_FUNCS = ["etable", "modelsummary", "esttab"]


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 0.28 * df["x"] + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.regress("y ~ x", df), sp.regress("y ~ x + z", df)


def _render(func, arg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return str(
            getattr(sp, func)(*arg)
            if isinstance(arg, tuple)
            else getattr(sp, func)(arg)
        )


class TestUnwrapHelper:
    def test_lone_sequence_is_unwrapped(self):
        assert unwrap_single_sequence((["a", "b"],)) == ("a", "b")
        assert unwrap_single_sequence((("a", "b"),)) == ("a", "b")

    def test_varargs_pass_through(self):
        assert unwrap_single_sequence(("a", "b")) == ("a", "b")

    def test_nested_sequences_survive_as_panels(self):
        # regtable reads a list of lists as panels; unwrapping would
        # silently collapse that structure.
        nested = ([["a"], ["b"]],)
        assert unwrap_single_sequence(nested) == nested

    def test_multiple_lists_pass_through(self):
        args = (["a"], ["b"])
        assert unwrap_single_sequence(args) == args


class TestListSpellingAccepted:
    @pytest.mark.parametrize("func", TABLE_FUNCS)
    def test_list_matches_varargs(self, func, models):
        m1, m2 = models
        assert _render(func, [m1, m2]) == _render(func, (m1, m2))

    @pytest.mark.parametrize("func", TABLE_FUNCS)
    def test_list_output_is_not_empty(self, func, models):
        # The precise regression: a non-empty input yielding an empty table.
        out = _render(func, list(models))
        assert "x" in out
        assert out.strip()


class TestOutreg2:
    def test_list_matches_varargs(self, models, tmp_path):
        m1, m2 = models
        varargs = tmp_path / "a.tex"
        as_list = tmp_path / "b.tex"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.outreg2(m1, m2, filename=str(varargs))
            sp.outreg2([m1, m2], filename=str(as_list))
        assert varargs.read_text(encoding="utf-8") == as_list.read_text(
            encoding="utf-8"
        )


class TestUnusableInputIsLoud:
    def test_etable_names_the_offending_argument(self, models):
        # TypeError is etable's released contract (since 1.29.0).
        with pytest.raises(TypeError) as excinfo:
            sp.etable(models[0], "not a model")
        message = str(excinfo.value)
        assert "argument 2" in message
        assert "str" in message
        # An error without a way forward just relocates the problem.
        assert "recovery" in message.lower()

    def test_all_unusable_arguments_raise_rather_than_empty(self):
        with pytest.raises(TypeError):
            sp.etable("a", "b")

    def test_empty_call_still_rejected(self):
        with pytest.raises((ValueError, MethodIncompatibility)):
            sp.modelsummary()
