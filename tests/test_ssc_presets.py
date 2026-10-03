"""``sp.ssc``: Stata / fixest small-sample presets for ``sp.feols``."""

import sys
from unittest import mock

import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, MissingDependencyError


def test_presets_resolve_to_pyfixest_switches():
    pytest.importorskip("pyfixest")
    assert sp.ssc("areg")["k_fixef"] == "full"
    assert sp.ssc("ivregress")["k_adj"] is False
    # the stata_ prefix and an override are both accepted
    assert sp.ssc("stata_areg", k_adj=False)["k_adj"] is False


def test_unknown_preset_names_the_valid_ones():
    with pytest.raises(MethodIncompatibility, match="Unknown ssc preset"):
        sp.ssc("no_such_command")


def test_missing_pyfixest_says_which_extra_to_install():
    # pyfixest is an optional extra; its absence used to surface as a bare
    # ModuleNotFoundError from inside the function.
    with mock.patch.dict(sys.modules, {"pyfixest": None}):
        with pytest.raises(MissingDependencyError) as err:
            sp.ssc("areg")
    assert isinstance(err.value, ImportError)
    assert "fixest" in str(err.value)
