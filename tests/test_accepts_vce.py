"""``accepts_vce``: Stata's ``vce=`` on estimators with their own SE arguments.

The decorator must (a) translate every accepted spelling into the
estimator's own arguments, (b) leave calls without ``vce=`` untouched, and
(c) refuse any kind the estimator does not implement -- returning some other
standard error instead is the silent failure the grammar exists to remove.
"""

import pytest

from statspai._aliases import accepts_aliases
from statspai.core._vcov_spec import accepts_vce
from statspai.exceptions import MethodIncompatibility


@accepts_vce(default="robust", native={"bootstrap": {"se_method": "bootstrap"}})
def _influence(data=None, cluster=None, se_method="influence"):
    return {"cluster": cluster, "se_method": se_method}


@accepts_vce(default="cluster")
def _panel(data=None, cluster=None):
    return {"cluster": cluster}


@accepts_vce(
    default=("hc1", "robust"),
    cluster="cluster",
    native={"nonrobust": {"robust": False}},
)
def _boolean(data=None, robust=True, cluster=None):
    return {"robust": robust, "cluster": cluster}


@accepts_aliases(id="group")
@accepts_vce(default=None, cluster="clustervars", multiway=True)
def _aliased(data=None, group=None, clustervars=None):
    return {"group": group, "clustervars": clustervars}


def test_no_vce_is_untouched():
    assert _influence() == {"cluster": None, "se_method": "influence"}
    assert _influence(vce=None) == {"cluster": None, "se_method": "influence"}


@pytest.mark.parametrize(
    "spelling",
    ["cluster firm", "cl firm", "vce(cluster firm)", "CLUSTER firm", "cr1 firm"],
)
def test_cluster_spellings_set_the_cluster_argument(spelling):
    assert _influence(vce=spelling)["cluster"] == "firm"


def test_cluster_given_twice_must_agree():
    assert _influence(vce="cluster firm", cluster="firm")["cluster"] == "firm"
    with pytest.raises(MethodIncompatibility):
        _influence(vce="cluster firm", cluster="state")


def test_default_kind_is_a_no_op():
    assert _influence(vce="robust") == {"cluster": None, "se_method": "influence"}


def test_robust_with_cluster_argument_means_cluster():
    # Stata's ``vce(robust) cluster(c)`` idiom.
    assert _influence(vce="robust", cluster="firm")["cluster"] == "firm"


def test_native_kind_writes_its_argument():
    assert _influence(vce="bootstrap")["se_method"] == "bootstrap"
    out = _influence(vce="bootstrap firm")
    assert out == {"cluster": "firm", "se_method": "bootstrap"}


def test_native_kind_conflict_raises():
    with pytest.raises(MethodIncompatibility):
        _influence(vce="bootstrap", se_method="influence")


@pytest.mark.parametrize("spelling", ["hc3", "nonrobust", "hac", "conley"])
def test_unimplemented_kind_raises(spelling):
    with pytest.raises(MethodIncompatibility, match="not available"):
        _influence(vce=spelling)


def test_unknown_spelling_raises():
    with pytest.raises(MethodIncompatibility, match="Unknown"):
        _influence(vce="sandwhich")


def test_bare_cluster_on_a_unit_clustering_estimator_is_its_default():
    assert _panel(vce="cluster") == {"cluster": None}
    assert _panel(vce="cluster state") == {"cluster": "state"}
    with pytest.raises(MethodIncompatibility):
        _panel(vce="robust")


def test_bare_cluster_elsewhere_needs_a_variable():
    with pytest.raises(MethodIncompatibility, match="requires cluster"):
        _influence(vce="cluster")


def test_boolean_robust_switch():
    assert _boolean(vce="robust") == {"robust": True, "cluster": None}
    assert _boolean(vce="hc1") == {"robust": True, "cluster": None}
    assert _boolean(vce="ols") == {"robust": False, "cluster": None}
    assert _boolean(vce="cluster c") == {"robust": True, "cluster": "c"}
    with pytest.raises(MethodIncompatibility):
        _boolean(vce="ols", robust=True)


def test_under_accepts_aliases_and_multiway():
    out = _aliased(id="u", vce="cluster a b")
    assert out == {"group": "u", "clustervars": ["a", "b"]}
    # No default kind: only cluster spellings.
    with pytest.raises(MethodIncompatibility):
        _aliased(vce="robust")


def test_vce_is_advertised_as_an_alias():
    assert _influence.__statspai_aliases__["vce"] == "cluster"
    assert _boolean.__statspai_aliases__["vce"] == "cluster"
    assert _aliased.__statspai_aliases__ == {"vce": "clustervars", "id": "group"}
    assert _influence.__statspai_vce__["supported"] == [
        "bootstrap",
        "cluster",
        "robust",
    ]


def test_decorating_a_function_without_the_named_argument_fails():
    with pytest.raises(ValueError):

        @accepts_vce(default="robust")
        def _no_cluster(data=None):  # pragma: no cover - never called
            return data
