"""Option scan round 6: ``dag/``, ``causal_discovery/`` and ``causal_llm/``.

Every test states a true property of the correct behaviour: a hand-checked
d-separation, an invariance (adding a constant to a column, adding pure
noise predictors), the definition of a procedure, or the documented
convention of the reference implementation (R ``dagitty``, R
``InvariantCausalPrediction``), which were run as black boxes only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# --------------------------------------------------------------------- #
#  dag/
# --------------------------------------------------------------------- #


def test_bidirected_latents_do_not_collide_on_underscored_names():
    # A_B <-> C and A <-> B_C share no node: A_B and A are d-separated.
    # Both latents used to be named _L_A_B_C, so one latent fed all four.
    g = sp.dag("A_B <-> C; A <-> B_C")
    assert len(g.latent_nodes) == 2
    assert g.d_separated("A_B", "A")
    assert g.d_separated("C", "B_C")
    assert not g.d_separated("A_B", "C")
    assert not g.d_separated("A", "B_C")
    # The same pair written twice is still one latent.
    assert len(sp.dag("X <-> Y; X <-> Y; X -> Y").latent_nodes) == 1


def test_add_edge_rejects_a_cycle():
    g = sp.dag("X -> M -> Y")
    with pytest.raises(MethodIncompatibility, match="cycle"):
        g.add_edge("Y", "X")
    with pytest.raises(MethodIncompatibility, match="cycle"):
        g.add_edge("Y", "Y")
    # The graph is unchanged after the refused edge.
    assert sorted(g.edges) == [("M", "Y"), ("X", "M")]
    g.add_edge("X", "Y")
    assert ("X", "Y") in g.edges


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.adjustment_sets("Q", "Y"),
        lambda g: g.adjustment_sets("X", "Q"),
        lambda g: g.frontdoor_sets("Q", "Y"),
        lambda g: g.d_separated("Q", "Y"),
        lambda g: g.d_separated("X", "Y", {"Q"}),
        lambda g: g.all_paths("Q", "Y"),
        lambda g: g.backdoor_paths("X", "Q"),
        lambda g: g.path_status("Q", "Y"),
        lambda g: g.classify_variable("Z", "Q", "Y"),
        lambda g: g.bad_controls("Q", "Y"),
        lambda g: g.do("Q"),
        lambda g: sp.swig(g, {"Q": "q"}),
    ],
)
def test_queries_reject_nodes_not_in_the_graph(call):
    # R dagitty::adjustmentSets(g, "Q", "Y") stops with "Q is not a
    # variable". Before, adjustment_sets returned [set()] (adjust for
    # nothing), frontdoor_sets [{'X'}] and d_separated True.
    g = sp.dag("Z -> X -> Y; Z -> Y")
    with pytest.raises(MethodIncompatibility, match="Q"):
        call(g)


def test_adjustment_sets_all_is_the_backdoor_criterion():
    # W is a child of X off the causal path. The backdoor criterion
    # excludes descendants of X; dagitty's type = "all" uses the
    # adjustment criterion and also lists {W, Z}. The docstring says which.
    g = sp.dag("X -> W; X -> Y; Z -> X; Z -> Y")
    assert g.adjustment_sets("X", "Y", minimal=False) == [{"Z"}]
    doc = sp.DAG.adjustment_sets.__doc__
    assert "adjustment criterion" in doc


# --------------------------------------------------------------------- #
#  causal_discovery/: data handling
# --------------------------------------------------------------------- #


def _chain(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = a + b + rng.normal(size=n)
    d = c + rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


@pytest.mark.parametrize("fn", [sp.pc_algorithm, sp.fci])
def test_constant_column_is_refused(fn):
    # A constant has no partial correlation. PC used to get a NaN
    # p-value, keep every edge to it and orient a -> k, b -> k ...
    df = _chain()
    df["k"] = 1.0
    with pytest.raises(DataInsufficient, match="k"):
        fn(df)


@pytest.mark.parametrize("fn", [sp.pc_algorithm, sp.fci])
def test_exactly_collinear_columns_are_refused(fn):
    # a2 = 2a. PC's partial correlation returned 0 ("independent") on the
    # singular matrix, which deleted the true edge a - c.
    df = _chain()
    df["a2"] = 2.0 * df["a"]
    with pytest.raises(DataInsufficient, match="a2|collinear"):
        fn(df)


def test_pc_constant_column_fine_under_discrete_tests():
    # Under a categorical test a constant column is a variable with one
    # level: independent of everything, so it has no edge.
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 500)
    y = (x ^ (rng.random(500) < 0.1)).astype(int)
    df = pd.DataFrame({"x": x, "y": y, "k": 0})
    res = sp.pc_algorithm(df, ci_test="chi-square")
    assert res["skeleton"].loc["k"].sum() == 0
    assert res["skeleton"].loc["x", "y"] == 1


@pytest.mark.parametrize("fn", [sp.pc_algorithm, sp.fci, sp.notears])
def test_boolean_columns_are_used(fn):
    # pandas counts bool as numeric (is_numeric_dtype), and sp.ges /
    # sp.lingam use it. pc_algorithm, fci and notears dropped it.
    df = _chain(n=1500)
    df["t"] = df["d"] > 0
    res = fn(df)
    variables = res["variables"] if isinstance(res, dict) else res.variables
    assert "t" in list(variables)


@pytest.mark.parametrize(
    "kw",
    [{"alpha": 1.5}, {"alpha": 0.0}, {"alpha": -1}, {"max_cond_size": -1}],
)
def test_pc_rejects_out_of_range_options(kw):
    with pytest.raises(MethodIncompatibility):
        sp.pc_algorithm(_chain(n=300), **kw)


# --------------------------------------------------------------------- #
#  LiNGAM
# --------------------------------------------------------------------- #


def _lingam_data(seed=5, n=5000, shift=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    e = rng.uniform(-1, 1, size=(n, 3))
    x0 = e[:, 0]
    x1 = 0.9 * x0 + e[:, 1]
    x2 = -0.7 * x1 + e[:, 2]
    X = np.column_stack([x0, x1, x2]) + np.asarray(shift)
    return pd.DataFrame(X, columns=["x0", "x1", "x2"])


def test_lingam_unstandardised_fit_ignores_column_means():
    # B is a matrix of slopes: adding a constant to a column cannot move
    # it. With standardize=False the regressions ran through the origin
    # on uncentred data; with means (5, -3, 10) B[1, 0] came out wrong by
    # several units.
    base = sp.lingam(_lingam_data(), standardize=False)
    shifted = sp.lingam(_lingam_data(shift=(5.0, -3.0, 10.0)), standardize=False)
    assert shifted.order == base.order
    np.testing.assert_allclose(shifted.adjacency, base.adjacency, atol=1e-9)
    # And it agrees with the standardised path, which rescales the same
    # OLS slopes.
    std = sp.lingam(_lingam_data(shift=(5.0, -3.0, 10.0)), standardize=True)
    assert std.order == shifted.order
    np.testing.assert_allclose(std.adjacency, shifted.adjacency, atol=1e-9)
    assert abs(shifted.adjacency[1, 0] - 0.9) < 0.05


def test_lingam_drops_incomplete_rows_like_the_other_learners():
    df = _lingam_data(n=2000)
    holes = df.copy()
    holes.iloc[[3, 50, 700], [0, 2, 1]] = np.nan
    res = sp.lingam(holes)
    ref = sp.lingam(holes.dropna())
    assert res.order == ref.order
    np.testing.assert_allclose(res.adjacency, ref.adjacency, atol=1e-12)


# --------------------------------------------------------------------- #
#  NOTEARS / DYNOTEARS
# --------------------------------------------------------------------- #


def _sem(seed, k=6, n=500):
    rng = np.random.default_rng(seed)
    mask = rng.random((k, k)) < 0.4
    B = np.triu(mask * rng.uniform(0.5, 2, (k, k)) * rng.choice([-1, 1], (k, k)), 1)
    X = rng.normal(size=(n, k)) @ np.linalg.inv(np.eye(k) - B)
    return pd.DataFrame(X, columns=[f"v{i}" for i in range(k)])


def _acyclic(edges):
    # A directed graph is acyclic iff its adjacency matrix is nilpotent.
    nodes = sorted({v for a, b, _ in edges for v in (a, b)})
    pos = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    for a, b, _ in edges:
        A[pos[a], pos[b]] = 1
    return not np.linalg.matrix_power(A, max(len(nodes), 1)).any()


def test_notears_flags_a_cyclic_result():
    # Stopped after three outer iterations the thresholded graph has a
    # cycle; it used to come back as a "DAG" without a word.
    df = _sem(0)
    with pytest.warns(RuntimeWarning, match="cycle"):
        res = sp.notears(df, max_iter=3)
    assert res["is_dag"] is False
    assert not _acyclic(res["edges"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ok = sp.notears(df)
    assert ok["is_dag"] is True
    assert _acyclic(ok["edges"])


def test_dynotears_flags_a_cyclic_contemporaneous_graph():
    from statspai.causal_discovery._data import is_acyclic_matrix

    rng = np.random.default_rng(6)
    T, d = 400, 5
    B = np.triu((rng.random((d, d)) < 0.5) * rng.uniform(0.4, 1, (d, d)), 1)
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = np.linalg.solve(np.eye(d) - B.T, 0.3 * X[t - 1] + rng.normal(size=d))
    df = pd.DataFrame(X, columns=list("abcde"))
    with pytest.warns(RuntimeWarning, match="cycle"):
        res = sp.dynotears(df, lag=1, max_iter=1)
    assert not is_acyclic_matrix(res.W)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ok = sp.dynotears(df, lag=1)
    assert is_acyclic_matrix(ok.W)


# --------------------------------------------------------------------- #
#  Time-series learners: missing values
# --------------------------------------------------------------------- #


def _var_series(T=300, seed=0):
    rng = np.random.default_rng(seed)
    g = np.zeros(T)
    i_ = np.zeros(T)
    for t in range(1, T):
        g[t] = 0.5 * g[t - 1] + rng.normal()
        i_[t] = 0.4 * g[t - 1] + 0.3 * i_[t - 1] + rng.normal()
    return pd.DataFrame({"gdp": g, "inf": i_})


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.pcmci(d, tau_max=2),
        lambda d: sp.lpcmci(d, tau_max=2),
        lambda d: sp.dynotears(d, lag=1),
    ],
)
def test_time_series_learners_refuse_missing_values(call):
    # One NaN made PCMCI drop two of its three links and DYNOTEARS return
    # an all-zero W; deleting the row would shift every later lag.
    df = _var_series()
    assert len(call(df).__dict__) > 0
    df.iloc[50, 0] = np.nan
    with pytest.raises(DataInsufficient, match="missing"):
        call(df)


# --------------------------------------------------------------------- #
#  ICP
# --------------------------------------------------------------------- #


def _icp_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    env = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    x1 = 0.5 * env + rng.normal(size=n)
    y = x1 + rng.normal(size=n)
    noise = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"N{i}" for i in range(5)])
    return pd.DataFrame({"X1": x1}), y, env, noise


def test_icp_tests_each_set_at_level_alpha():
    # ICP accepts a set when its invariance test is not rejected at
    # alpha; the coverage statement P(S_hat in S*) >= 1 - alpha needs
    # only the test of the true set. R InvariantCausalPrediction::ICP
    # documents alpha the same way ("If no model has a p-value of at
    # least alpha ... modelReject"). sp.icp divided alpha by 2^p - 1.
    from statspai.causal_discovery.icp import _invariance_test

    X, y, env, noise = _icp_data()
    Xn = pd.concat([X, noise[["N0", "N1"]]], axis=1)
    res = sp.icp(Xn, y, env, alpha=0.05)
    from itertools import combinations

    expected = set()
    for k in range(0, 4):
        for S in combinations(Xn.columns, k):
            p, _ = _invariance_test(Xn[list(S)].to_numpy(), y, env, "linear")
            if p >= 0.05:
                expected.add(frozenset(S))
    assert set(res.accepted_subsets) == expected


def test_icp_finds_the_parent_whatever_noise_columns_are_added():
    # Seed 0: the empty set's invariance p-value is 0.034. With the
    # parent alone ICP returns {X1}; adding five pure-noise predictors
    # turned the threshold into 0.05 / 63 and the answer into the empty set.
    X, y, env, noise = _icp_data(seed=0)
    assert sp.icp(X, y, env).parents == {"X1"}
    assert sp.icp(pd.concat([X, noise], axis=1), y, env).parents == {"X1"}


# --------------------------------------------------------------------- #
#  causal_llm/
# --------------------------------------------------------------------- #


def test_causal_mas_keeps_multiword_names_and_drops_unknown_ones():
    client = sp.echo_client(
        lambda system, prompt: (
            "Foo -> Bar\n- years of schooling -> log wage\n1. educ -> wage"
        )
    )
    with pytest.warns(UserWarning, match="Foo"):
        res = sp.causal_mas(
            variables=["educ", "wage", "years of schooling", "log wage"],
            client=client,
            treatment="educ",
            outcome="wage",
        )
    assert ("years of schooling", "log wage") in res.edges
    assert ("educ", "wage") in res.edges
    names = {"educ", "wage", "years of schooling", "log wage"}
    assert all(a in names and b in names for a, b in res.edges)


def _xyz(n=300):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"X": x, "Y": x + rng.normal(size=n), "Z": rng.normal(size=n)})


def test_llm_dag_constrained_warns_when_the_oracle_fails():
    def broken(variables, descriptions):
        raise RuntimeError("API down")

    with pytest.warns(RuntimeWarning, match="API down"):
        res = sp.llm_dag_constrained(_xyz(), oracle=broken)
    assert "API down" in res.provenance["oracle_error"]


def test_llm_dag_constrained_warns_about_unknown_variables():
    with pytest.warns(UserWarning, match="Q"):
        sp.llm_dag_constrained(_xyz(), oracle=lambda v, d: [("X", "Q", 0.9)])


def test_llm_dag_validate_separates_untestable_edges():
    # An edge into a node that is not in the data, or into a declared
    # latent, or from a constant column, has no CI test. It used to be
    # counted "unsupported", the verdict for an edge the data rejects.
    df = _xyz()
    df["K"] = 1.0
    g = sp.dag("U -> X; U -> Y; X -> Y; Q -> Y; K -> Y", latent=["U"])
    with pytest.warns(UserWarning, match="Q"):
        res = sp.llm_dag_validate(g, df)
    ev = res.edge_evidence.set_index("edge")
    assert res.n_supported == 1
    assert res.n_unsupported == 0
    assert res.n_untestable == 2  # Q -> Y and K -> Y; U is latent, skipped
    assert ("U", "X") not in ev.index
    assert not bool(ev.loc[[("Q", "Y")], "testable"].iloc[0])
    assert bool(ev.loc[[("X", "Y")], "testable"].iloc[0])


def test_dag_outputs_do_not_depend_on_the_string_hash_seed():
    # The children of a node are a set, so `edges`, `bad_controls` and
    # everything iterating them came out in an order that changed with
    # PYTHONHASHSEED, i.e. between two runs of the same script.
    import os
    import subprocess
    import sys

    code = (
        "import statspai as sp;"
        "g = sp.dag('Z -> X; Z -> Y; X -> M -> Y; X -> Y; A -> M; A -> Y;"
        " X -> C; Y -> C');"
        "print(g.edges, list(g.bad_controls('X', 'Y')), g.summary('X', 'Y'))"
    )
    outs = set()
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        outs.add(out)
    assert len(outs) == 1
