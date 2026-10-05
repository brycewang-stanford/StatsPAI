"""Loops, branches and computed macros; more than one dataset; matrices.

``sp.stata`` runs these the way Stata does: a block is collected up to its
closing brace and its body handed back line by line with the loop macro
set. The multi-dataset commands were run beside Stata 18 on small frames
built for the purpose (``merge`` 1:1, m:1 and 1:m, ``append``, ``reshape``
both ways, ``frlink`` / ``frget``); the expected frames below are Stata's.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.exceptions import MethodIncompatibility

NAN = np.nan


@pytest.fixture()
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = pd.DataFrame({"g": np.repeat(np.arange(12), 30)})
    out["x"] = rng.normal(size=360)
    out["w"] = rng.normal(size=360)
    out["y"] = 1 + 0.3 * out.x - 0.2 * out.w + rng.normal(size=360)
    return out


def run(commands: str, data: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(commands, data=data, **kwargs)


def held(data: pd.DataFrame, commands: str, **files) -> pd.DataFrame:
    session = StataSession(data)
    session.files.update(files)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in commands.split("\n"):
            session.run(line)
    return session.data


# -------------------------------------------------------------------- loops
@pytest.mark.parametrize(
    "script, want",
    [
        ("scalar s = 0\nforvalues i = 1/10 {\n scalar s = s + `i'\n}\ndisplay s", 55),
        (
            "scalar s = 0\nforvalues i = 10(-2)2 {\n scalar s = s + `i'\n}\ndisplay s",
            30,
        ),
        (
            "scalar s = 0\nforvalues i = 1 3 : 9 {\n scalar s = s + `i'\n}\ndisplay s",
            25,
        ),
        (
            "scalar s = 0\nforeach n of numlist 1/3 10 {\n scalar s = s + `n'\n}\ndisplay s",
            16,
        ),
        (
            "scalar k = 0\nforeach v of varlist x-y {\n scalar k = k + 1\n}\ndisplay k",
            3,
        ),
        ("scalar k = 0\nforeach v of varlist g* {\n scalar k = k + 1\n}\ndisplay k", 1),
        (
            "local vs x w\nscalar k = 0\nforeach v of local vs {\n su `v'\n"
            " scalar k = k + r(N)\n}\ndisplay k",
            720,
        ),
        (
            "scalar s = 0\nforvalues i = 1/3 {\n quietly {\n  forvalues j = 1/2 {\n"
            "   scalar s = s + `i' * `j'\n  }\n }\n}\ndisplay s",
            18,
        ),
        (
            "local i = 1\nscalar s = 0\nwhile `i' <= 4 {\n scalar s = s + `i'\n"
            " local ++i\n}\ndisplay s",
            10,
        ),
        (
            "scalar s = 0\nforvalues i = 1/10 {\n if `i' == 3 {\n  continue\n }\n"
            " if `i' > 5 {\n  continue, break\n }\n scalar s = s + `i'\n}\ndisplay s",
            12,
        ),
    ],
)
def test_loops(df, script, want):
    assert run(script, df) == want


def test_loop_over_variables_feeds_estimation(df):
    out = run("foreach v in x w {\n gen double `v'2 = `v'^2\n}\nreg y x2 w2", df)
    direct = sp.regress("y ~ x2 + w2", data=df.assign(x2=df.x**2, w2=df.w**2))
    assert float(out.params["x2"]) == pytest.approx(
        float(direct.params["x2"]), rel=1e-12
    )


@pytest.mark.parametrize(
    "a, want",
    [(3, 1), (1, 5), (-1, 2)],
)
def test_if_else_chain(df, a, want):
    script = (
        f"scalar a = {a}\nif a > 2 {{\n scalar b = 1\n}}\nelse if a > 0 {{\n scalar b = 5\n}}\n"
        "else {\n scalar b = 2\n}\ndisplay b"
    )
    assert run(script, df) == want
    cuddled = f"scalar a = {a}\nif a > 2 {{\n scalar b = 1\n}} else {{\n scalar b = 9\n}}\ndisplay b"
    assert run(cuddled, df) == (1 if a > 2 else 9)


def test_else_needs_its_if(df):
    with pytest.raises(MethodIncompatibility, match="without an `if`"):
        run("scalar a = 1\nelse {\n scalar b = 2\n}", df)


def test_a_loop_that_does_not_end_is_stopped(df, monkeypatch):
    import statspai.agent._translation._stata_flow as flow

    monkeypatch.setattr(flow, "_MAX_PASSES", 50)
    with pytest.raises(MethodIncompatibility, match="did not end"):
        run("while 1 {\n scalar a = 1\n}", df)


# ------------------------------------------------------------------- macros
def test_computed_macros(df):
    assert run("su x\nlocal m = r(mean)\ngen double c = x - `m'\nsu c\n"
               "display abs(r(mean)) < 1e-12", df) == 1  # fmt: skip
    assert run("display `=2+3'", df) == 5
    assert run("count if x > 0\ndisplay `r(N)' + 0", df) == float((df.x > 0).sum())
    assert run("global k = 2 * 3\ndisplay $k", df) == 6
    assert run("local i = 5\nlocal --i\nlocal --i\ndisplay `i'", df) == 3
    assert run("local third = 1/3\ndisplay `third' * 3", df) == pytest.approx(1.0)


def test_tempvar_names_do_not_collide(df):
    out = held(df, "tempvar a b\ngen `a' = 1\ngen `b' = 2")
    fresh = [c for c in out.columns if c.startswith("__var")]
    assert len(fresh) == 2 and out[fresh].sum().tolist() == [360.0, 720.0]


def test_what_only_stata_can_read_is_still_refused(df):
    # an extended macro function the session computes ...
    assert run("local n : word count a b c\ndisplay `n'", df) == 3
    # ... and one that only a running Stata can answer
    with pytest.raises(MethodIncompatibility, match="not implemented"):
        run('local f : dir . files "*.dta"\ndisplay "`f\'"', df)
    with pytest.raises(MethodIncompatibility, match="control flow"):
        run("mata\n x = 1\nend", df)


def test_display_with_text(df):
    assert run('scalar a = 2\ndisplay "the value is: " a', df) == 2
    assert run('scalar a = 2\ndi as text "twice " as result %9.3f a * 2', df) == 4
    assert run('reg y x w\ndisplay "b: " _b["x"]', df) == run(
        "reg y x w\ndisplay _b[x]", df
    )


# ----------------------------------------------------------------- programs
def test_program_with_arguments(df):
    script = (
        "program addone\n args v k\n quietly {\n  su `v'\n }\n"
        " scalar out = r(mean) + `k'\nend\naddone x 1\ndisplay out"
    )
    assert run(script, df) == float(df.x.mean()) + 1


def test_program_locals_do_not_leak(df):
    session = StataSession(df)
    for line in (
        "local keep 7",
        "program p",
        " args v",
        " local inside 1",
        "end",
        "p x",
    ):
        session.run(line)
    assert session._macros.locals == {"keep": "7"}


def test_randomisation_inference_program(df):
    """The shape of the book's program: a permutation loop with a counter."""
    d = df.assign(d=(df.g % 2).astype(float))
    script = (
        "set seed 1\nprogram ri\n args v\n quietly {\n  su `v' if d == 1\n"
        "  scalar t1 = r(mean)\n  su `v' if d == 0\n  scalar obs = t1 - r(mean)\n }\n"
        " scalar hits = 0\n forvalues i = 1/200 {\n  gen perm = runiform() > 0.5\n"
        "  quietly {\n   su `v' if perm\n   scalar a = r(mean)\n   su `v' if !perm\n"
        "   scalar b = r(mean)\n  }\n  if (abs(a - b) >= abs(obs)) {\n"
        "   scalar hits = hits + 1\n  }\n  drop perm\n }\n scalar p = hits / 200\nend\n"
        "ri y\ndisplay p"
    )
    p = run(script, d)
    assert 0 <= p <= 1


# ----------------------------------------------------------- save, use, append
def test_tempfile_save_use_append(df):
    out = held(df, "tempfile t\npreserve\nkeep in 1/10\nsave `t'\nrestore\n"
               "keep in 1/5\nappend using `t'")  # fmt: skip
    assert len(out) == 15
    back = held(df, "tempfile t\nsave `t'\nclear\nuse `t', clear")
    pd.testing.assert_frame_equal(back, df, check_dtype=False)


def test_save_keeps_a_copy_and_says_so(df):
    session = StataSession(df)
    with pytest.warns(UserWarning, match="no file is written"):
        session.run("save panel")
    assert "panel" in session.files
    with pytest.raises(MethodIncompatibility, match="already exists"):
        session.run("save panel")


def test_use_of_an_unknown_file_is_refused(df):
    with pytest.raises(MethodIncompatibility):
        run("use nosuchfile, clear", df)


# -------------------------------------------------------------------- merge
MASTER = pd.DataFrame({"id": [6, 9, 4, 3, 2, 1.0], "x": [60, 50, 40, 30, 20, 10.0]})
USING = pd.DataFrame(
    {"id": [2, 3, 4, 5, 6.0], "z": [200, 300, 400, 500, 600.0], "x": -1.0}
)


def test_merge_one_to_one_is_statas():
    out = held(MASTER, "merge 1:1 id using u", u=USING)
    # Stata: master rows sorted by the key, the using-only row last; a
    # shared variable keeps the master's value, the using-only row its own
    assert out["id"].tolist() == [1, 2, 3, 4, 6, 9, 5]
    assert out["x"].tolist() == [10, 20, 30, 40, 60, 50, -1]
    np.testing.assert_array_equal(out["z"], [NAN, 200, 300, 400, 600, NAN, 500])
    assert out["_merge"].tolist() == [1, 3, 3, 3, 3, 1, 2]


def test_merge_options():
    out = held(
        MASTER,
        "merge 1:1 id using u, keep(match master) nogenerate keepusing(z)",
        u=USING,
    )
    assert list(out.columns) == ["id", "x", "z"] and len(out) == 6
    out = held(MASTER, "merge 1:1 id using u, keep(3) gen(src)", u=USING)
    assert out["src"].eq(3).all() and len(out) == 4
    with pytest.raises(MethodIncompatibility, match="not all observations"):
        held(MASTER, "merge 1:1 id using u, assert(match)", u=USING)


def test_merge_many_to_one_and_back():
    long = pd.DataFrame({"g": [2, 3, 1, 2, 3, 1, 2, 3.0], "t": np.arange(1.0, 9.0)})
    groups = pd.DataFrame({"g": [3, 2, 1.0], "w": [1.5, 1.0, 0.5]})
    out = held(long, "merge m:1 g using gs, gen(mm)", gs=groups)
    assert out["g"].tolist() == sorted(long.g) and out["mm"].eq(3).all()
    assert (out["w"] == out["g"] / 2).all()
    back = held(groups, "merge 1:m g using long", long=long)
    assert len(back) == 8 and set(back.columns) == {"g", "w", "t", "_merge"}


def test_merge_checks_the_keys():
    dup = pd.concat([USING, USING.iloc[:1]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="do not uniquely identify"):
        held(MASTER, "merge 1:1 id using u", u=dup)
    with pytest.raises(MethodIncompatibility, match="m:m is not implemented"):
        held(MASTER, "merge m:m id using u", u=USING)
    with pytest.raises(MethodIncompatibility, match="not in the session"):
        held(MASTER, "merge 1:1 id using nowhere")


def test_files_argument_of_sp_stata():
    out = run(
        "merge 1:1 id using other\ncount if _merge == 3", MASTER, files={"other": USING}
    )
    assert out == 4
    out = run(
        'merge 1:1 id using "data/other.dta"\ncount', MASTER, files={"other": USING}
    )
    assert out == 7


# ------------------------------------------------------------------ reshape
WIDE = pd.DataFrame(
    {
        "id": [4, 3, 2, 1.0],
        "inc2001": [6, 4.5, 3, 1.5],
        "inc2002": [10, 7.5, 5, 2.5],
        "ue2001": [4, 3, 2, 1.0],
        "ue2002": [-4, -3, -2, -1.0],
        "sex": [0, 1, 0, 1.0],
    }
)


def test_reshape_long_then_wide_is_statas():
    # reshape itself is `_stata_reshape.py`; this is the frame Stata 18 gave
    # for these commands in the multi-dataset run
    long = held(WIDE, "reshape long inc ue, i(id) j(year)")
    assert list(long.columns) == ["id", "year", "inc", "ue", "sex"]
    assert long["id"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert long["year"].tolist() == [2001, 2002] * 4
    assert long["inc"].tolist() == [1.5, 2.5, 3, 5, 4.5, 7.5, 6, 10]
    wide = held(long, "reshape wide inc ue, i(id) j(year)")
    assert list(wide.columns) == ["id", "inc2001", "ue2001", "inc2002", "ue2002", "sex"]
    pd.testing.assert_frame_equal(
        wide[sorted(wide.columns)],
        WIDE.sort_values("id").reset_index(drop=True)[sorted(WIDE.columns)],
        check_dtype=False,
    )


def test_reshape_refuses_what_stata_refuses():
    long = held(WIDE, "reshape long inc ue, i(id) j(year)")
    with pytest.raises(MethodIncompatibility, match="not constant within"):
        held(long.assign(v=np.arange(8.0)), "reshape wide inc ue, i(id) j(year)")
    with pytest.raises(MethodIncompatibility, match="do not identify"):
        held(pd.concat([WIDE, WIDE]), "reshape long inc ue, i(id) j(year)")


# ------------------------------------------------------------------- frames
def test_frames_put_change_link_get():
    d = pd.DataFrame({"cl": [1, 2, 0, 1, 2, 0.0], "v": np.arange(1.0, 7.0)})
    out = held(
        d,
        "frame put cl, into(cls)\nframe change cls\nduplicates drop\n"
        "gen ag = cl * 10 + 1\nframe change default\nfrlink m:1 cl, frame(cls)\n"
        "frget ag, from(cls)",
    )
    assert out["ag"].tolist() == [11, 21, 1, 11, 21, 1]  # Stata's
    # an unambiguous abbreviation of the link variable, as Stata allows
    out = held(
        d,
        "frame put cl, into(clusters)\nframe change clusters\nduplicates drop\n"
        "gen ag = cl + 100\ncwf default\nfrlink m:1 cl, frame(clusters)\n"
        "frget ag, from(cluster)",
    )
    assert out["ag"].tolist() == [101, 102, 100, 101, 102, 100]


def test_frame_post_collects_results(df):
    out = held(
        df,
        "frame create res b se\nforvalues k = 1/3 {\n reg y x if g >= `k'\n"
        " frame post res (_b[x]) (_se[x])\n}\nframe change res",
    )
    assert out.shape == (3, 2)
    fit = sp.regress("y ~ x", data=df[df.g >= 2])
    assert out["b"].iloc[1] == pytest.approx(float(fit.params["x"]), rel=1e-12)


def test_frame_prefix_and_housekeeping(df):
    assert (
        run("frame put x, into(other)\nframe other: count if x > 0", df)
        == (df.x > 0).sum()
    )
    session = StataSession(df)
    for line in ("frame copy default twin", "frame change twin", "frame drop default"):
        session.run(line)
    assert session.frame == "twin" and len(session.data) == 360
    session.run("frames reset")
    assert session.frame == "default" and session.data.empty
    with pytest.raises(MethodIncompatibility, match="not found"):
        run("frame change nowhere", df)
    with pytest.raises(MethodIncompatibility, match="block is not implemented"):
        run("frame put x, into(o)\nframe o {\n count\n}", df)


# ----------------------------------------------------------------- matrices
def test_matrix_cells_and_functions(df):
    assert run("matrix A = J(3, 2, .)\nmatrix A[2,1] = 5\n"
               "matrix define A[3,2] = A[2,1] * 2\ndisplay A[3,2]", df) == 10  # fmt: skip
    assert (
        run(
            "matrix A = J(4, 2, 1)\ndisplay rowsof(A) * 10 + colsof(A) + el(A, 1, 1)",
            df,
        )
        == 43
    )
    assert (
        run("matrix A = (1, 2 \\ 3, 4)\nmatrix B = A'\ndisplay B[1,2] + A[1,2]", df)
        == 5
    )
    assert np.isnan(run("matrix A = J(2, 2, 1)\ndisplay el(A, 3, 1)", df))
    with pytest.raises(MethodIncompatibility, match="not implemented"):
        run("matrix A = I(2)\nmatrix C = cholesky(A)", df)
    with pytest.raises(MethodIncompatibility, match="not understood"):
        run("matrix A = I(2)\nmatrix C = A # A", df)
    with pytest.raises(MethodIncompatibility, match="conformability"):
        run("matrix A = J(2,3,1)\nmatrix C = A * A", df)
    with pytest.raises(MethodIncompatibility, match="out of range"):
        run("matrix A = J(2,2,1)\nmatrix A[3,1] = 0", df)


def test_matrix_algebra(df):
    # the variance of a linear combination, as a textbook do-file writes it
    se = run(
        "reg y x w, r\nscalar d1 = 2\nmatrix vb = get(VCE)\nmatrix d = (d1, 1, 0)\n"
        "matrix ve = d*vb*d'\ndisplay sqrt(ve[1,1])",
        df,
    )
    lincom = run("reg y x w, r\nlincom 2*x + w", df)
    assert se == pytest.approx(float(lincom["se"]), rel=1e-12)
    # least squares by hand
    b = run(
        "mkmat x w, matrix(X)\nmkmat y, matrix(Y)\nmatrix B = inv(X'*X)*X'*Y\n"
        "display B[1,1]",
        df,
    )
    want = np.linalg.lstsq(df[["x", "w"]].to_numpy(), df.y.to_numpy(), rcond=None)[0]
    assert b == pytest.approx(want[0], rel=1e-10)
    assert run("matrix A = (2, 0 \\ 0, 4)\nmatrix B = 3*inv(A) + I(2) - A/2\n"
               "display B[2,2]", df) == pytest.approx(-0.25)  # fmt: skip
    assert (
        run("matrix A = (2, 0 \\ 0, 4)\nmatrix t = trace(A)\ndisplay t[1,1]", df) == 6
    )


def test_e_b_has_the_constant_last(df):
    fit = sp.regress("y ~ x + w", data=df)
    assert run("reg y x w\nmatrix b = e(b)\ndisplay b[1,3]", df) == float(
        fit.params["Intercept"]
    )
    assert run("reg y x w\nmatrix b = e(b)\ndisplay b[1,1]", df) == float(
        fit.params["x"]
    )
    se = run("reg y x w\nmatrix V = e(V)\ndisplay sqrt(V[2,2])", df)
    assert se == pytest.approx(float(fit.std_errors["w"]), rel=1e-12)
    # a fitted value built from the matrix is `predict`'s
    out = run("reg y x w\nmatrix b = e(b)\ngen double yh = b[1,3] + b[1,1]*x + b[1,2]*w\n"
              "predict double yp\ngen double d = abs(yh - yp)\nsu d\ndisplay r(max)", df)  # fmt: skip
    assert out < 1e-12


def test_svmat_and_mkmat(df):
    out = held(df, "matrix T = J(5, 1, .)\nforvalues i = 1/5 {\n matrix T[`i',1] = `i'^2\n}\n"
               "clear\nsvmat T, names(t)")  # fmt: skip
    assert out["t1"].tolist() == [1, 4, 9, 16, 25]
    out = held(df, 'matrix T = J(2, 2, 3)\nmatrix colnames T = "a" "b"\nclear\n'
               "svmat double T, names(col)")  # fmt: skip
    assert list(out.columns) == ["a", "b"]
    assert run("mkmat x w in 1/3, matrix(M)\ndisplay rowsof(M) + colsof(M)", df) == 5


def test_bootstrap_by_hand_marks_its_results_random(df):
    """The book's bootstrap: resample, estimate, store in a matrix, svmat."""
    session = StataSession(df)
    script = (
        "set seed 1\nmatrix taus = J(40, 1, .)\nforvalues i = 1/40 {\n quietly {\n"
        "  preserve\n  bsample\n  reg y x\n  matrix define taus[`i',1] = _b[x]\n"
        '  restore\n }\n}\nclear\nsvmat taus, names("taus")'
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in script.split("\n"):
            session.run(line)
    draws = session.data["taus1"]
    assert len(draws) == 40 and draws.notna().all() and draws.std() > 0
    assert abs(draws.mean() - 0.3) < 0.1
    assert session.simulated  # numbers that follow are not comparable


# ------------------------------------------------- synth, keep() in a session
def test_synth_keep_leaves_the_paths_in_the_session():
    d = sp.datasets.california_prop99()
    codes = {s: i + 1.0 for i, s in enumerate(sorted(d.state.unique()))}
    d = d.assign(sid=d.state.map(codes))
    session = StataSession(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("tsset sid year")
        session.run(
            f"synth cigsale cigsale(1980) cigsale(1988), trunit({codes['California']:.0f}) "
            "trperiod(1989) keep(res) replace"
        )
        session.run("use res, clear")
    assert list(session.data.columns) == ["_time", "_Y_treated", "_Y_synthetic"]
    assert len(session.data) == d.year.nunique()


# ----------------------------------------------------------- causal forest
def test_causal_forest_accepts_house_style_names():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    d["t"] = (rng.uniform(size=n) < 0.5).astype(float)
    d["y"] = d.a + 0.5 * d.t + rng.normal(size=n)
    kw = dict(data=d, y="y", n_estimators=200, random_state=1)
    house = sp.causal_forest(treat="t", covariates=["a", "b"], **kw)
    short = sp.causal_forest(d="t", x=["a", "b"], **kw)
    assert house.average_treatment_effect()["estimate"] == (
        short.average_treatment_effect()["estimate"]
    )
