"""Value labels in use: ``sp.label_values`` / ``sp.decode`` / ``sp.describe``
/ ``sp.tab`` / ``sp.regtable(labels=)`` / ``sp.label_vars`` from a frame, and
the ``label`` commands of ``sp.stata``.

The expectations follow Stata: ``tabulate`` prints labels in the order of the
codes, ``decode`` gives the label text, ``label define`` refuses a name that
exists unless told ``add`` / ``modify`` / ``replace``.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.exceptions import MethodIncompatibility
from statspai.utils.labels import term_labels

EXTMISS = Path(__file__).parent / "fixtures" / "dta_labels" / "extmiss118.dta"


@pytest.fixture
def survey():
    rng = np.random.default_rng(7)
    n = 240
    df = pd.DataFrame(
        {
            "wage": rng.normal(10, 2, n),
            "tenure": rng.normal(5, 1, n),
            # codes whose label order is not their alphabetical order
            "edu": rng.integers(1, 4, n),
            "female": rng.integers(0, 2, n),
            "firm": rng.integers(0, 12, n),
        }
    )
    sp.label_vars(df, {"wage": "Hourly wage", "tenure": "Tenure", "edu": "Education"})
    sp.label_values(df, "edu", {1: "tertiary", 2: "secondary", 3: "primary"})
    sp.label_values(df, "female", {0: "No", 1: "Yes", ".a": "Refused"})
    return df


# ------------------------------------------------------------ label_values
def test_label_values_stores_codes_and_missing_codes_apart(survey):
    assert survey.attrs["_value_labels"]["female"] == {0: "No", 1: "Yes"}
    assert survey.attrs["_missing_labels"] == {"female": {".a": "Refused"}}
    # one mapping for several variables; each gets its own copy
    sp.label_values(survey, ["female", "edu"], {1: "one"})
    survey.attrs["_value_labels"]["female"][2] = "two"
    assert survey.attrs["_value_labels"]["edu"] == {1: "one"}
    # None removes
    sp.label_values(survey, "edu", None)
    assert "edu" not in survey.attrs["_value_labels"]
    assert "_missing_labels" not in survey.attrs


@pytest.mark.parametrize(
    "var, labels, match",
    [
        ("nope", {1: "x"}, "not found"),
        ("edu", {1.5: "x"}, "integer codes"),
        ("edu", {"maybe": "x"}, "neither an integer"),
    ],
)
def test_label_values_rejects_what_stata_cannot_store(survey, var, labels, match):
    with pytest.raises(ValueError, match=match):
        sp.label_values(survey, var, labels)


def test_label_values_rejects_a_string_column():
    df = pd.DataFrame({"city": ["a", "b"]})
    with pytest.raises(ValueError, match="not numeric"):
        sp.label_values(df, "city", {1: "x"})


# ------------------------------------------------------------------ decode
def test_decode_follows_the_codes_not_the_alphabet(survey):
    out = sp.decode(survey)
    assert list(out["edu"].cat.categories) == ["tertiary", "secondary", "primary"]
    assert out["edu"].cat.ordered
    # text matches code row by row, and the input still holds codes
    expected = survey["edu"].map({1: "tertiary", 2: "secondary", 3: "primary"})
    assert out["edu"].astype(str).tolist() == expected.tolist()
    assert pd.api.types.is_integer_dtype(survey["edu"])
    # decoded columns stop being listed as coded; variable labels stay
    assert "_value_labels" not in out.attrs
    assert sp.get_label(out, "edu") == "Education"
    assert "edu" in survey.attrs["_value_labels"]


def test_decode_keeps_unlabelled_codes_and_separates_shared_texts():
    df = pd.DataFrame({"q": [1, 2, 3, 9, np.nan]})
    sp.label_values(df, "q", {1: "agree", 2: "agree", 3: "disagree"})
    out = sp.decode(df, "q")["q"]
    assert out.tolist()[:4] == ["agree (1)", "agree (2)", "disagree", "9"]
    assert pd.isna(out.iloc[4])


def test_decode_shows_extended_missing_labels_on_request():
    df = sp.read_data(str(EXTMISS), extended_missing="column")
    plain = sp.decode(df, "region")["region"]
    assert int(plain.isna().sum()) == 4
    shown = sp.decode(df, "region", missing=True)["region"].set_axis(df["id"])
    assert shown[3] == "Refused" and shown[9] == "Don't know"
    assert pd.isna(shown[11])  # a plain '.' has no label to show
    # the missing labels come after the real codes
    assert list(shown.cat.categories)[-2:] == ["Refused", "Don't know"]


def test_decode_refuses_an_unlabelled_column(survey):
    with pytest.raises(ValueError, match="no value labels"):
        sp.decode(survey, "wage")
    with pytest.raises(ValueError, match="not found"):
        sp.decode(survey, "nope")


# ---------------------------------------------------------------- describe
def test_describe_lists_value_labels(survey):
    table = sp.describe(survey).set_index("variable")
    assert table.loc["female", "value_labels"] == "0=No, 1=Yes, .a=Refused"
    assert table.loc["wage", "value_labels"] == ""
    # a long label set is cut, with the number of codes
    sp.label_values(survey, "firm", {i: f"firm number {i}" for i in range(12)})
    cell = sp.describe(survey).set_index("variable").loc["firm", "value_labels"]
    assert len(cell) <= 60 and cell.endswith("... (12 codes)")
    # same columns on a frame without labels
    assert list(sp.describe(pd.DataFrame({"a": [1]})).columns) == [
        "variable",
        "type",
        "n",
        "n_missing",
        "label",
        "value_labels",
    ]


# --------------------------------------------------------------------- tab
def test_tab_prints_labels_in_code_order(survey):
    one = sp.tab(survey, "edu", output="dataframe")
    assert list(one.index) == ["tertiary", "secondary", "primary", "Total"]
    codes = sp.tab(survey, "edu", output="dataframe", labels=False)
    assert list(codes.index) == [1, 2, 3, "Total"]
    # the counts are the same table
    assert one["Freq"].tolist() == codes["Freq"].tolist()

    two = sp.tab(survey, "edu", "female", output="dataframe")
    raw = sp.tab(survey, "edu", "female", output="dataframe", labels=False)
    assert list(two.index) == ["tertiary", "secondary", "primary", "Total"]
    assert list(two.columns) == ["No", "Yes", "Total"]
    assert (two.to_numpy() == raw.to_numpy()).all()


def test_tab_without_labels_is_unchanged():
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": [0, 1, 0, 1]})
    assert sp.tab(df, "a", "b") == sp.tab(df, "a", "b", labels=False)
    assert sp.tab(df, "a") == sp.tab(df, "a", labels=False)


# ---------------------------------------------------------------- regtable
def test_term_labels_reads_each_naming_style(survey):
    names = [
        "Intercept",
        "tenure",
        "C(edu)[T.2]",
        "C(edu)[3]",
        "edu::3",
        "edu[T.2.0]",
        "female:tenure",
        "C(edu)[T.3]:tenure",
        "C(firm)[T.4]",
    ]
    assert term_labels(names, survey) == {
        "tenure": "Tenure",
        "C(edu)[T.2]": "Education: secondary",
        "C(edu)[3]": "Education: primary",
        "edu::3": "Education: primary",
        "edu[T.2.0]": "Education: secondary",
        "female:tenure": "female × Tenure",
        "C(edu)[T.3]:tenure": "Education: primary × Tenure",
        # nothing known about firm: left alone, as is Intercept
    }


def test_regtable_labels_rows_from_the_data(survey):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.regress("wage ~ tenure + C(edu) + female:tenure", data=survey)
        plain = sp.regtable(m).to_text()
        labelled = sp.regtable(m, labels=survey).to_text()
        override = sp.regtable(
            m, labels=survey, coef_labels={"tenure": "Years at firm"}
        ).to_text()
    assert "C(edu)[T.2]" in plain and "Education" not in plain
    assert "Education: secondary" in labelled and "C(edu)" not in labelled
    assert "female × Tenure" in labelled
    # only the row names differ: the standard-error lines are the same

    def se_lines(text):
        return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("(")]

    assert se_lines(plain) == se_lines(labelled) and len(se_lines(plain)) == 6
    assert "Years at firm" in override and "Education: secondary" in override


def test_regtable_labels_argument_is_checked(survey):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.regress("wage ~ tenure", data=survey)
    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        sp.regtable(m, labels={"tenure": "Tenure"})
    with pytest.raises(MethodIncompatibility, match="coef_map or labels"):
        sp.regtable(m, labels=survey, coef_map={"tenure": "Tenure"})


# ------------------------------------------------- labels lost by pandas
def test_label_vars_copies_labels_back_after_a_merge(survey):
    other = pd.DataFrame({"firm": np.arange(12), "size": np.arange(12) * 10})
    merged = survey.merge(other, on="firm")
    assert merged.attrs == {}  # what pandas does; the reason for the copy
    sp.label_vars(merged, survey)
    assert sp.get_label(merged, "wage") == "Hourly wage"
    assert merged.attrs["_value_labels"] == survey.attrs["_value_labels"]
    assert merged.attrs["_missing_labels"] == survey.attrs["_missing_labels"]
    # independent of the source afterwards
    merged.attrs["_value_labels"]["edu"][1] = "changed"
    assert survey.attrs["_value_labels"]["edu"][1] == "tertiary"


def test_label_vars_copy_keeps_existing_labels_and_follows_a_rename(survey):
    short = survey.rename(columns={"wage": "w", "edu": "school"})
    sp.label_var(short, "tenure", "Mine")
    sp.label_vars(short, survey, rename={"wage": "w", "edu": "school"})
    assert sp.get_label(short, "w") == "Hourly wage"
    assert sp.get_label(short, "tenure") == "Mine"
    assert short.attrs["_value_labels"]["school"][2] == "secondary"
    with pytest.raises(ValueError, match="not in the source"):
        sp.label_vars(short, survey, rename={"nope": "x"})
    with pytest.raises(ValueError, match="only when labels is a DataFrame"):
        sp.label_vars(short, {"w": "x"}, rename={"a": "b"})


# ------------------------------------------------------- sp.stata: label
@pytest.fixture
def session():
    rng = np.random.default_rng(3)
    n = 120
    data = pd.DataFrame(
        {
            "wage": rng.normal(10, 2, n),
            "edu": rng.integers(1, 4, n),
            "union": rng.integers(0, 2, n),
            "female": rng.integers(0, 2, n),
            "city": rng.choice(["b", "a", "c"], n),
        }
    )
    return StataSession(data), data


def run(session, *lines):
    for line in lines:
        session.run(line)


def test_stata_label_commands_change_the_labels(session):
    s, original = session
    run(
        s,
        'label variable wage "Hourly wage, USD"',
        "la var edu Education",
        'label define yn 0 "No" 1 "Yes" .a "Refused"',
        "label values union female yn",
        'lab def edul 1 "primary" 2 "secondary, lower" 3 `"tertiary"\'',
        "label val edu edul",
        'label data "demo"',
    )
    attrs = s.data.attrs
    assert attrs["_labels"] == {"wage": "Hourly wage, USD", "edu": "Education"}
    assert attrs["_value_labels"]["union"] == {0: "No", 1: "Yes"}
    assert attrs["_value_labels"]["edu"][2] == "secondary, lower"
    assert attrs["_missing_labels"]["female"] == {".a": "Refused"}
    assert attrs["_data_label"] == "demo"
    # the caller's frame is never touched
    assert original.attrs == {}
    # a later change to the set reaches every variable that uses it
    run(s, 'label define yn 1 "Yes!", modify')
    assert s.data.attrs["_value_labels"]["female"][1] == "Yes!"
    assert s.data.attrs["_value_labels"]["union"][1] == "Yes!"
    # detach, and drop
    run(s, "label values union .")
    assert "union" not in s.data.attrs["_value_labels"]
    run(s, "label drop edul")
    assert "edu" not in s.data.attrs["_value_labels"]


def test_stata_label_define_follows_statas_rules(session):
    s, _ = session
    run(s, 'label define yn 0 "No" 1 "Yes"')
    with pytest.raises(MethodIncompatibility, match="already defined"):
        s.run('label define yn 0 "Nope"')
    with pytest.raises(MethodIncompatibility, match="use `modify`"):
        s.run('label define yn 1 "Si", add')
    run(s, 'label define yn 2 "Maybe", add', "label values union yn")
    assert s.data.attrs["_value_labels"]["union"] == {0: "No", 1: "Yes", 2: "Maybe"}
    run(s, 'label define yn 5 "Only", replace')
    assert s.data.attrs["_value_labels"]["union"] == {5: "Only"}
    with pytest.raises(MethodIncompatibility, match="not in the data"):
        s.run("label values nope yn")
    with pytest.raises(MethodIncompatibility, match="not in the data"):
        s.run('label variable nope "x"')
    # label list only prints: skipped, as before
    run(s, "label list", "codebook union")


def test_stata_tabulate_prints_labels_unless_nolabel(session):
    s, data = session
    run(
        s,
        'label define edul 1 "tertiary" 2 "secondary" 3 "primary"',
        "label values edu edul",
        "tabulate edu",
    )
    labelled = s.output
    assert list(labelled.index) == ["tertiary", "secondary", "primary"]
    run(s, "tabulate edu, nolabel")
    assert list(s.output.index) == [1, 2, 3]
    assert labelled["Freq."].tolist() == s.output["Freq."].tolist()
    assert labelled["Freq."].tolist() == (
        data["edu"].value_counts().sort_index().tolist()
    )
    run(s, "tabulate edu union")
    assert list(s.output.index) == ["tertiary", "secondary", "primary", "Total"]


def test_stata_decode_encode_and_rename_carry_labels(session):
    s, data = session
    run(
        s,
        'label variable edu "Education"',
        'label define edul 1 "primary" 2 "secondary"',
        "label values edu edul",
        "decode edu, generate(edu_s)",
        "encode city, gen(city_n)",
        "rename edu school",
    )
    out = s.data
    # decode: the text, "" where the code has no label (3 is unlabelled)
    expected = data["edu"].map({1: "primary", 2: "secondary", 3: ""}).tolist()
    assert out["edu_s"].tolist() == expected
    assert sp.get_label(out, "edu_s") == "Education"
    # encode: codes 1..K in sorted order, labelled with the strings
    assert out.attrs["_value_labels"]["city_n"] == {1: "a", 2: "b", 3: "c"}
    assert (
        out["city_n"].map(out.attrs["_value_labels"]["city_n"]).tolist()
        == data["city"].tolist()
    )
    # rename: labels go with the variable, and stay tied to the set
    assert sp.get_label(out, "school") == "Education"
    assert "edu" not in out.attrs["_value_labels"]
    run(s, 'label define edul 3 "tertiary", add')
    assert s.data.attrs["_value_labels"]["school"][3] == "tertiary"
    with pytest.raises(MethodIncompatibility, match="no value label"):
        s.run("decode wage, generate(w)")


def test_stata_preserve_restore_puts_the_labels_back(session):
    s, _ = session
    run(
        s,
        'label define yn 0 "No" 1 "Yes"',
        "label values union yn",
        "preserve",
        'label define yn 1 "Changed", modify',
        'label variable wage "temp"',
        "restore",
    )
    assert s.data.attrs["_value_labels"]["union"] == {0: "No", 1: "Yes"}
    assert "_labels" not in s.data.attrs
    run(s, 'label define yn 1 "Again", modify')
    assert s.data.attrs["_value_labels"]["union"][1] == "Again"
