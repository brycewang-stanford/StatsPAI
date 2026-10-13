"""``sp.search_functions`` answers a query written in Chinese.

Until 1.39.6 a Chinese query returned ``[]``: the ranker splits on whitespace
and every description is English. Known terms are now rewritten to the
English vocabulary before ranking (``statspai._discovery_zh``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import statspai as sp
from statspai import _discovery_zh as zh

# (query, a function that must be in the top three). Queries are spelled as
# escapes for the same reason the vocabulary is; the comment is the text.
QUERIES = [
    # 我有面板数据，政策分批实施
    (
        "我有面板数据，政策" "分批实施",
        "callaway_santanna",
    ),
    # 断点回归怎么做
    ("断点回归怎么做", "rdrobust"),
    # 工具变量 弱工具检验
    ("工具变量 弱工具检验", "iv"),
    # 合成控制法
    ("合成控制法", "synth"),
    # 倾向得分匹配
    ("倾向得分匹配", "match"),
    # 异质性处理效应
    ("异质性处理效应", "metalearner"),
    # 双重差分的平行趋势检验
    (
        "双重差分的平行趋势检验",
        "did",
    ),
    # 中介效应分析
    ("中介效应分析", "mediate"),
    # 用DID估计最低工资的影响
    (
        "用DID估计最低工资的影响",
        "did",
    ),
    # 敏感性分析 遗漏变量
    (
        "敏感性分析 遗漏变量",
        "sensemakr",
    ),
    # 双重机器学习
    ("双重机器学习", "dml"),
    # 事件研究
    ("事件研究", "event_study"),
]


@pytest.mark.parametrize("query,expected", QUERIES)
def test_chinese_query_reaches_the_estimator(query, expected):
    top = [hit["name"] for hit in sp.search_functions(query)[:3]]
    assert expected in top, top


def test_chinese_and_english_spellings_agree():
    # 合成控制 / synthetic control
    assert [h["name"] for h in sp.search_functions("合成控制")] == [
        h["name"] for h in sp.search_functions("synthetic control")
    ]


def test_english_queries_are_untouched():
    assert zh.translate_query("staggered adoption") == "staggered adoption"
    assert not zh.has_cjk("difference in differences")


def test_longer_terms_win_over_the_terms_they_contain():
    # 合成双重差分 must not be read as 合成 + 双重差分.
    text = zh.translate_query("合成双重差分")
    assert "synthetic difference in differences" in text
    assert sp.search_functions("合成双重差分")[0]["name"] == "sdid"


def test_a_query_with_no_known_term_returns_nothing_rather_than_noise():
    # 你好
    assert sp.search_functions("你好") == []


def test_vocabulary_is_ascii_source_and_comments_spell_the_entries():
    """The escapes and the readable comment above each must not drift."""
    path = Path(zh.__file__)
    source = path.read_text(encoding="utf-8")
    code_lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith("#")]
    assert "\n".join(code_lines).isascii()
    block = source[source.index("ZH_TERMS") : source.index("_ORDERED")]
    comments = re.findall(r"^\s*# (\S+)\s*$", block, flags=re.M)
    assert comments == [term for term, _english in zh.ZH_TERMS]
    assert len(set(comments)) == len(comments)
    for _term, english in zh.ZH_TERMS:
        assert english.isascii() and english == english.lower()
