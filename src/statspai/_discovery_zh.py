"""Chinese vocabulary for :func:`statspai.search_functions`.

A query written in Chinese used to return ``[]``: the registry is described
in English and the ranker splits on whitespace, which Chinese does not use.
Each entry maps one Chinese econometrics term to the English wording the
ranker already understands (its phrases, synonyms and concept targets live in
:mod:`statspai._discovery_search`), so the translation step adds vocabulary
and no second ranking rule.

The terms are written as unicode escapes with the characters in the comment
above each entry. The source archive built for the JSS replication package
transliterates non-ASCII source text; an escape survives that, a literal
would not. ``tests/test_discovery_search_zh.py`` checks that every comment
still spells its entry.
"""

from __future__ import annotations

import re
from typing import Tuple

#: ``(Chinese term, English query text)``. Longer terms are applied first, so
#: a term may contain a shorter one.
ZH_TERMS: Tuple[Tuple[str, str], ...] = (
    # 示例数据集
    ("\u793a\u4f8b\u6570\u636e\u96c6", "load example dataset"),
    # 示例数据
    ("\u793a\u4f8b\u6570\u636e", "load example dataset"),
    # 内置数据
    ("\u5185\u7f6e\u6570\u636e", "load example dataset"),
    # 数据集
    ("\u6570\u636e\u96c6", "dataset"),
    # 双重差分
    ("\u53cc\u91cd\u5dee\u5206", "difference in differences"),
    # 倍差法
    ("\u500d\u5dee\u6cd5", "difference in differences"),
    # 差分中的差分
    ("\u5dee\u5206\u4e2d\u7684\u5dee\u5206", "difference in differences"),
    # 三重差分
    ("\u4e09\u91cd\u5dee\u5206", "triple difference"),
    # 交错处理
    ("\u4ea4\u9519\u5904\u7406", "staggered adoption"),
    # 交错采纳
    ("\u4ea4\u9519\u91c7\u7eb3", "staggered adoption"),
    # 交错
    ("\u4ea4\u9519", "staggered"),
    # 多期
    ("\u591a\u671f", "staggered"),
    # 分批实施
    ("\u5206\u6279\u5b9e\u65bd", "staggered adoption"),
    # 分批
    ("\u5206\u6279", "staggered"),
    # 渐进
    ("\u6e10\u8fdb", "staggered"),
    # 试点
    ("\u8bd5\u70b9", "policy staggered"),
    # 事件研究
    ("\u4e8b\u4ef6\u7814\u7a76", "event study"),
    # 平行趋势
    ("\u5e73\u884c\u8d8b\u52bf", "parallel trends"),
    # 事前趋势
    ("\u4e8b\u524d\u8d8b\u52bf", "pretrend"),
    # 预趋势
    ("\u9884\u8d8b\u52bf", "pretrend"),
    # 合成双重差分
    ("\u5408\u6210\u53cc\u91cd\u5dee\u5206", "synthetic difference in differences"),
    # 合成控制
    ("\u5408\u6210\u63a7\u5236", "synthetic control"),
    # 断点回归
    ("\u65ad\u70b9\u56de\u5f52", "regression discontinuity"),
    # 断点
    ("\u65ad\u70b9", "regression discontinuity"),
    # 拐点回归
    ("\u62d0\u70b9\u56de\u5f52", "regression kink"),
    # 带宽
    ("\u5e26\u5bbd", "bandwidth"),
    # 操纵检验
    ("\u64cd\u7eb5\u68c0\u9a8c", "manipulation density"),
    # 聚束
    ("\u805a\u675f", "bunching"),
    # 工具变量
    ("\u5de5\u5177\u53d8\u91cf", "instrumental variables"),
    # 弱工具变量
    ("\u5f31\u5de5\u5177\u53d8\u91cf", "weak instruments"),
    # 弱工具
    ("\u5f31\u5de5\u5177", "weak instruments"),
    # 两阶段最小二乘
    ("\u4e24\u9636\u6bb5\u6700\u5c0f\u4e8c\u4e58", "two stage least squares"),
    # 内生
    ("\u5185\u751f", "iv endogenous"),
    # 局部平均处理效应
    (
        "\u5c40\u90e8\u5e73\u5747\u5904\u7406\u6548\u5e94",
        "local average treatment effect",
    ),
    # 倾向得分匹配
    ("\u503e\u5411\u5f97\u5206\u5339\u914d", "propensity score matching"),
    # 倾向得分
    ("\u503e\u5411\u5f97\u5206", "propensity score"),
    # 倾向评分
    ("\u503e\u5411\u8bc4\u5206", "propensity score"),
    # 匹配
    ("\u5339\u914d", "matching"),
    # 熵平衡
    ("\u71b5\u5e73\u8861", "ebalance entropy balancing"),
    # 逆概率加权
    ("\u9006\u6982\u7387\u52a0\u6743", "inverse probability weighting"),
    # 双重稳健
    ("\u53cc\u91cd\u7a33\u5065", "doubly robust"),
    # 双稳健
    ("\u53cc\u7a33\u5065", "doubly robust"),
    # 双重机器学习
    ("\u53cc\u91cd\u673a\u5668\u5b66\u4e60", "double machine learning"),
    # 双重/去偏机器学习
    ("\u53cc\u91cd/\u53bb\u504f\u673a\u5668\u5b66\u4e60", "double machine learning"),
    # 因果森林
    ("\u56e0\u679c\u68ee\u6797", "causal forest"),
    # 随机森林
    ("\u968f\u673a\u68ee\u6797", "forest"),
    # 元学习器
    ("\u5143\u5b66\u4e60\u5668", "metalearner"),
    # 异质性处理效应
    ("\u5f02\u8d28\u6027\u5904\u7406\u6548\u5e94", "heterogeneous treatment effects"),
    # 异质性
    ("\u5f02\u8d28\u6027", "heterogeneity"),
    # 条件平均处理效应
    (
        "\u6761\u4ef6\u5e73\u5747\u5904\u7406\u6548\u5e94",
        "conditional average treatment effect",
    ),
    # 处理组平均处理效应
    (
        "\u5904\u7406\u7ec4\u5e73\u5747\u5904\u7406\u6548\u5e94",
        "average treatment effect on the treated",
    ),
    # 平均处理效应
    ("\u5e73\u5747\u5904\u7406\u6548\u5e94", "average treatment effect"),
    # 处理效应
    ("\u5904\u7406\u6548\u5e94", "treatment effect"),
    # 分位数处理效应
    ("\u5206\u4f4d\u6570\u5904\u7406\u6548\u5e94", "quantile treatment effect"),
    # 分位数回归
    ("\u5206\u4f4d\u6570\u56de\u5f52", "quantile regression qreg"),
    # 中介效应
    ("\u4e2d\u4ecb\u6548\u5e94", "mediation analysis"),
    # 中介
    ("\u4e2d\u4ecb", "mediation"),
    # 调节效应
    ("\u8c03\u8282\u6548\u5e94", "interaction moderation"),
    # 敏感性分析
    ("\u654f\u611f\u6027\u5206\u6790", "sensitivity analysis"),
    # 遗漏变量
    ("\u9057\u6f0f\u53d8\u91cf", "omitted variable bias"),
    # 未观测混杂
    ("\u672a\u89c2\u6d4b\u6df7\u6742", "unobserved confounding"),
    # 混杂
    ("\u6df7\u6742", "confounding"),
    # 稳健性检验
    ("\u7a33\u5065\u6027\u68c0\u9a8c", "robustness"),
    # 稳健性
    ("\u7a33\u5065\u6027", "robustness"),
    # 安慰剂检验
    ("\u5b89\u6170\u5242\u68c0\u9a8c", "placebo"),
    # 安慰剂
    ("\u5b89\u6170\u5242", "placebo"),
    # 设定曲线
    ("\u8bbe\u5b9a\u66f2\u7ebf", "specification curve spec_curve"),
    # 多重检验
    ("\u591a\u91cd\u68c0\u9a8c", "multiple testing"),
    # 多重假设检验
    ("\u591a\u91cd\u5047\u8bbe\u68c0\u9a8c", "multiple hypothesis testing"),
    # 目标试验
    ("\u76ee\u6807\u8bd5\u9a8c", "target trial"),
    # 孟德尔随机化
    ("\u5b5f\u5fb7\u5c14\u968f\u673a\u5316", "mendelian randomization"),
    # 因果发现
    ("\u56e0\u679c\u53d1\u73b0", "causal discovery"),
    # 有向无环图
    ("\u6709\u5411\u65e0\u73af\u56fe", "dag"),
    # 因果图
    ("\u56e0\u679c\u56fe", "dag"),
    # 后门
    ("\u540e\u95e8", "backdoor adjustment"),
    # 前门
    ("\u524d\u95e8", "front door"),
    # 政策学习
    ("\u653f\u7b56\u5b66\u4e60", "policy learning"),
    # 政策树
    ("\u653f\u7b56\u6811", "policy tree"),
    # 政策
    ("\u653f\u7b56", "policy"),
    # 溢出效应
    ("\u6ea2\u51fa\u6548\u5e94", "spillover interference"),
    # 干扰
    ("\u5e72\u6270", "interference"),
    # 主分层
    ("\u4e3b\u5206\u5c42", "principal stratification"),
    # 份额移动
    ("\u4efd\u989d\u79fb\u52a8", "shift share"),
    # 巴蒂克
    ("\u5df4\u8482\u514b", "bartik"),
    # 面板数据
    ("\u9762\u677f\u6570\u636e", "panel"),
    # 面板
    ("\u9762\u677f", "panel"),
    # 双向固定效应
    ("\u53cc\u5411\u56fa\u5b9a\u6548\u5e94", "two way fixed effects"),
    # 高维固定效应
    ("\u9ad8\u7ef4\u56fa\u5b9a\u6548\u5e94", "hdfe fixed effects"),
    # 固定效应
    ("\u56fa\u5b9a\u6548\u5e94", "fixed effects"),
    # 随机效应
    ("\u968f\u673a\u6548\u5e94", "random effects"),
    # 豪斯曼检验
    ("\u8c6a\u65af\u66fc\u68c0\u9a8c", "hausman"),
    # 豪斯曼
    ("\u8c6a\u65af\u66fc", "hausman"),
    # 动态面板
    ("\u52a8\u6001\u9762\u677f", "dynamic panel xtabond"),
    # 系统广义矩
    ("\u7cfb\u7edf\u5e7f\u4e49\u77e9", "xtdpdsys gmm"),
    # 广义矩
    ("\u5e7f\u4e49\u77e9", "gmm"),
    # 多层模型
    ("\u591a\u5c42\u6a21\u578b", "multilevel mixed"),
    # 混合效应
    ("\u6df7\u5408\u6548\u5e94", "mixed effects"),
    # 分层模型
    ("\u5206\u5c42\u6a21\u578b", "multilevel"),
    # 最小二乘
    ("\u6700\u5c0f\u4e8c\u4e58", "ols regress"),
    # 线性回归
    ("\u7ebf\u6027\u56de\u5f52", "regress ols"),
    # 回归表
    ("\u56de\u5f52\u8868", "regtable"),
    # 回归
    ("\u56de\u5f52", "regression"),
    # 逻辑回归
    ("\u903b\u8f91\u56de\u5f52", "logit"),
    # 泊松回归
    ("\u6cca\u677e\u56de\u5f52", "poisson"),
    # 泊松
    ("\u6cca\u677e", "poisson"),
    # 负二项
    ("\u8d1f\u4e8c\u9879", "nbreg negative binomial"),
    # 计数模型
    ("\u8ba1\u6570\u6a21\u578b", "count poisson"),
    # 有序
    ("\u6709\u5e8f", "ordered ologit"),
    # 多项
    ("\u591a\u9879", "multinomial mlogit"),
    # 截断
    ("\u622a\u65ad", "truncated truncreg"),
    # 删失
    ("\u5220\u5931", "censored tobit"),
    # 样本选择
    ("\u6837\u672c\u9009\u62e9", "heckman selection"),
    # 赫克曼
    ("\u8d6b\u514b\u66fc", "heckman"),
    # 广义线性模型
    ("\u5e7f\u4e49\u7ebf\u6027\u6a21\u578b", "glm"),
    # 边际效应
    ("\u8fb9\u9645\u6548\u5e94", "marginal effects"),
    # 稳健标准误
    ("\u7a33\u5065\u6807\u51c6\u8bef", "robust standard errors"),
    # 聚类标准误
    ("\u805a\u7c7b\u6807\u51c6\u8bef", "cluster standard errors"),
    # 聚类
    ("\u805a\u7c7b", "cluster"),
    # 标准误
    ("\u6807\u51c6\u8bef", "standard errors"),
    # 异方差
    ("\u5f02\u65b9\u5dee", "heteroskedasticity"),
    # 自助法
    ("\u81ea\u52a9\u6cd5", "bootstrap"),
    # 野自助
    ("\u91ce\u81ea\u52a9", "wild cluster bootstrap"),
    # 刀切法
    ("\u5200\u5207\u6cd5", "jackknife"),
    # 置信区间
    ("\u7f6e\u4fe1\u533a\u95f4", "confidence interval"),
    # 假设检验
    ("\u5047\u8bbe\u68c0\u9a8c", "test"),
    # 联合检验
    ("\u8054\u5408\u68c0\u9a8c", "joint test wald"),
    # 共线性
    ("\u5171\u7ebf\u6027", "collinearity vif"),
    # 缺失值
    ("\u7f3a\u5931\u503c", "missing imputation"),
    # 多重插补
    ("\u591a\u91cd\u63d2\u8865", "multiple imputation"),
    # 缩尾
    ("\u7f29\u5c3e", "winsor"),
    # 描述统计
    ("\u63cf\u8ff0\u7edf\u8ba1", "summary statistics sumstats"),
    # 平衡性检验
    ("\u5e73\u8861\u6027\u68c0\u9a8c", "balance table"),
    # 平衡表
    ("\u5e73\u8861\u8868", "balance table"),
    # 样本量
    ("\u6837\u672c\u91cf", "sample size"),
    # 功效分析
    ("\u529f\u6548\u5206\u6790", "power analysis"),
    # 统计功效
    ("\u7edf\u8ba1\u529f\u6548", "power analysis"),
    # 功效
    ("\u529f\u6548", "power"),
    # 分解
    ("\u5206\u89e3", "decomposition"),
    # 奥萨卡
    ("\u5965\u8428\u5361", "oaxaca"),
    # 不平等
    ("\u4e0d\u5e73\u7b49", "inequality"),
    # 随机前沿
    ("\u968f\u673a\u524d\u6cbf", "stochastic frontier"),
    # 生产函数
    ("\u751f\u4ea7\u51fd\u6570", "production function"),
    # 需求估计
    ("\u9700\u6c42\u4f30\u8ba1", "blp demand"),
    # 结构模型
    ("\u7ed3\u6784\u6a21\u578b", "structural"),
    # 时间序列
    ("\u65f6\u95f4\u5e8f\u5217", "time series"),
    # 单位根
    ("\u5355\u4f4d\u6839", "unit root"),
    # 协整
    ("\u534f\u6574", "cointegration"),
    # 向量自回归
    ("\u5411\u91cf\u81ea\u56de\u5f52", "var"),
    # 脉冲响应
    ("\u8109\u51b2\u54cd\u5e94", "impulse response irf"),
    # 格兰杰因果
    ("\u683c\u5170\u6770\u56e0\u679c", "granger causality"),
    # 格兰杰
    ("\u683c\u5170\u6770", "granger"),
    # 预测
    ("\u9884\u6d4b", "forecast"),
    # 季节
    ("\u5b63\u8282", "seasonal"),
    # 波动率
    ("\u6ce2\u52a8\u7387", "garch volatility"),
    # 结构突变
    ("\u7ed3\u6784\u7a81\u53d8", "structural break"),
    # 间断时间序列
    ("\u95f4\u65ad\u65f6\u95f4\u5e8f\u5217", "interrupted time series"),
    # 中断时间序列
    ("\u4e2d\u65ad\u65f6\u95f4\u5e8f\u5217", "interrupted time series"),
    # 局部投影
    ("\u5c40\u90e8\u6295\u5f71", "local projection"),
    # 状态空间
    ("\u72b6\u6001\u7a7a\u95f4", "state space"),
    # 卡尔曼
    ("\u5361\u5c14\u66fc", "kalman"),
    # 空间计量
    ("\u7a7a\u95f4\u8ba1\u91cf", "spatial"),
    # 空间自相关
    ("\u7a7a\u95f4\u81ea\u76f8\u5173", "moran spatial"),
    # 空间
    ("\u7a7a\u95f4", "spatial"),
    # 限制平均生存时间
    ("\u9650\u5236\u5e73\u5747\u751f\u5b58\u65f6\u95f4", "rmst"),
    # 生存时间处理效应
    ("\u751f\u5b58\u65f6\u95f4\u5904\u7406\u6548\u5e94", "stteffects"),
    # 生存分析
    ("\u751f\u5b58\u5206\u6790", "survival"),
    # 比例风险
    ("\u6bd4\u4f8b\u98ce\u9669", "cox"),
    # 生存
    ("\u751f\u5b58", "survival"),
    # 贝叶斯
    ("\u8d1d\u53f6\u65af", "bayes"),
    # 后验
    ("\u540e\u9a8c", "posterior"),
    # 先验
    ("\u5148\u9a8c", "prior"),
    # 机器学习
    ("\u673a\u5668\u5b66\u4e60", "machine learning"),
    # 套索
    ("\u5957\u7d22", "lasso"),
    # 岭回归
    ("\u5cad\u56de\u5f52", "ridge"),
    # 交叉验证
    ("\u4ea4\u53c9\u9a8c\u8bc1", "cross validation"),
    # 共形
    ("\u5171\u5f62", "conformal"),
    # 公平性
    ("\u516c\u5e73\u6027", "fairness"),
    # 随机对照试验
    ("\u968f\u673a\u5bf9\u7167\u8bd5\u9a8c", "randomized experiment rct"),
    # 随机实验
    ("\u968f\u673a\u5b9e\u9a8c", "randomized experiment"),
    # 实验设计
    ("\u5b9e\u9a8c\u8bbe\u8ba1", "experimental design"),
    # 随机化推断
    ("\u968f\u673a\u5316\u63a8\u65ad", "randomization inference"),
    # 抽样权重
    ("\u62bd\u6837\u6743\u91cd", "survey weights"),
    # 调查数据
    ("\u8c03\u67e5\u6570\u636e", "survey"),
    # 权重
    ("\u6743\u91cd", "weights"),
    # 剂量反应
    ("\u5242\u91cf\u53cd\u5e94", "dose response"),
    # 连续处理
    ("\u8fde\u7eed\u5904\u7406", "continuous treatment dose"),
    # 选择偏误
    ("\u9009\u62e9\u504f\u8bef", "selection bias"),
    # 边界
    ("\u8fb9\u754c", "bounds"),
    # 部分识别
    ("\u90e8\u5206\u8bc6\u522b", "partial identification bounds"),
    # 协变量
    ("\u534f\u53d8\u91cf", "covariates"),
    # 控制变量
    ("\u63a7\u5236\u53d8\u91cf", "controls"),
    # 处理
    ("\u5904\u7406", "treatment"),
    # 导出
    ("\u5bfc\u51fa", "export"),
    # 作图
    ("\u4f5c\u56fe", "plot"),
    # 画图
    ("\u753b\u56fe", "plot"),
    # 表格
    ("\u8868\u683c", "table"),
    # 引用
    ("\u5f15\u7528", "citation bibtex"),
    # 文献
    ("\u6587\u732e", "citation"),
)

_ORDERED = tuple(sorted(ZH_TERMS, key=lambda kv: -len(kv[0])))
#: CJK ideographs, CJK punctuation and full-width forms.
_CJK_RE = re.compile("[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]+")


def has_cjk(text: str) -> bool:
    """Whether text contains Chinese characters or punctuation."""
    return bool(_CJK_RE.search(text))


def translate_query(query: str) -> str:
    """Replace known Chinese terms by English wording; drop the rest.

    Characters that belong to no known term (particles, pronouns, domain
    nouns) are removed, the same way English stopwords are ignored. ASCII
    parts of the query (``DID``, ``Stata``, a function name) pass through.
    """
    if not has_cjk(query):
        return query
    out = query
    for zh, en in _ORDERED:
        if zh in out:
            out = out.replace(zh, f" {en} ")
    return _CJK_RE.sub(" ", out)


__all__ = ["ZH_TERMS", "has_cjk", "translate_query"]
