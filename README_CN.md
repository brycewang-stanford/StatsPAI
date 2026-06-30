[English](https://github.com/brycewang-stanford/statspai/blob/main/README.md) | [中文](https://github.com/brycewang-stanford/statspai/blob/main/README_CN.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/main/docs/logo/readme-1.png" alt="StatsPAI - Stata 与 R 的 Python 平替工具包" width="780">
</p>

# StatsPAI：面向实证研究的 Stata/R Python 平替

[![PyPI version](https://img.shields.io/pypi/v/StatsPAI.svg)](https://pypi.org/project/StatsPAI/)
[![Python versions](https://img.shields.io/pypi/pyversions/StatsPAI.svg)](https://pypi.org/project/StatsPAI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/brycewang-stanford/statspai/blob/main/LICENSE)
[![Tests](https://github.com/brycewang-stanford/statspai/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/brycewang-stanford/statspai/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/statspai?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/statspai)
[![status](https://joss.theoj.org/papers/9f1c837b1b1df7adfcdd538c3698e332/status.svg)](https://joss.theoj.org/papers/9f1c837b1b1df7adfcdd538c3698e332)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19933900-blue.svg)](https://doi.org/10.5281/zenodo.19933900)

StatsPAI 面向那些原本需要在 Stata、R 和 Python 之间来回切换的实证研究者。它的目标很直接：把常见的计量经济学、因果推断、诊断、稳健性、表格导出和 Agent 可读元数据，放到一个 Python-native API 里。

你可以把它理解成新项目的 Stata/R 平替入口：

- Stata 风格：`regress`、`ivregress`、`reghdfe`、`csdid`、`rdrobust`、`synth`、`psmatch2`、`outreg2`。
- R 风格：`lm`、`fixest`、`did`、`rdrobust`、`Synth`、`DoubleML`、`MatchIt`、`modelsummary`、`broom`。
- Python 输出：在支持的结果对象上直接用 `.summary()`、`.tidy()`、`.plot()`、`.to_latex()`、`.to_docx()`、`.to_agent_summary()`。
- Stata Agent 协同：我们自己开发的 [`stata-code`](https://github.com/brycewang-stanford/stata-code/)
  可以和 StatsPAI 配合，让 agent 更顺畅地理解既有 Stata 工作流、迁移到 Python，并做结果对照。
- Skills repo 协同：[`Auto-Empirical-Research-Skills`](https://github.com/brycewang-stanford/Auto-Empirical-Research-Skills)、
  [`AER-Skills`](https://github.com/brycewang-stanford/AER-Skills)、
  [`Awesome-Journal-Skills`](https://github.com/brycewang-stanford/Awesome-Journal-Skills)
  和 [`Paper-WorkFlow`](https://github.com/brycewang-stanford/Paper-WorkFlow)
  可以和 StatsPAI 以及 agent 一起使用，作为方法选择、期刊要求、论文流程和可复现检查的技能层。

这不是说每个 Stata/R 命令都已经逐字节复现。需要严肃对齐时，请看函数的 `validation_status`、参考对齐测试和 `sp.cross_validate`。

---

## 安装

```bash
pip install statspai
```

然后：

```python
import statspai as sp

print(sp.datasets.list_datasets()[["name", "design", "n_obs"]].head())
```

StatsPAI 随包带有 Card (1995)、Callaway-Sant'Anna `mpdta`、Lee (2008) RD、LaLonde/NSW、California Proposition 99 等教学数据集。下面的例子安装后可以离线运行。

一眼概览：1,139 个注册函数，分布在 87 个子模块；339k 行核心代码 + 182k 行测试。运行 `python scripts/registry_stats.py` 可复现这些数字。

---

## 如果你原来用 Stata 或 R

| 原来的工作流 | Stata / R 写法 | StatsPAI 入口 |
| --- | --- | --- |
| OLS / 稳健标准误 | `reg y x, vce(robust)` / `lm()` + `sandwich` | `sp.regress(..., robust="hc1")` |
| IV / 2SLS | `ivregress 2sls` / `AER::ivreg()` | `sp.ivreg("y ~ (d ~ z) + x", data=df)` |
| 高维固定效应 | `reghdfe` / `fixest::feols()` | `sp.feols("y ~ x | firm + year", data=df)` |
| 交错 DiD | `csdid` / `did::att_gt()` | `sp.callaway_santanna()` + `sp.aggte()` |
| 断点回归 | `rdrobust` / `rdrobust::rdrobust()` | `sp.rdrobust()` |
| 合成控制 | `synth` / `Synth::synth()` | `sp.synth()` |
| 匹配 / PSM | `psmatch2` / `MatchIt` | `sp.psmatch2()` 和 matching helpers |
| 论文表格 | `outreg2`、`esttab` / `modelsummary` | `sp.outreg2()`、`sp.modelsummary()` |

---

## 和其他 Python 因果推断包的简洁对比

StatsPAI 的定位是一个更宽的 Stata/R 风格实证研究工作台，而不是只覆盖某一种建模范式。

| 包 | 更适合什么 | StatsPAI 的不同点 |
| --- | --- | --- |
| [`causallib`](https://github.com/BiomedSciAI/causallib) | 观测数据因果推断，偏 scikit-learn 风格流程：IPW、匹配、标准化、双重稳健估计和评估。 | StatsPAI 更偏 Stata/R 平替：OLS、IV、高维固定效应、DiD、RD、合成控制、匹配、诊断、验证元数据和论文表格导出都在同一个 API 里。 |
| [`CausalPy`](https://github.com/pymc-labs/CausalPy) | 基于 PyMC 的 Bayesian quasi-experimental 因果分析，强调不确定性和可视化诊断。 | StatsPAI 优先服务熟悉 Stata/R 的实证工作流：常用计量命令、频率学派估计、跨 Stata/R 的验证证据、内置教学数据和 Agent 可读结果摘要。 |

如果主要想要 sklearn 风格的 treatment-effect pipeline，`causallib` 很合适；如果想做 PyMC Bayesian causal modeling，`CausalPy` 很合适；如果目标是用一个 Python 包替代日常 Stata/R 实证工作流，StatsPAI 更贴近这个目标。

---

## 新手案例：代码和结果一起看

下面的结果来自本仓库随带示例，使用 StatsPAI 1.20.0 运行后四舍五入展示。

### 1. OLS：替代第一条 `regress` / `lm`

问题：在 Card (1995) 教学数据中，多上一年学和 log wage 的关系有多大？

```python
import statspai as sp

card = sp.datasets.card_1995()
ols = sp.regress("lwage ~ educ + exper", data=card, robust="hc1")
print(ols.summary())
```

结果：

```text
Model: OLS
Dependent Variable: lwage

           Coefficient  Std. Error  t-statistic  P>|t|
Intercept       4.9060      0.0599      81.8392 0.0000
educ            0.1088      0.0042      25.8730 0.0000
exper           0.0164      0.0014      11.3496 0.0000

R-squared: 0.2102
```

像读 Stata/R 回归表一样读：在这个 replica 里，多上一年学与约 `0.109` 的 log wage 增加相关；这还没有处理教育的内生性。

### 2. IV / 2SLS：替代 `ivregress 2sls` 或 `AER::ivreg`

问题：用是否接近四年制大学（`nearc4`）作为教育年限的工具变量。

```python
import statspai as sp

card = sp.datasets.card_1995()
iv = sp.ivreg(
    "lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
    data=card,
)
print(iv.summary())
```

结果：

```text
Model: IV-2SLS
Dependent Variable: lwage

           Coefficient  Std. Error  t-statistic  P>|t|
educ            0.1418      0.0188       7.5606 0.0000

Model Diagnostics:
First-stage F (educ): 159.8305
Partial R2 (educ)   : 0.0505
Hausman p-value     : 0.0322
```

StatsPAI 会把系数和常见诊断一起打印出来，不需要再手动拼多个 post-estimation 命令。

### 3. 交错 DiD：替代 `csdid` 或 R `did`

问题：Callaway-Sant'Anna `mpdta` 例子里，最低工资政策对 teen employment 的平均影响是多少？

```python
import statspai as sp

mp = sp.datasets.mpdta()
gt = sp.callaway_santanna(
    data=mp,
    y="lemp",
    t="year",
    i="countyreal",
    g="first_treat",
)
overall = sp.aggte(gt, type="simple", bstrap=False)
print(overall.summary())
```

结果：

```text
Callaway and Sant'Anna (2021) - aggte[simple]

ATT:        -0.032977
Std. Error:  0.005493
95% CI:     [-0.043742, -0.022211]
P-value:     0.0000
Observations: 2,500
```

这个 replica 的总体 ATT 为负，而且估计很精确。

### 4. RD：替代 `rdrobust`

问题：Lee (2008) 参议院选举设计里，在 0 margin 附近是否存在 incumbent advantage？

```python
import statspai as sp

lee = sp.datasets.lee_2008_senate()
rd = sp.rdrobust(data=lee, y="voteshare_next", x="margin", c=0)
print(rd.summary())
```

结果：

```text
Sharp RD Estimation

RD Effect:   0.061599
Std. Error:  0.022662
95% CI:     [0.017183, 0.106015]
P-value:     0.0066

Bandwidth H: 0.042287
N Effective Left: 440
N Effective Right: 443
```

稳健偏差校正后的 RD 估计约为 `0.062` vote-share points。

### 5. 合成控制：替代 Stata/R `synth`

问题：California Proposition 99 对香烟销量有什么影响？

```python
import statspai as sp

prop99 = sp.datasets.california_prop99()
sc = sp.synth(
    data=prop99,
    outcome="cigsale",
    unit="state",
    time="year",
    treated_unit="California",
    treatment_time=1989,
)
print(sc.summary())
```

结果：

```text
Synthetic Control Method

ATT:        -13.085166
Std. Error:  4.164718
95% CI:     [-21.247862, -4.922469]
P-value:     0.0789

Active donor weights:
Montana  0.8420
Nevada   0.1580
```

在这个 replica 中，干预后 California 的人均香烟销量大约少了 13 包。

---

## 交互式图表编辑

如果你怀念 Stata 的 Graph Editor，可以对 StatsPAI 返回的任意 matplotlib 图使用
`sp.interactive(fig)`。它会在 Jupyter 里打开一个带实时预览的编辑面板，新手不用先记住
matplotlib 的所有参数，也能把图调到适合论文或汇报的样子。

它适合做这些事：

- 修改标题、坐标轴标签、字体、颜色、点线样式、网格、图例、坐标范围、图尺寸和导出 DPI；
- 一键切换学术论文、ggplot 风格、FiveThirtyEight 风格、深色演示等主题；
- 保护数据图层，只编辑外观元素；
- 自动导出可复现 Python 代码，避免最终图只停留在手工截图里。

```python
import statspai as sp

mp = sp.datasets.mpdta()
gt = sp.callaway_santanna(data=mp, y="lemp", t="year",
                          i="countyreal", g="first_treat")
agg = sp.aggte(gt, type="dynamic", bstrap=False)
fig, ax = sp.ggdid(agg)

editor = sp.interactive(fig)   # 在 Jupyter 里编辑图表
print(editor.generate_code())  # 复制可复现的 matplotlib 编辑代码
```

<p align="center">
  <img src="https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/main/docs/assets/StatsPAI-interactive.png" alt="StatsPAI 交互式图表编辑器截图" width="820">
</p>

上面的截图展示了典型使用方式：一边预览图，一边调参数，最后导出可复现代码。

---

## 日常工作流

```python
import statspai as sp

card = sp.datasets.card_1995()
r1 = sp.regress("lwage ~ educ + exper", data=card, robust="hc1")
r2 = sp.ivreg("lwage ~ (educ ~ nearc4) + exper", data=card)

print(r1.summary())                         # 人类可读表格
print(r1.tidy().head())                      # broom 风格 dataframe
sp.modelsummary(r1, r2, output="table.docx") # Word 表
sp.outreg2(r1, r2, filename="results.xlsx")  # Stata 风格导出
```

常用文档：

- [Getting started](docs/getting-started.md)
- [Cookbook](docs/cookbook.md)
- [Choosing an IV estimator](docs/guides/choosing_iv_estimator.md)
- [Choosing a DID estimator](docs/guides/choosing_did_estimator.md)
- [Choosing an RD estimator](docs/guides/choosing_rd_estimator.md)
- [Migrating from R to StatsPAI](docs/guides/migration-from-r.md)
- [Exporting regression tables](docs/guides/exporting-regression-tables.md)

---

## 验证状态与 Agent 使用

StatsPAI 的 API 面很大，所以一定要看 validation status。

```python
import statspai as sp

print(sp.describe_function("ivreg")["validation_status"])
print(sp.list_functions(validation_status="certified")[:5])
```

建议按以下层级理解：

- certified：有外部数值证据；
- validated：有内部测试或发表参考值检查；
- api-stable：接口稳定，但精确 Stata/R 对齐可能依赖设计；
- experimental：前沿或实验性工作流。

Agent 可读元数据可通过 `sp.list_functions()`、`sp.describe_function()`、`sp.function_schema()` 获取。

---

## Changelog

版本历史已经独立到 README 之外：

- [CHANGELOG.md](CHANGELOG.md)：完整版本记录。
- [Docs changelog page](https://brycewang-stanford.github.io/StatsPAI/changelog/)：文档站渲染版。

README 首页只保留新手上路所需信息。

---

## 审稿人入口

StatsPAI 正在 JOSS 审稿中。审稿人可从这里开始：

- [JOSS reviewer guide](docs/joss_reviewer_guide.md)
- [JOSS validation dossier](docs/joss_validation_dossier.md)
- [Design rationale and FAQ](docs/joss_reviewer_qa.md)
- [Examples](examples/)
- [Contributing](CONTRIBUTING.md)
- [Support](SUPPORT.md)

---

## 引用

如果在研究中使用 StatsPAI，请同时引用 StatsPAI 和具体估计器背后的方法论文。`sp.citation()` 会返回包引用，许多结果对象也提供 estimator-level citation helpers。

```bibtex
@software{wang2026statspai,
  author  = {Wang, Biaoyue and Rozelle, Scott},
  title   = {StatsPAI: Validation-Tiered Causal Inference and
             Econometrics Workflows for Python},
  year    = {2026},
  version = {1.20.0},
  url     = {https://github.com/brycewang-stanford/StatsPAI}
}
```

---

## 许可证

MIT。见 [LICENSE](LICENSE)。
