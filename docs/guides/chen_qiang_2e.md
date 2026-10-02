# 陈强《计量经济学及Stata应用》(第2版) 与 StatsPAI

陈强老师的《计量经济学及Stata应用》是国内本科计量课用得最多的教材之一。第 2
版在传统内容(OLS、异方差、自相关、工具变量、面板、时间序列)之后，新增了匹配、
断点回归、双重差分、合成控制与回归控制五章，并随书提供程序与数据。

这份指南做三件事。

1. 把书里的 do 文件原样贴进 Python 运行。
2. 说明怎样核对结果与 Stata 输出逐位一致。
3. 逐章列出，按 2026 年的实证规范，书里的分析还应该补什么，以及对应的 StatsPAI
   调用。

教材的程序与数据不随 StatsPAI 分发。下文把解压后的文件夹记作 `files/`。

## 把 do 文件贴进来

`sp.stata` 按 do 文件的写法逐行执行。数据步骤在 DataFrame 的私有副本上运行，
`if` / `in` 按 Stata 的缺失值规则处理，返回最后一条命令的结果。

```python
import pandas as pd
import statspai as sp

grilic = pd.read_stata("files/grilic.dta", convert_categoricals=False)

iv = sp.stata("""
    ivregress 2sls lnw s expr tenure rns smsa (iq = med kww), r
    estat overid
""", data=grilic)
```

第 10 章的这两行给出 2SLS 估计和过度识别检验。`use` 一行不要贴，数据由
`data=` 传入。下面几点和 Stata 的行为一致，值得先知道。

- **`generate` 存单精度**，除非写 `gen double`。这是数字能对上 Stata 的原因。
- **`tsset` / `xtset` 会排序**。之后的 `L.` `D.` 算子、`newey`、`prais`、
  `estat bgodfrey` 都按时间变量取滞后，遇到缺期得到缺失值。
- **频数权重按行展开**。`[fweight=freq]` 等价于把每行重复 `freq` 次。权重可以
  写表达式，例如 `[aw=1/e2f]`。
- **`estimates store` 的模型留在会话里**，后面的 `hausman`、`estimates table`、
  `esttab` 可以直接用名字。

一个稍长的例子，第 12 章的面板部分。

```python
lin = pd.read_stata("files/lin_1992.dta", convert_categoricals=False)

session_output = sp.stata("""
    xtset province year
    xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca, fe
    estimates store FE
    xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca, re
    estimates store RE
    hausman FE RE, constant sigmamore
""", data=lin)

session_output["statistic"], session_output["pvalue"]
```

需要逐条查看中间结果时，用会话对象。

```python
from statspai.agent._translation._stata_run import StataSession

s = StataSession(lin)
s.run("xtset province year")
s.run("xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca, re r theta")
s.output.params            # 系数
s.stored["e"]              # sigma_u, sigma_e, rho, theta, 三个 R 方
s.run("xtoverid")          # 稳健的 Hausman 检验
s.output["statistic"]      # 221.225
```

蒙特卡罗演示(第 6、14 章)也能运行。随机数来自 numpy，不是 Stata 的生成器，
所以设计相同、样本不同。

```python
s = StataSession()
for line in """
    program onesample, rclass
        drop _all
        set obs 30
        gen x = runiform()
        sum x
        return scalar mean_sample = r(mean)
    end
    simulate xbar = r(mean_sample), seed(101) reps(10000) nodots: onesample
""".strip().splitlines():
    s.run(line.strip())

s.data["xbar"].describe()   # 均值约 0.5，标准差约 0.0527
```

## 各章命令的对应

| 章 | Stata 命令 | StatsPAI |
| --- | --- | --- |
| 2 至 6 | `gen` `replace` `sort` `gsort` `rename` `tabulate` `summarize` `pwcorr` `regress` `test` `predict` `simulate` | `sp.stata` 的数据步骤，`sp.regress`，`sp.test` |
| 7 | `estat hettest` `estat imtest, white`，加权最小二乘 | `sp.estat(fit, 'hettest' / 'white' / 'imtest')`，`weights=` |
| 8 | `estat bgodfrey` `estat dwatson` `corrgram` `wntestq` `newey` `prais` | `sp.estat`，`sp.corrgram`，`sp.regress(robust='hac')`，`sp.prais` |
| 9 | `estat ic` `estat ovtest` `estat vif` `predict, leverage`，邹检验，`ipolate` | `sp.estat(fit, 'ic' / 'reset' / 'vif')`，`sp.test` |
| 10 | `ivregress 2sls / liml` `estat overid / firststage / endogenous` `hausman` | `sp.iv`，`sp.estat`，`sp.hausman` |
| 11 | `logit` `probit` `margins` `estat classification` | `sp.logit`，`sp.probit`，`sp.margins`，`sp.estat(fit, 'classification')` |
| 12 | `xtreg, fe / re / be / mle` `xttest0` `hausman` `xtoverid` `xtserial` `xtsum` | `sp.feols`，`sp.panel`，`sp.hausman`，`sp.xtoverid`，`sp.xtserial`，`sp.xtsum` |
| 13 | `var` `varsoc` `varwle` `varlmar` `varstable` `vargranger` `fcast` | `sp.var`，`sp.varsoc`，`sp.estat(fit, 'varlmar')` 等，`fit.forecast()` |
| 14 | `dfuller` `vecrank` `vec` `veclmar` `vecstable` | `sp.unitroot`，`sp.johansen`，`sp.vec` |
| 15 | `teffects psmatch` `tebalance summarize` | `sp.match` |
| 16 | `rdrobust` `rddensity` | `sp.rdrobust`，`sp.rddensity` |
| 17 | `reghdfe`，事件研究，`test` | `sp.hdfe_ols`，`sp.test` |
| 18 | `synth` | `sp.synth(method='classic')` |
| 19 | `rcm` | `sp.synth(method='rcm')` |

`sp.estat` 的默认设置跟随 R 的 `lmtest`，和 Stata 的默认不完全相同。翻译会把
Stata 的默认写出来，所以贴 do 文件得到的是 Stata 的数字。

| 检验 | Stata 默认 | `sp.estat` 默认 |
| --- | --- | --- |
| `hettest` | 对拟合值，正态形式 | 对全部解释变量，Koenker 的 N R² |
| `ovtest` / `reset` | 拟合值的 2 至 4 次方 | 2 至 3 次方 |
| `bgodfrey` | 缺失的滞后残差记为 0 | 相同 |

## 核对数字

教材的 do 文件没有附输出。核对需要先在 Stata 里带日志运行一次，再让
`sp.stata` 重放日志里的每条命令，把 Stata 打印的每个数字和 StatsPAI 的结果
比较，精确到 Stata 打印的最后一位。

```bash
python tests/external_parity/chen_qiang_2e_prepare.py files     # 生成 files/run
# 在 Stata 18 中，进入 files/run 后执行：do _master.do
python scripts/stata_log_replay.py files/run --data files/run
STATSPAI_CHENQIANG_DIR=files/run pytest tests/external_parity/test_chen_qiang_2e_logs.py
```

18 章共比较约 2,900 个数字。除下面四处外全部一致。

1. **`estat ovtest, rhs`(第 9 章，模型含 `expr` 与 `expr2 = expr^2`)。** Stata
   报告 `F(11, 741) = 1.73`。它在辅助回归里把原变量 `expr2` 当作共线性剔除，
   再检验替代它的幂项，所以检验的原假设是不含 `expr2` 的模型。对实际估计的模型做
   RESET 检验是 10 个约束，`F(10, 741) = 1.27`，这是 StatsPAI 的结果。
2. **`pscore[match1]`(第 15 章)。** Stata 列出最近邻的顺序不是距离顺序，
   StatsPAI 把最近的排在第一。匹配集合、ATET 及其标准误一致。
3. **`synth, nested`(第 18 章)。** 预测变量权重 V 的搜索是非凸问题。StatsPAI 找到
   的解在干预前的 MSPE 更低(3.086 对 Stata 的 3.227)，因此合成权重略有不同。
4. **`synth2` 没有翻译。** 它是 `synth` 加安慰剂检验与留一法。可以分别调用
   `sp.synth(..., placebo=True)`、`sp.synth_loo`、`sp.synth_time_placebo`。

另外，`bysort treat: sum` 会排序，而 Stata 的排序不稳定，所以之后按行号读取的
内容(如 `list in 1/2`)在 Stata 里每次运行也可能不同。

## 今天还应该补什么

教材第 2 版出版于 2023 年。下面是逐章对照，只是判断，不是文献综述。每个
方法的出处见相应函数的文档。

| 章 | 教材的做法 | 现在的常见做法 | StatsPAI |
| --- | --- | --- | --- |
| 5、6 | 先普通标准误，再稳健标准误 | 默认稳健。小样本用 HC2 或 HC3 | `sp.regress(robust='hc2')` |
| 7 | 先检验异方差，再做 WLS / FGLS | 不做预检验，始终报告稳健标准误。若为效率做 GLS，仍配稳健标准误 | `weights=` 加 `robust=` |
| 8 | BG、Q、DW 检验，经验法则滞后的 Newey-West，Prais-Winsten | 更大带宽的 HAC 配 fixed-b 临界值。FGLS 只在严格外生时一致 | `sp.regress(robust='ewc')`，`sp.regress(robust='hac', hac_lags=)`，`sp.prais` |
| 9 | 信息准则、RESET、VIF、杠杆值、邹检验、线性插值 | 同样的诊断，加上遗漏变量敏感性分析与设定曲线。缺失值用多重插补 | `sp.sensemakr`，`sp.oster_bounds`，`sp.spec_curve` |
| 10 | 2SLS，第一阶段 F 大于 10，过度识别检验，LIML，Hausman | 有效 F、Anderson-Rubin 置信集、tF 临界值。稳健标准误下 F 大于 10 的规则不能控制检验水平 | `sp.iv_diag`，`sp.effective_f_test`，`sp.anderson_rubin_ci`，`sp.tF_adjustment` |
| 11 | Logit、Probit，用 `margins` 报告平均边际效应 | 没有变化 | `sp.margins` |
| 12 | FE、RE、Hausman 检验及其稳健版本，双向固定效应，序列相关检验 | 固定效应加聚类到个体的标准误，不必先检验序列相关。聚类数少时用 wild cluster bootstrap。政策分批实施时双向固定效应不再是默认 | `sp.panel`，`sp.xtoverid`，`sp.wild_cluster_bootstrap`，`sp.callaway_santanna` |
| 13 | AR、ADL 预测，VAR，正交化脉冲响应 | VAR 之外同时报告局部投影 | `sp.local_projections` |
| 14 | ADF、Johansen、VECM | 没有变化。DF-GLS 的功效高于 ADF | `sp.unitroot(test='dfgls')` |
| 15 | 倾向得分匹配与平衡性检验 | 双重稳健估计、平衡权重、重叠权重，并对不可观测混杂做敏感性分析 | `sp.aipw`，`sp.ebalance`，`sp.overlap_weights`，`sp.dml`，`sp.rosenbaum_bounds` |
| 16 | `rdrobust` 多种带宽、协变量、安慰剂结果变量、密度检验 | 没有变化。补充甜甜圈、伪断点与带宽敏感性 | `sp.rdplacebo`，`sp.rdbwsensitivity` |
| 17 | `asinh` 结果变量的双向固定效应事件研究，加地区趋势与事前趋势联合检验 | 见下 | 见下 |
| 18 | 嵌套权重的合成控制，安慰剂比值，留一法 | 预测区间或共形推断，并以合成 DID、增广合成控制作对照 | `sp.synth(method='scpi' / 'sdid' / 'augmented')`，`sp.conformal_synth` |
| 19 | 最优子集加 AICc 的回归控制法 | 控制单位多时用向前选择，并做时间安慰剂 | `sp.synth(method='rcm', selection='forward', placebo_time=)` |

第 17 章是成书以来变化最大的部分。

- 结果变量是比率的 `asinh`。数据里有零时，`asinh` 或 `log(1 + y)` 的估计效应
  依赖 `y` 的计量单位。带固定效应的泊松回归没有这个问题(`sp.ppmlhdfe`)。无论
  用哪种，都值得展示换单位后的敏感性。
- 事前系数的联合检验对真正要紧的趋势功效很低。`sp.pretrends_power` 告诉你它能
  检出多大的趋势，`sp.honest_did` 给出在有界偏离下仍然有效的区间。
- 事件研究图上的逐点置信区间低估了整条路径的不确定性。`sp.uniform_bands` 给出
  sup-t 同时置信带。
- 书中案例只有一个处理时点，双向固定效应没有问题。处理分批发生时则不然，应改用
  异质性稳健的估计量(`sp.callaway_santanna`、`sp.sun_abraham`、
  `sp.did_imputation`)，并用 `sp.bacon_decomposition` 看双向固定效应的权重。

## 回归控制法

第 19 章的 `rcm` 是作者自己写的命令。StatsPAI 的实现是
`sp.synth(method='rcm')`。

```python
growth = pd.read_stata("files/growth.dta", convert_categoricals=False, convert_dates=False)

fit = sp.synth(
    growth, "gdp", "region", "time",
    treated_unit=9, treatment_time=176,
    method="rcm", placebo=True, placebo_cutoff=2, placebo_time=168,
)
fit.estimate                          # 0.0403，处理后平均效应
fit.model_info["selection_table"]     # 各模型规模的 AICc / AIC / BIC / MBIC
fit.model_info["coefficients"]        # 处理前的回归
fit.detail                            # 各期的实际值、预测值、效应与安慰剂 p 值
fit.model_info["placebo_units"]       # 各控制单位的处理前后 MSPE
```

最优子集用分支定界精确求解。上面这个例子有 24 个控制单位，并对每个单位做一次
安慰剂检验，Stata 的 `rcm` 需要约 25 分钟，这里约 2 秒，结果逐位相同。
