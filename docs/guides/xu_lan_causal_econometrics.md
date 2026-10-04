# 徐小君、蓝嘉俊《因果推断计量经济学》与 StatsPAI

徐小君、蓝嘉俊主编的《因果推断计量经济学》(清华大学出版社，2025) 把经典计量
(回归、异方差、序列相关、时间序列) 和因果推断 (因果图、工具变量、样本选择、
潜在结果、断点回归、双重差分) 放进同一门本科课。全书 12 章。

出版社没有公开这本书的数据和程序，所以这份指南不复现书里的表格。它按章列出
每类方法在 StatsPAI 里怎么调用，全部代码用 StatsPAI 自带的数据或现场生成的
数据，可以直接运行。书里用 Stata 讲的内容，文末说明怎样把 Stata 命令原样贴
进来。

下面每个数字背后的函数都和 Stata 18 对照过，核对记录见
`docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`。

```python
import numpy as np
import pandas as pd
import statspai as sp
```

## 第 1 章 概率统计复习与假设检验

书里的两个例子可以直接用汇总数代入。

例 1。英语成绩原来服从 N(85, 0.5)，辅导后 10 次考试平均 88 分，方差视为已知。
问均值是否提高。方差已知时用 z 检验。

```python
zt = sp.ztest(n=10, mean=88, sd=0.5 ** 0.5, mu=85)
print(zt.statistic, zt.pvalue_greater)    # z = 13.4，单边备择 mu > 85
```

例 2。成绩原来的标准差是 2，辅导后 10 次考试的样本方差是 1.3。问成绩是否更
稳定。统计量 (n-1)S²/σ₀² 服从 χ²(9)。

```python
v = sp.sdtest(n=10, sd=1.3 ** 0.5, sd0=2)
print(v.statistic, v.pvalue_less)         # 2.925，单边 p 值 0.033，拒绝原假设
print(v.ci)                               # 标准差的 95% 置信区间
```

有原始数据时把 DataFrame 和列名传进去。标准差未知时用 `sp.ttest`。

```python
rng = np.random.default_rng(1)
scores = pd.DataFrame({"score": rng.normal(87, 2, 40),
                       "class": np.repeat([0, 1], 20)})
sp.ttest(scores, "score", mu=85)          # 单样本 t 检验
sp.ttest(scores, "score", by="class")     # 两样本
sp.sdtest(scores, "score", by="class")    # 两个方差是否相等 (F 检验)
```

方差检验依赖正态假设。数据厚尾时它的实际拒绝率远高于名义水平。

## 第 2、3 章 线性回归及其检验

```python
card = sp.datasets.card_1995()
ols = sp.regress("lwage ~ educ + exper + expersq + black + south + smsa",
                 data=card)
print(ols.summary())

sp.estat(ols, "ic")                       # AIC / BIC
sp.estat(ols, "vif")                      # 多重共线性
sp.estat(ols, "hettest")                  # Breusch-Pagan
sp.estat(ols, "white")                    # White 检验
robust = sp.regress("lwage ~ educ + exper + expersq + black + south + smsa",
                    data=card, vce="robust")
q = sp.qreg(card, "lwage ~ educ + exper + expersq", quantile=0.25)
```

序列相关的检验和修正用在时间序列回归上。`durbinalt` 在解释变量含滞后因变量
时仍然有效，Durbin-Watson 统计量此时无效。

```python
T = 200
e = np.zeros(T)
for t in range(1, T):
    e[t] = 0.6 * e[t - 1] + rng.normal()
ts = pd.DataFrame({"x": rng.normal(size=T)})
ts["y"] = 1 + 0.5 * ts["x"] + e
fit = sp.regress("y ~ x", data=ts)
sp.estat(fit, "dwatson")
sp.estat(fit, "bgodfrey", lags=2)         # Breusch-Godfrey
sp.estat(fit, "durbinalt")                # Durbin 备择检验
sp.estat(fit, "archlm", lags=2)           # ARCH 效应的 LM 检验
sp.prais("y ~ x", data=ts)                # Prais-Winsten 可行 GLS
sp.regress("y ~ x", data=ts, robust="hac", hac_lags=4)   # Newey-West
```

## 第 4 章 虚拟变量、参数约束、受限因变量

Stata 的因子变量写法可以直接用在 `sp.test` 和 `sp.lincom` 里。

```python
m = sp.regress("lwage ~ educ * C(black) + exper + expersq", data=card)
sp.test(m, "1.black 1.black#c.educ")      # 两组的截距和斜率是否相同 (Chow 型检验)
sp.lincom(m, "educ + 1.black#c.educ")     # 黑人样本的教育回报
sp.test(m, "educ = 0.07")                 # 单个线性约束
```

非线性组合用 `sp.nlcom`，标准误由 delta 方法给出。二次项的转折点是一个例子。

```python
sp.nlcom(ols, "-exper / (2 * expersq)")   # 工资对经验的转折点
```

二值因变量、归并和截断。

```python
card["high"] = (card["lwage"] > card["lwage"].median()).astype(int)
p = sp.probit("high ~ educ + exper + black", data=card)
sp.margins(p)                             # 平均边际效应

lat = 1 + card["educ"] * 0.1 + rng.normal(size=len(card))
cens = pd.DataFrame({"y": lat.clip(lower=2.0), "educ": card["educ"]})
sp.tobit(cens, y="y", x=["educ"], ll=2.0)                 # 归并
sp.truncreg(cens[cens["y"] > 2.0], y="y", x=["educ"], ll=2.0)   # 截断
```

## 第 5 章 因果图

```python
g = sp.dag("Z -> X; Z -> Y; X -> Y")      # Z 是混杂因素
g.adjustment_sets("X", "Y")               # [{'Z'}]：控制 Z 即可识别
g.backdoor_paths("X", "Y")

c = sp.dag("X -> C; Y -> C")              # C 是对撞因子
c.d_separated("X", "Y", set())            # True：不控制时独立
c.d_separated("X", "Y", {"C"})            # False：控制 C 反而打开路径

m_bias = sp.dag("U1 -> X; U1 -> M; U2 -> M; U2 -> Y; X -> Y")
m_bias.bad_controls("X", "Y")             # M 是坏控制变量
```

控制对撞因子造成的选择偏差，就是书里 5.4 至 5.7 节讨论的样本选择问题。

## 第 6 章 工具变量

```python
iv = sp.ivreg("lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)",
              data=card)
print(iv.summary())
sp.estat(iv, "firststage")                # 弱工具变量
sp.estat(iv, "endogenous")                # 内生性检验 (Durbin-Wu-Hausman)

over = sp.ivreg("lwage ~ exper + expersq + black + south + smsa"
                " + (educ ~ nearc4 + nearc2)", data=card)
sp.estat(over, "overid")                  # 过度识别检验
gmm = sp.iv("lwage ~ exper + expersq + black + south + smsa"
            " + (educ ~ nearc4 + nearc2)", data=card,
            method="gmm", robust="hc1", small=False)   # Stata 的 ivregress gmm
```

处理效应异质时，2SLS 估计的是依从者的局部平均处理效应 (LATE)，不是全体的平均
效应。`sp.iv_diag` 给出这方面的诊断。

## 第 7 章 样本选择模型

```python
n = 2000
z = rng.normal(size=n)
educ = rng.normal(12, 2, n)
u = rng.normal(size=n)
eps = 0.7 * u + rng.normal(size=n)                    # 两个方程的误差相关
work = 0.2 * educ - 2.4 + z + u > 0
wage = np.where(work, 1 + 0.8 * educ + 2 * eps, np.nan)
lab = pd.DataFrame({"wage": wage, "educ": educ, "z": z})

sp.heckman(lab, y="wage", x=["educ"], z=["educ", "z"])                 # 两步法
h = sp.heckman(lab, y="wage", x=["educ"], z=["educ", "z"], method="ml")  # 极大似然
print(h.detail)                            # 含 rho、sigma、lambda
```

`z` 里要有不进入工资方程的变量 (排他性约束)。没有它，识别只靠正态分布的函数
形式。不传 `select=` 时，工资缺失即视为未被选中，和 Stata 的
`heckman wage educ, select(educ z)` 一致。Stata 的 `heckman` 默认是极大似然，
`sp.heckman` 默认是两步法。

内生的二值处理变量用 `sp.etregress`。

```python
d = (0.5 * z + u > 0).astype(int)
treat = pd.DataFrame({"y": 1 + 2 * d + 0.5 * educ + eps, "d": d,
                      "educ": educ, "z": z})
sp.etregress(treat, y="y", x=["educ"], treatment="d", z=["z"])
```

## 第 8 章 潜在结果、随机实验与匹配

```python
nsw = sp.datasets.nsw_dw()
X = ["age", "education", "black", "hispanic", "married", "nodegree",
     "re74", "re75"]
sp.ttest(nsw, "re78", by="treat")          # 朴素比较
sp.regress("re78 ~ treat + " + " + ".join(X), data=nsw)   # 回归调整
sp.psm(nsw, y="re78", d="treat", X=X)      # 倾向得分匹配
sp.ipw(nsw, y="re78", treat="treat", covariates=X, estimand="ATT")
sp.aipw(nsw, y="re78", treat="treat", covariates=X)       # 双重稳健
```

匹配估计量的选择见 `choosing_matching_estimator` 指南。

## 第 9 章 断点回归

```python
senate = sp.datasets.lee_2008_senate().dropna()
senate["d"] = (senate["x"] >= 0).astype(int)

# 参数化估计：断点两侧各自的线性趋势，限制在带宽内
local = senate[senate["x"].abs() <= 20]
sp.regress("y ~ d * x", data=local, vce="robust")

# 非参数估计：数据驱动的带宽和偏差校正的置信区间
rd = sp.rdrobust(senate, y="y", x="x", c=0)
print(rd.summary())
sp.rddensity(senate, x="x", c=0)           # 驱动变量是否被操纵
```

全局高阶多项式对远离断点的观测很敏感，现在的做法是局部线性加稳健置信区间。

## 第 10 章 面板数据与双重差分

```python
ids, years = 200, 6
panel = pd.DataFrame({"id": np.repeat(np.arange(ids), years),
                      "year": np.tile(np.arange(2010, 2010 + years), ids)})
alpha = np.repeat(rng.normal(size=ids), years)
panel["x"] = 0.5 * alpha + rng.normal(size=len(panel))
panel["y"] = 1 + 0.8 * panel["x"] + alpha + rng.normal(size=len(panel))

fe = sp.panel(panel, "y ~ x", entity="id", time="year", method="fe")
re = sp.panel(panel, "y ~ x", entity="id", time="year", method="re")
sp.hausman(fe, re, sigmamore=True)         # 个体效应与 x 相关，拒绝随机效应
sp.panel(panel, "y ~ x", entity="id", time="year", method="mle")
```

`sigmamore=True` 让两个估计量用同一个误差方差估计。不加这个选项时，两个协方差
矩阵之差在有限样本里可能不是正定的，统计量会出现负值 (Stata 也一样)，这时检验
没有结论。

双重差分。处理组在 2013 年之后受到政策影响，真实效应是 1。

```python
panel["treated"] = (panel["id"] < 100).astype(int)
panel["post"] = (panel["year"] >= 2013).astype(int)
panel["y2"] = panel["y"] + 1.0 * panel["treated"] * panel["post"]

sp.regress("y2 ~ treated * post", data=panel, cluster="id")     # 回归表达
twfe = sp.feols("y2 ~ treated:post | id + year", data=panel,
                vce={"CRV1": "id"})                              # 双向固定效应
```

共同趋势用事件研究图检查。处理时点不一致时，双向固定效应的系数不再是平均处理
效应，见 `choosing_did_estimator` 指南。

## 第 11 章 时间序列基础

```python
T = 300
y = np.zeros(T)
for t in range(1, T):
    y[t] = 2.0 + 0.6 * y[t - 1] + rng.normal()        # 均值为 5 的 AR(1)
walk = np.cumsum(rng.normal(size=T))

sp.corrgram(y, lags=8)                     # 自相关、偏自相关、Q 统计量
ar = sp.arima(y, order=(1, 0, 0))          # 估计常数项 (序列均值) 和 AR 系数
print(ar.params)
sp.arima(walk, order=(1, 1, 0), trend="c") # 差分后带漂移项，同 Stata 的 arima

sp.unitroot(walk, test="adf")              # 原假设：有单位根
sp.unitroot(walk, test="pp")               # Phillips-Perron
sp.unitroot(walk, test="kpss")             # 原假设：平稳。方向相反
```

ADF 和 PP 不拒绝、KPSS 拒绝，三者一致地指向单位根。

ARCH 效应的检验和估计。`sp.garch` 的 `p` 是滞后条件方差的阶数，`q` 是滞后残差
平方的阶数，ARCH(1) 写成 `p=0, q=1`。

```python
r = np.zeros(600)
s2 = np.ones(600)
for t in range(1, 600):
    s2[t] = 0.2 + 0.3 * r[t - 1] ** 2 + 0.5 * s2[t - 1]
    r[t] = np.sqrt(s2[t]) * rng.normal()
sp.estat(sp.regress("r ~ 1", data=pd.DataFrame({"r": r})), "archlm")
sp.garch(r, p=1, q=1)
```

协整。两个序列共享一个随机趋势。

```python
pair = pd.DataFrame({"c": walk + rng.normal(size=T), "inc": walk})
sp.engle_granger(pair, ["c", "inc"])       # E-G 两步法
```

## 第 12 章 分布滞后、VAR 与结构 VAR

部分调整模型 y = a + b·x + c·y(-1) 的长期乘数是 b/(1-c)，用 `sp.nlcom`。

```python
x = rng.normal(size=T)
yy = np.zeros(T)
for t in range(1, T):
    yy[t] = 0.5 + 0.4 * x[t] + 0.6 * yy[t - 1] + rng.normal(scale=0.5)
adl = pd.DataFrame({"y": yy, "x": x})
adl["y_lag"] = adl["y"].shift(1)
fit = sp.regress("y ~ x + y_lag", data=adl.dropna())
sp.nlcom(fit, "x / (1 - y_lag)")           # 长期乘数，真值 1.0
sp.estat(fit, "durbinalt")                 # 含滞后因变量时的序列相关检验
```

VAR、格兰杰因果检验和脉冲响应。

```python
u = rng.normal(size=(T, 2))
P = np.array([[1.0, 0.5], [-0.6, 0.8]])    # 供给冲击、需求冲击对 (产出, 价格) 的当期影响
data = np.zeros((T, 2))
for t in range(1, T):
    data[t] = 0.4 * data[t - 1] + P @ u[t]
macro = pd.DataFrame(data, columns=["output", "prices"])

sp.varsoc(macro, ["output", "prices"], maxlag=4)   # 滞后阶数选择
var = sp.var(macro, variables=["output", "prices"], lags=1)
var.granger_table()
var.irf(periods=8)                         # Cholesky 正交化的脉冲响应
var.fevd(8)                                # 预测误差方差分解
```

结构 VAR 的三种识别方法对应书里 12.3 节。矩阵里用 `np.nan` 表示待估元素。

```python
nan = np.nan
# 短期约束：价格当期不影响产出 (递归识别)
short = sp.svar(var, B=[[nan, 0], [nan, nan]])
print(short.table)
short.irf(8)

# 长期约束 (Blanchard-Quah)：第二个冲击对产出没有长期影响
long = sp.svar(var, long_run=[[nan, 0], [nan, nan]])

# 符号约束：供给冲击使产出上升、价格下降；需求冲击使两者都上升
signs = sp.svar(var, sign={"supply": {"output": "+", "prices": "-"},
                           "demand": {"output": "+", "prices": "+"}},
                n_draws=1000, seed=0)
signs.irf(8)                               # 中位数和分位数带
```

符号约束得到的是一个集合，不是一个模型。`irf` 里的上下界描述这个集合在估计出
的简化式下有多宽，它不是置信区间，不含抽样误差。

局部投影法直接对每个期数做一次回归。

```python
macro["shock"] = u[:, 1]
sp.local_projections(macro, outcome="output", shock="shock", horizons=8)
```

## 把 Stata 命令贴进来

书里用 Stata 讲的例子可以原样执行。`sp.stata` 逐行运行，返回最后一条命令的
结果。

```python
out = sp.stata("""
    regress lwage educ exper expersq i.black
    testparm i.black
""", data=card)

sp.stata("ivregress gmm lwage exper expersq (educ = nearc4 nearc2)", data=card)
sp.stata("sdtesti 10 . 1.14 2")
```

2026 年 10 月的几轮教材对照之后，`sdtest`、`ztest`、`truncreg`、`etregress`、
`heckman`、`pperron`、`kpss`、`arima`、`arch`、`testparm`、`nlcom`、
`estat durbinalt`、`estat archlm` 和 `ivregress gmm` 都可以直接运行。Stata 的默认设置和
StatsPAI 函数的默认设置不同时 (例如 `heckman` 默认极大似然、`ivregress gmm`
默认稳健权重矩阵)，翻译会把 Stata 的默认写出来。

还不能翻译的写法列在评审文档的 "Left open" 一节，其中最常见的是变量名缩写
(`educ` 代替 `education`)，请写全名。
