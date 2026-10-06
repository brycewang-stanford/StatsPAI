# 邱嘉平《因果推断实用计量方法》与 StatsPAI

邱嘉平老师的《因果推断实用计量方法》是面向金融与经济实证研究的方法课教材。全书
十二章，从回归与标准误讲起，依次是随机实验、匹配、面板、双重差分、工具变量、
样本自选择和断点回归，每章配 Stata 代码。

这份指南做三件事。

1. 把书里的 Stata 代码原样贴进 Python 运行。
2. 说明怎样核对结果与 Stata 输出逐位一致。
3. 逐章列出，按 2026 年的实证规范，书里的分析还应该补什么，以及对应的 StatsPAI
   调用。

教材的代码与数据不随 StatsPAI 分发。下文的例子用包内自带的数据，可以直接运行。

## 把 Stata 代码贴进来

`sp.stata` 按 do 文件的写法逐行执行。数据步骤在 DataFrame 的私有副本上运行，
`if` / `in` 按 Stata 的缺失值规则处理，返回最后一条命令的结果。

第 6 章的匹配流程是先用 `pscore` 估计倾向得分并检验平衡性，再用 `attnd` 或
`psmatch2` 在存好的得分上做最近邻匹配。

```python
import statspai as sp

lalonde = sp.datasets.nsw_lalonde().drop(columns=["race"])

att = sp.stata("""
    pscore treat age educ black hispanic married nodegree re74 re75, ///
        pscore(myscore) blockid(block) logit comsup
    attnd re78 treat, pscore(myscore) comsup
    psmatch2 treat, pscore(myscore) outcome(re78) neighbor(1) ties
    pstest age educ re74 re75, both
""", data=lalonde)
```

`use` 一行不要贴，数据由 `data=` 传入。下面几点和 Stata 的行为一致，值得先知道。

- **`generate` 存单精度**，除非写 `gen double`。这是数字能对上 Stata 的原因。
- **`replace` 逐行顺序执行**。`replace x = 0.4 * l.x + e in 2/l` 这样引用自身
  前一行的写法得到递推序列，第 4 章模拟 AR(1) 误差靠的就是它。
- **`pscore` 会写回变量**。`pscore()` 和 `blockid()` 里的名字，以及选了
  `comsup` 时的 `comsup`，之后的命令可以直接用。
- **`psmatch2` 之后有 `r(att)` 和 `r(seatt)`**，所以
  `bootstrap r(att), reps(100): psmatch2 ...` 可以运行。自助抽样用的是 numpy
  的随机数，标准误与 Stata 只在模拟误差范围内一致。

同样的分析直接用 Python 函数写是这样。

```python
covariates = ["age", "educ", "black", "hispanic", "married", "nodegree",
              "re74", "re75"]

ps = sp.pscore(lalonde, "treat", covariates, common_support=True)
print(ps.n_blocks, ps.balanced)        # 7 False
print(ps.unbalanced)                   # re75 在第 2 块不平衡

data = ps.assign(lalonde, pscore="myscore")
m = sp.psmatch2(data, treat="treat", pscore="myscore", outcome="re78",
                ties=True, common_support="treated")
print(m.att, m.se)                     # 1981.08 1007.29，与 attnd, comsup 相同
print(m.pstest(["age", "educ", "re74", "re75"]).summary())
```

## 核对数字

教材的代码不带输出。核对的办法是在 Stata 18 里运行一遍并记录日志，再让 StatsPAI
重放日志里的每一条命令，把 Stata 打印的每个数字与 StatsPAI 的结果逐个比较，精度
取 Stata 打印的位数。

```bash
python tests/external_parity/qiu_jiaping_prepare.py <解压后的文件夹>
# 在 Stata 18 中进入 <文件夹>/run，执行 do _master.do
STATSPAI_QIU_DIR=<文件夹>/run pytest tests/external_parity/test_qiu_jiaping_logs.py
```

十章共 2,007 个数字一致，没有命令被拒绝。有三处不同，都有说明。

| 位置 | Stata | StatsPAI | 原因 |
| --- | --- | --- | --- |
| 第 12 章全局四次多项式回归的模型 F（15 个聚类） | 42621.11 | 42620.89 | 斜率协方差矩阵的条件数是 5e7。用 60 位精度从同样的数据算出的值是 42620.8909。系数和标准误每一位都一致 |
| 第 4 章只含常数项的回归 | F(0, 29) = 0.00 | 缺失 | 没有约束的检验没有统计量 |
| `attnd` 遇到与上下两个对照等距的处理个体 | 随机取一个 | 两个都保留 | 与 `psmatch2, ties` 相同。教材数据里没有这种情况 |

这次核对改掉了 StatsPAI 的几个错误，其中两个值得使用匹配方法的读者知道。

- 倾向得分模型里有冗余协变量时（书中第 6 章的设定把同一个虚拟变量写了两个名字），
  旧版本的得分在第三位小数上就错了，ATT 是 1562.29，而 Stata 的三个命令都给
  1627.36。现在冗余变量会像 Stata 那样被剔除。
- `ties='all'` 判断"距离完全相同"的比较有舍入问题，约六百个数里有一个会漏掉并列
  的对照。

如果你用旧版本做过倾向得分匹配，而模型里有成套的类别虚拟变量或者协变量取值离散，
请重新运行。详见 `MIGRATION.md`。

## 逐章：书里做了什么，今天还该补什么

### 第 3、4 章 回归与标准误

书里用 `loneway` 估计组内相关系数，说明为什么聚类数据的普通标准误偏小。

```python
icc = sp.loneway(lalonde, "re78", by="educ")
rho, g = icc.estimates["icc"], icc.estimates["avg_group_size"]
print(1 + (g - 1) * rho)               # 设计效应
```

今天该补的是聚类很少时的推断。聚类数低于 40 左右时，聚类稳健标准误的 t 检验
过度拒绝。

```python
fit = sp.regress("re78 ~ treat + age + married", data=lalonde, cluster="educ")
boot = sp.wild_cluster_boot(fit, lalonde, cluster="educ", variable="treat",
                            seed=1)
```

### 第 5 章 随机实验

STAR 实验的回归加学校固定效应并按学校聚类，书里的做法今天仍然成立。可以补
`sp.lm_lin`（Lin 2013 的协变量调整）和 `sp.ri_test`（随机化推断）。

### 第 6、7 章 匹配

书里的 `pscore` / `attnd`（Becker 和 Ichino 2002）和 `psmatch2` 的标准误都把倾向
得分当作已知。今天的做法有四点不同。

1. **标准误要计入得分的估计误差**（Abadie 和 Imbens 2016），这也是
   `teffects psmatch` 的默认。

    ```python
    te = sp.match(lalonde, y="re78", treat="treat", covariates=covariates,
                  distance="propensity", estimand="ATT", ties="all",
                  se_method="abadie_imbens_2016")
    ```

2. **先看重叠，再谈平衡**。`sp.overlap_plot(lalonde, "treat", covariates)`
   画两组的得分分布，`sp.trimming(lalonde, "treat", covariates)` 按 Crump
   规则截尾。
3. **平衡性看标准化差异**，不只看分块 t 检验。`m.pstest()` 给出与 Stata
   `pstest` 相同的表，`sp.ps_balance` 给出加权后的标准化均值差。
4. **用双重稳健估计量做对照**。得分模型或结果模型有一个设对即可。

    ```python
    dr = sp.aipw(lalonde, y="re78", treat="treat", covariates=covariates,
                 estimand="ATT")
    eb = sp.ebalance(lalonde, y="re78", treat="treat", covariates=covariates)
    ```

Becker 和 Ichino 的另外几个命令也可以直接贴。`atts` 是按 `pscore` 的分块做分层
估计，对应 `sp.match(method="stratify", strata="block")`。`attk` 是核匹配，对应
`sp.psmatch2(method="kernel", kernel="normal", bwidth=0.06)`。`attr` 没有翻译：
它按"半径内有几个处理个体"给对照加权，周围对照多的处理个体权重更大，这不是半径
匹配估计量的权重，`sp.stata` 会说明原因并拒绝。要做半径匹配请用
`sp.psmatch2(method="radius", caliper=r)`。

匹配只处理可观测的混杂。`sp.sensemakr` 回答"遗漏变量要多强才能推翻结论"。

### 第 8、9 章 面板与双重差分

书里的双重差分是两组两期的设计，加上动态效应和事前趋势的交互项。处理时点不一致
时，双向固定效应回归的系数是各组各期效应的加权平均，权重可以为负。今天的做法是

- `sp.bacon_decomposition` 分解双向固定效应估计量，看它用了哪些比较；
- `sp.callaway_santanna`、`sp.sun_abraham`、`sp.did_imputation` 给出对异质性
  稳健的估计；
- `sp.honest_did` 给出平行趋势被违反到一定程度时效应的置信区间。事前系数不
  显著并不说明平行趋势成立。

选择哪一个见 [选择 DID 估计量](choosing_did_estimator.md)。

### 第 10 章 工具变量

书里报告第一阶段 F、Durbin-Wu-Hausman 检验和过度识别检验。今天该补的是对弱工具
变量稳健的推断。

```python
card = sp.datasets.card_1995()
f = sp.effective_f_test(card, endog="educ", instruments=["nearc4"],
                        exog=["exper", "black"])
ar = sp.anderson_rubin_test(card, y="lwage", endog="educ",
                            instruments=["nearc4"], exog=["exper", "black"])
```

`sp.estat(result, "firststage")` 现在连同 Stock-Yogo 临界值一起返回（一个内生
变量时）。书里第 10 章的例子只有一个工具变量，第一阶段 F 是 13.69。它过了"大于
10"的经验规则，却没到 16.38，也就是让名义 5% 的 Wald 检验实际水平不超过 10% 所需
的值。经验规则和临界值不是一回事。

这些临界值假定同方差。有异方差或聚类时应看有效 F（Montiel Olea 和 Pflueger
2013），结论依赖于 Anderson-Rubin 置信集。

### 第 11 章 样本自选择

`heckman, twostep` 和 `etregress` 的系数与标准误与 Stata 一致。`sp.heckman`
两步法的结果里现在带选择方程。

```python
data = lalonde.assign(y=lalonde.re78.where(lalonde.re78 > 0))
h = sp.heckman(data, y="y", x=["age", "educ"], z=["age", "educ", "married"])
print(h.model_info["selection_equation"])
```

这个例子里两步法估计的 rho 是 -1.34，落在 [-1, 1] 之外。StatsPAI 和 Stata 一样
把它截到 -1 再算标准误，并给出警告。这是模型设定有问题的信号。选择模型的
识别依赖排除性约束，即选择方程里要有不进入结果方程的变量。没有它时识别只来自
函数形式，结果不可靠。

### 第 12 章 断点回归

书里的流程（散点图、`rdplot`、密度检验、协变量连续性、全局多项式、`rdrobust`）
今天有两点要改。

1. **不要用高次全局多项式**（Gelman 和 Imbens 2019）。主结果用局部线性回归、
   数据驱动的带宽和偏差校正的稳健置信区间，这是 `sp.rdrobust` 的默认。
2. **密度检验用 `sp.rddensity`**。McCrary 的 `DCdensity` 可以用
   `sp.mccrary_test` 复现，需要自己选带宽和箱宽。

```python
senate = sp.datasets.lee_2008_senate()
rd = sp.rdrobust(senate, y="y", x="x", c=0)
dens = sp.rddensity(senate, x="x", c=0)
```

配置变量取值离散时见 [选择 RD 估计量](choosing_rd_estimator.md)。
