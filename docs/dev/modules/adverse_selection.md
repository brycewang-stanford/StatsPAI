# 逆向选择（保险市场）

> 2026-10-28 随 Econometrica 94(5) 一轮新建（动机是 Azevedo、Beauchamp、Karlsson Linnér 的基因预测论文）。改动 `adverse_selection/` 之前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **模块名 `tests.py` 是有意的**（惰性导出 `adverse_selection.tests`）。它不匹配 pytest 的 `test_*.py`，不会被收集。改名必须同时改 `_register_lazy` 那一行。
- **`positive_correlation_test(method='biprobit')` 复用 `sp.biprobit`**。rho = 0 的 LR 检验与同一完整样本上的两个独立 probit 比较，与 Stata 相同；给 `cluster=` 时改为 athrho 的 Wald 检验，也与 Stata 相同（`chi2_c`）。Parity：`tests/reference_parity/test_positive_correlation_test_stata_parity.py`（T2，Stata 18，三个情形；对数似然与 LR 1e-14，rho / athrho 1.6e-9，即两边优化器停点之差，athrho SE 1e-10）。夹具 JSON 用 `%24.17e` 写，因为 Stata 的 `%21.17g` 会打出 `.2007...`，不是合法 JSON。
- **`'residuals'` 是双变量 probit 在 rho = 0 处的得分**。`sum e1 e2` 等于 rho = 0 处 `d logL / d rho` 在两个独立 probit 拟合值上的取值（测试断言）。rho = 0 时信息矩阵块对角，所以第一步不改变极限。方差用外积，因此加权重和聚类都便宜。权重是抽样权重：得分与复制行相同，方差有意不同。Chiappori-Salanié（2000）原文付费、仓储副本有机器人墙，没能核对他们的 W 是否用同一个方差形式，所以 docstring 按构造描述统计量，不说是原文的转写。
- **`efc_welfare` 在覆盖率空间里求解**。需求与平均成本都是价格的线性函数时，需求减边际成本对覆盖率是线性的。均衡是 `alpha + beta gamma / (1 - delta)` 截到 [0, 1]，有效覆盖率是剩余在 {0, 1, 内点根} 上的 argmax，DWL = S(D_eff) - S(D_eq)。`beta >= 0` 或 `delta = 1`（均衡不唯一）时抛异常。beta 的 delta 方法 SE 等于 HC0 OLS（测试断言）；成本回归的影响函数行按全样本 n 缩放，恰好复现被保样本上的 HC0（测试断言）。协变量线性进入，曲线在全样本协变量均值处求值。
- **`pooled_price_ratio` 的分位数跟 Azevedo et al.**：`quantile` 是严格低于 r 的质量，"到分位数 q 为止"指 `quantile < q`，即 `r <= Q(q)`。`t_min_all` 在任何有限样本里都约等于 0（最高支撑点是一个人的池子），所以全支撑上的 no-trade 检验几乎总是 False，摘要因此停在一个分位数上（默认 t80，论文报告的统计量）。bootstrap 只重抽给定的概率，不含产生这些概率的模型的估计误差（论文对整条流程做 bootstrap）。

未决：

- `efc_welfare` 与 `pooled_price_ratio` 没有 Stata / R 命令可对照，证据是 T1（闭式 DGP）。
- 未做：EFC 的非线性曲线设定；多于两档合同（有序或多项覆盖）的 Chiappori-Salanié 检验；Hendren 用主观概率构造的私人信息分布的非参数界；Azevedo et al. 第 3 步"标准风险档"（保险人信息下风险在均值 0.75 到 1.25 倍之间）的构造并对 probit 一步做 bootstrap，这一步补上才是论文的完整流程。
