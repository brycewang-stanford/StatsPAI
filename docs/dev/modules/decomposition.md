# 分解

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`decomposition/`**：影响函数 / statistic-value / WLS 在 `_common.py`。RIF / FFL / inequality / Oaxaca 都委托到该文件。
- **`sp.oaxaca` 的标准误口径（2026-11-13 探测）**：delta 法，分组回归用 HC1，协变量均值视为固定，等于 Stata `oaxaca ..., vce(robust) fixed`（`reference=0/1`、Reimers、Cotton、三重分解都对到 1e-12）。Stata 默认把均值当随机量（Jann 2008），Card 黑白工资差的 explained SE 是 0.0116 对我们的 0.0080；这是未实现的功能，不是口径之争里我们占理的一方，`summary()` 现在写明口径。`reference='cotton'/'reimers'` 的 unexplained 是 `(w_B Xbar_A + w_A Xbar_B)'(beta_A - beta_B)` 一个线性组合；`'pooled'` 的 beta* 与两组系数同源，方差用三个回归的影响函数逐行相加（各带自己的 HC1 因子）。旧代码把两半的方差直接相加（注释自称 conservative），高估 14%–34%。Stata `omega` 走 suest，三个回归统一用 N/(N-1)，所以 pooled 的 explained SE 比我们小 sqrt((N-1)/(N-k))，unexplained SE 差 0.3%，不要为此改因子。
- **`sp.oaxaca(means='stochastic')`（2026-11-13）**：Jann (2008) 的随机均值方差，即 Stata `oaxaca ..., vce(robust)` 的默认口径。每个分量在固定均值方差上加 `g_x' V(xbar) g_x`，`V(xbar)` 是组内协方差（除数 n-1）除以组样本量；均值与系数视为不相关，不加 `trace(V(xbar)V(beta))` 项（Stata 也不加）。`reference=0/1`、Reimers、Cotton、三重分解和明细对 Stata 到 1e-9；pooled 只比较新增的那一项（固定均值部分仍有上面说的 suest 因子差）。默认仍是 `'fixed'`，改默认属于 owner 决定。`overall['gap_se']` 是新增键。
- **分解家族的 `summary()` 会 `print` 再返回字符串**（`_results.py` 与各文件里的 `print(text)`），`print(r.summary())` 因此出现两遍；其余结果类只返回。未改，属于全家族行为，改之前先定去留。
