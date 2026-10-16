# 工具变量与 MTE

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`regression/iv.py`**：投影走 QR、2SLS 第二阶段走 `lstsq`，不要改回 `(W'W)^-1` / `(X'PX)^-1`（条件数平方两次，立方工具变量设计上丢 7 位）。与外生回归元和其它工具变量共线的排除工具变量由 `_independent_instruments` 丢弃并告警（`model_info['omitted_instruments']`），不再抛 numpy 的 `Singular matrix`。稳健 2SLS 之后 `sp.estat(result, 'overid')` 的 Hansen J 同时就是 Stata `estat overid` 的稳健 score 统计量（键 `score`）；`iid_errors` 里是 Sargan / Basmann（LIML 为 Anderson-Rubin / Basmann F）。Stata 在 `ivregress, perfect` 下的 score 统计量随工具变量书写顺序变化，那是参考实现的问题，不要去对它。
- **`iv/mte.py`**：ATT / ATU 是系数的精确线性函数（`_aggregate_functionals`）：处理组是 `U < P` 的单位，所以对**全样本**按 `p_i` 加权积分 `∫_0^{p_i} MTE`，不是用处理组内的倾向得分分布（2026-10 前的做法，线性 MTE 下 ATT 1.22 对真值 1.30）。聚合量与曲线的标准误必须用系数的**完整**协方差（多项式各项系数强负相关，只取对角会把 ATE 标准误夸大三倍），解析标准误以拟合的倾向得分和协变量为条件，`bootstrap=` 才包含两者。bootstrap 必须重抽**修剪前**的数据并在每次抽样里重做修剪。
