# 回归断点（RD）

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`rd/`**：kernel / 局部多项式 / sandwich 走 `rd/_core.py`，不要重新实现。
- **`rd/optimized.py`（`sp.rd_optimized`，Imbens-Wager 2019）**：权重必须**精确**满足四个矩条件（两侧各自和为 ±1、与 running variable 正交，`_project`），否则最坏偏差无界；报告的 `max_bias` 是对*实际使用的权重*按 `M·∫|G|` 精确积分得到的（`worst_case_bias`），不是优化问题里离散化后的那个数——对任何权重都成立，对局部线性权重退化成 RDHonest 的闭式。求解用「单位跳跃、曲率 ≤ kappa 的最不利函数」的有界最小二乘（scipy `lsq_linear(method='bvls')`），再对 kappa 做一维搜索；**不要**换成乘子形式上的 L-BFGS-B 或自写的投影牛顿：Hessian 是二次积分算子，条件数太差，两者都试过、都不收敛。与 `optrdd` 是同一规划的两种离散化，只能对到约 0.01 个标准误（估计）和 0.3%（偏差），按 S 记，不要写成 parity；optrdd 的权重对 running variable 的正交只到 5e-5。连续 running variable 下相对 `sp.rd_honest` 的增益只有千分之几（三角核本来就接近最优），不要在文档里夸大。`fuzzy=` 用结果方程选出的同一组权重，区间是 Anderson-Rubin 式的集合 `{t: |γ'(Y − tD)| ≤ cv·se(t)}`，偏差界为 `(M + |t|·M_fuzzy)·∫|G|`；弱第一阶段时区间无界是正确行为，不要改成 delta 法区间。完全依从时必须精确退化为 sharp 结果（测试守着）。二维 running variable 没做：需要一般线性不等式约束的 QP，scipy 没有合适的求解器。
