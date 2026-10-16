# 合成控制

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`synth/`**：20+ 估计器全部经 `sp.synth(method=...)` 分发。新增方法要同时加到 dispatcher 和 `synth_compare()`。`method='classic'` 的 `ci` 是与安慰剂秩检验 p 值对偶的常数效应反演区间（`_core.placebo_inversion_ci`，Firpo-Possebom 2018；供体少于 `1/alpha - 1` 个时为 `(-inf, inf)`），旧的 `estimate ± z·sd(placebo ATT)` 只留在 `model_info['ci_normal']`——**不要**再让 p 值和区间出自两套程序（2026-10 Gaillac-L'Hour 教材第 10 章查出）。
- **`synth/_gsynth_multi.py`（`sp.gsynth(treat=)`，1.39 起）**：多处理单位、交错处理、带协变量的广义合成控制（Xu 2017）。对照组上的 IFE 是「β 的最小二乘」与「残差矩阵双向去均值后截断到秩 r」的交替最小二乘；固定 r 时与 R `gsynth` 对到 1e-12（r = 0..3、有无协变量、交错）。`covariates=` 的调用（含单处理单位）一律走这条路径：旧的 `_partial_out_covariates` 是不含截距和固定效应的混合 OLS，书中数据上 6.68 对 5.70，已在 1.39 记 ⚠️。因子数用论文的留一处理前期准则；`gsynth` 1.4 转调 `fect`，交叉验证方案不同，**不要声称选 r 的规则与 R 一致**。无协变量的单处理单位路径（Track A 19 号模块）没有动，它的选 r 规则是随机遮盖对照矩阵的单元格，与两者都不同，改它会动冻结产物。gsynth / fect 是 MIT。
