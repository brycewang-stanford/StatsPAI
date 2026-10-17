# 面板、多层模型、前沿、动态面板 GMM

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`multilevel/` / `frontier/` / GLMM**：v0.9.3–v0.9.4 有含正确性修复的大重构——用户引用旧数值时主动提示。
- **`gmm/_dynpanel/`**（`sp.xtabond` / `sp.xtdpdsys`）：求逆一律走 `_estimate.safe_inv`，不要直接 `np.linalg.inv`——`inv` 只在主元恰为 0 时报错，数值奇异的权重矩阵会原样返回量级 1e15 的噪声（Hansen 表 17.3 的两步权重秩 191/199，修复前标准误是 1e14 且无告警）。奇异时用的广义逆是 `sweep_ginv`（每步取剩余对角最大者为主元、共线矩条件置零），即 Mata 的 `invsym`；两步估计量在权重奇异时**依赖广义逆的取法**，Moore-Penrose 给出的是另一组数，换掉它就对不上 `xtdpd` / `xtabond2`。Windmeijer 校正里的一步协方差要去掉被置零的矩条件（`_fit.py` 的 `V1_step`）。`time_dummies=True` 的取舍规则跟 Stata `xtdpd`：没有变换方程行的期不建虚拟变量；有常数项时再从回归元里去掉最后一个、但保留在工具变量里。证据在 `test_hansen_methods_stata_parity.py`（abdata 上一步 / 两步 / 奇异权重三组，1e-9）。Stata 报的 instrument 数是秩，我们报列数。
- **`panel/` 的随机效应方差分量只有一份实现**：`panel/xt_tools._swamy_arora`，自由度按**秩**计（组内不变的回归元不占组内回归的参数），`panel/_cre.py::variance_components` 调它；每个回归元都随组内变化时仍用 linearmodels 自己的数。linearmodels 的 `RandomEffects` 按列数计，含行业虚拟变量这类回归元时 `sigma_e` / `theta` / 系数都会偏（2026-10 Hansen 第 17 章对照 Stata 查出）。**不要**再在别处重算这套分量。
- **`mackinnon1994_pvalue`（`panel/unit_root.py`）在拟合范围之上返回 1**（含常数 2.74、含趋势 0.70），与 Stata、statsmodels 一致；plm 在范围外继续代三次多项式，带趋势时 +3.25 给出 0.06，不要为了对 plm 去掉截断。
- **`sp.panel` 的三个 R² 按 Stata `xtreg` 定义（2026-10-18 起）**：`fe` / `re` / `be`（不加权）的 within / between / overall 是相关系数平方（`panel/xt_tools._xtreg_rsquared`，与 `xt_statistics` 共用这一份），和 Stata 18 `e(r2_w)` / `e(r2_b)` / `e(r2_o)` 对到 1e-14（`tests/test_panel_xtreg_rsquared.py`）。linearmodels 的 1 − SSR/SST 版本留在 `model_info['r2_linearmodels']`，`model_info['r2_convention']` 写明用了哪一种。加权、`twoway`、`pooled`、`fd`、CRE 仍是 linearmodels 的数：`xtreg` 没有对应量，换了也无从对齐。只有一个回归元时三种方法的 R² 完全相同，这是定义使然（相关系数与斜率大小无关），不是 bug。
