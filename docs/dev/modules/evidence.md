# 证据映射（validation_scope）

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`sp.validation_scope`（`validation_scope.py`）是按配置 × 输出的证据映射**：每行对每个维度列出*实际运行过*的取值（禁止通配符），并列出比较过的输出（estimate / se / coverage / diagnostic）；只有输出可证明不依赖某维度时才在 `invariant` 里豁免并写理由。新增或改动核心估计量的选项、默认值或证据行时同步这张映射；行所归功的入口点必须出现在证据产物源码里（`test_each_artifact_calls_the_entry_point_it_is_credited_to`）。2026-09 建映射时抓出 `sp.iv(vce=)` 静默丢参、`sp.fast.feols` 默认 ssc 未被任何 parity 行覆盖、LIML 行其实跑的是 `sp.liml`。
- **输出 `interval`（2026-10-22 起）**：p 值与置信限，由 SE 与临界值的分布 / 自由度决定，`se` 行不替它作证。只有真的比较了 p 值或置信限的行才列它（目前是 `test_es_crit_parity.py` 的三个映射）。按“临界值规则”区分配置的维度叫 `crit_dist`，并在 `invariant` 里对 estimate / se / vcov 豁免；以后别的估计量加临界值选项时沿用这个名字和取值（`normal` / `t_resid` / `t_cluster` / `t_design` / `bootstrap`）。`docs/evidence_inventory.md` 的总表多了一列。
- **约定开关是自己的维度**：像 `parity=` 这种一次改多条约定（基期、权重、方差符号）、而且会改 headline 的开关，在映射里单列维度，旧行一律记 `none`，不要指望别的维度组合恰好覆盖它。CS 的 `variance_convention` 也是维度（2026-10-22 起）：只在重复截面路径上 DR 带协变量时取 `'delta'` / `'reference'`，其余格记 `'na'`（两种约定给出相同的数）。R `did`、`DRDID`、csdid 都用 `'reference'`，所以对照它们的行记 `('na', 'reference')`；默认 `'delta'` 的拟合只有点估计有 T2 证据（ATT 与 `'reference'` 逐位相同到 1e-12），SE 显示 not_covered，这是实情：它的依据是论文的影响函数与 `_fixtures/_drdid_rc_pre_regression_mc.py` 的 Monte Carlo，不是跨软件对数。
