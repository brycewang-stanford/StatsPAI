# 证据映射（validation_scope）

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`sp.validation_scope`（`validation_scope.py`）是按配置 × 输出的证据映射**：每行对每个维度列出*实际运行过*的取值（禁止通配符），并列出比较过的输出（estimate / se / coverage / diagnostic）；只有输出可证明不依赖某维度时才在 `invariant` 里豁免并写理由。新增或改动核心估计量的选项、默认值或证据行时同步这张映射；行所归功的入口点必须出现在证据产物源码里（`test_each_artifact_calls_the_entry_point_it_is_credited_to`）。2026-09 建映射时抓出 `sp.iv(vce=)` 静默丢参、`sp.fast.feols` 默认 ssc 未被任何 parity 行覆盖、LIML 行其实跑的是 `sp.liml`。
