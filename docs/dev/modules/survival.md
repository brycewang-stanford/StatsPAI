# 生存分析

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`survival/models.py`**：`sp.cox` 的稳健 / 聚类方差必须用**与 `ties=` 相同规则**的得分残差（`_cox_score_individual(breslow=)`）。1.39 前 Efron 拟合用的是 Breslow 残差，无并列时两者相同所以单元测试没抓到，并列多的数据上稳健 SE 偏 2%（⚠️ 已修）。约定：`robust='hc0'` 是裸三明治（= R `coxph(robust=TRUE)`；Stata `stcox, vce(robust)` 再乘 `N/(N-1)`），`cluster=` 带 `G/(G-1)`（= Stata；R 不带）。`sp.kaplan_meier(conf_type=)` 的默认值正在走弃用流程：1.39 不传时仍是 `'plain'` 并发 `DeprecationWarning`，**1.40 改为 `'log-log'`**（Stata `sts` 的默认；R `survfit` 是 `'log'`）；改默认时同步 MIGRATION 与文档里的调用（现已全部显式写出 `conf_type`）。
