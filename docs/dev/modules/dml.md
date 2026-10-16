# 双重机器学习

> 从 CLAUDE.md §11 原样迁来（2026-10-17）。改动本领域的代码前先读完本文件；新的约定直接写在这里，CLAUDE.md §11 只留索引。

- **`dml/`**：`sp.dml(model=...)` 是截面 DML 的 dispatcher（plr / irm / pliv / iivm）；面板另开两个函数，不进 dispatcher（签名要 unit/time）。`sp.dml_panel` 是带单位 FE 的静态 PLR；**`sp.dynamic_dml`（1.31.0）是时变处理的序列效应**（Lewis & Syrgkanis 2021）——处理会改变后续状态时，静态估计量不是低效而是错的。三角矩条件按**堆叠 GMM** 一次解出，点估计与倒推一致但附带跨期联合协方差（总效应 se 0.032 对独立相加 0.062）。给定相同的折与一阶段学习器，与 `econml.panel.dml.DynamicDML`（方法作者写的参考实现）在点估计**和** SE 上都对到 1e-15，是 T2。`lags=1` 是默认且不可省：状态里没有处理史时估计量不是变噪声而是**自信地错**（三期偏 −15% / −15% / +37%，无一覆盖真值）。
- **`dml/did.py`（`sp.dml_did`，1.39 起）**：两期 DML-DiD，面板（ΔY 或长表 `time=` + `id=`）与重复截面两种布局，不进 `sp.dml` dispatcher（签名不同）。canonical 参考是 Python `DoubleML` 的 `DoubleMLDID` / `DoubleMLDIDCS`：同折、同学习器下点估计与 SE 对到 1e-15（`test_dml_did_doubleml_parity.py`，夹具只收确定性学习器，森林版依赖 sklearn 版本）。处理组份额用**全样本**均值（DoubleML 的做法）；教材和 Chang (2020) 按训练折估，`in_sample_normalization=False` 时份额在估计量里约掉。倾向得分是截断不是删行，截断数进 `model_info` 并告警。**已知真值测试不要用随机森林当学习器**：二次趋势设计上 n=600 时森林的冗余参数偏差让估计停在 2.69（真值 2），n=20,000 仍有 2.12，这是学习器的收敛速度，不是估计量的 bug。
