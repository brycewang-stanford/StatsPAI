# 测试：新代码要求的来由

> 从 CLAUDE.md §5 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；补覆盖率或写断言之前先读完本文件。新的细则直接写在这里。

### 新代码要求

- 每个对外函数：**正确性测试 + 边界测试**各至少一个。
- 新估计器：**有参考实现走对齐，没有走解析/仿真**，容差 `atol` / `rtol` 就地标注并说明理由。
- 核心估计器（`did iv rd synth dml panel`）目标覆盖率 **≥ 95%**；整仓目标 **≥ 85%**。
- **Windows CI 注意**：`Path.read_text()` 必须传 `encoding="utf-8"`（cp1252 默认会挂，见 commit `8755996`）。
- **核心覆盖率棘轮只在 nightly 全量里跑**（`scripts/coverage_campaign.py report --check --min 95`），push 触发的 CI 不跑。2026-09-27 到 10-09 它连续失败而无人察觉（did 93.1 / rd 92.0 / synth 93.6 / dml 93.7）。给核心模块加代码后看一眼次日的 nightly；缺口用 nightly 日志里的逐文件 missing 行号定位，`python scripts/coverage_campaign.py gaps <模块>` 亦可。
- **未覆盖的分支就是 bug 藏身的地方。** 2026-10-10 给上述四个模块的 734 行未覆盖代码补测试，查出约二十个缺陷，其中七个静默返回错数或零标准误（`sp.fect(force='none')` 返回处理格原始均值、`Period` + `NaT` 的 cohort 列分错队列、`sp.drdid(id=, weights=)` 忽略权重、`did_multiplegt` 把不可估的安慰剂报成 0 / p=1 等，见 CHANGELOG）。补覆盖率时每条测试要断言一个真命题（已知真值、恒等式、参考实现的数），不要为了行数写冒烟测试。
- **「与另一个数不同」不是断言。** `test_cs_rcs_parity.py` 里一条测试只断言聚类标准误「不等于」不聚类的，而它的聚类标准误一直是 1e-16。能写恒等式就写恒等式（那条测试现在断言：平衡面板上按行求聚类和等于面板估计量的同一聚类标准误）。
- **判断退化不要和 0 比。** 精确拟合留下的标准误是「舍入意义上的零」：同一个设计，这个估计器给 0.0，那个给 1e-17，macOS 与 Linux 的 BLAS 也不同。`se <= 0` 这类判据在一个平台上拦得住、另一个平台上放行（2026-10-10 nightly 在 ubuntu / 3.13 上因此红了三条，macOS 全绿）。用相对阈值，如 `se <= 1e-10 * max|y|`。
- **改了行为之后，按「谁会调用它」找测试，不要按文件名挑。** `grep -rlE '<参数名>|<函数名>' tests` 得到的清单才是该跑的；2026-10-10 给 `callaway_santanna` 加聚类检查后跑了凭文件名挑的 188 个 DiD 文件，漏掉 `tests/reference_parity/test_cs_rcs_parity.py`，CI 的 Parity guards 红了一轮。推送后看一眼 Parity guards，它按六个 shard 跑全部 `reference_parity`，本地很少全跑。

### 常用命令（原 §5）

```bash
pytest                                        # 全量
pytest tests/test_did.py -q                   # 单文件
pytest -k bayes_iv                            # 关键字筛选
pytest --cov=statspai --cov-report=term-missing
pytest tests/reference_parity/ -q             # 纯 Python 已知真值 / 解析 DGP 回收
pytest tests/external_parity/ -q              # 论文数字对齐
python tests/r_parity/compare.py              # Track A：R / Stata 同字节 parity 汇总表
python tests/r_parity/verify_reproduce.py     # R 侧 golden 重推导（需 R）
python tests/stata_parity/verify_reproduce_stata.py   # Stata 侧 golden 重推导（需 Stata 许可）
```
