# Parity 口径统一与证据补强（2026-09）— 已完成

## 起因

用户提问 "1182 个函数只有 393 验证" 触发的审计发现三个互相纠缠的问题：

1. **两套计数体系并存且不一致。** `registry.validation_status`
   (certified 81 / validated 315 = 396) 与 `_parity_index.json`
   (verified 399) 对同一批函数给出不同的划分。交叉表：

   | registry | index | n |
   |---|---|---|
   | validated | bit-exact | 76 |
   | validated | aligned | 9 |
   | certified | analytical-only | 3 |
   | api_stable | bit-exact | 3 |
   | api_stable | analytical-only | 1 |
   | validated | unverified | 2 |
   | experimental | bit-exact | 1 |

   即 registry **低估** 85 个（有 T2 跨语言证据却只标 validated）、
   **高估** 3 个（标 certified 但索引只能给 T1）。

2. **别名等价性是断言而非证据（核心缺陷）。**
   `registry._TRACK_A_MODULE_ALIASES` 11 条里 7 条的 Track A 模块
   **根本没有调用那个别名**：

   | 模块 | 别名 | 模块实际调用 | 状态 |
   |---|---|---|---|
   | 02_iv | `iv` | `sp.ivreg` | 仅断言 |
   | 03_hdfe | `hdfe_ols` | `sp.fast.feols` | 仅断言 |
   | 15_hdfe_cluster | `hdfe_ols` | `sp.fast.feols` | 仅断言 |
   | 17_etwfe | `wooldridge_did` | `sp.etwfe` | 仅断言 |
   | 30_oaxaca | `oaxaca` | `sp.decompose` | 仅断言 |
   | 31_dfl | `dfl_decompose` | `sp.decompose` | 仅断言 |
   | 36_mediation | `mediate` | `sp.mediation` | 仅断言 |
   | 18/19/86/87 | augsynth/gsynth/fect/interflex | 同名 | 已背书（no-op） |

   更糟的是 **循环引用**：`scripts/build_parity_index.py::_DISPATCHER_ALIASES`
   的注释写 "Backed by the registry's own certified seed and the alias
   docstrings"，而 registry 的 certified seed 正来自上面那张手写表。
   两套系统互相引用，谁都没有 artifact。违反 CLAUDE.md §10。

3. **公开文档的汇总行把 T1 和 T2 加总。** `docs/parity.md` 报
   "verified (subtotal) 399"，其中 167 个才有 R/Stata 外部参考，
   227 个是解析真值回收（无外部参考）。审稿人先看文档会读成
   "399 个都对齐了 Stata/R"，比论文本身的主张更大。

## 工作项

### A. 别名等价性：从断言变成证据
- A1 `tests/reference_parity/test_track_a_alias_equivalence.py`：
  对 7 条仅断言的别名，在**同一份 Track A CSV 字节**上跑别名与
  canonical 入口，断言点估计与 SE 逐位一致（atol 机器精度）。
- A2 新增 `src/statspai/_parity_taxonomy.py`：别名表 + 非估计量叶子表
  的**唯一真源**，registry 与 build_parity_index 都从这里读，
  每条别名带 `proof` 字段指向 A1 的测试。
- A3 无法证明等价的别名一律降级，不留在 certified。

### B. 口径统一
- B1 registry 的 validation_status 改为**从 `_parity_index.json` 派生**：
  bit-exact/aligned -> certified；analytical-only/external-replication
  -> validated；unverified -> api_stable。扫描结果只作 notes 来源。
- B2 非估计量叶子（dgp_*/数据集加载器）永不晋升。
- B3 `tests/test_parity_index.py` 的调和测试升级为**双向零分歧**，
  删除 `_KNOWN_VALIDATION_OVER_MARKS` 白名单。

### C. 诚实的分母与汇总
- C1 `docs/parity.md` 汇总表拆成"有外部参考 (T2)"/"已知真值 (T1)"/
  "论文复现"三档，不再给一个会被误读的 verified 合计。
- C2 `sp.parity_summary()` 增加 `by_evidence_kind` 与
  `denominators`（估计量 / 基础设施 / 结果类）。

### D. Paper-JSS 同步
- D1 重生 `manuscript/generated_claims.tex`
- D2 §5 parity / §9 discussion 的相应文字随新口径更新
- D3 `make -C Paper-JSS audit` 全绿

### E. 闸门
- pytest 全量、registry-drift、schema-drift、parity --check、
  bib-subset、examples-coverage


---

## 完成记录

全部工作项完成，另有三项计划外发现。

### 计划内

| 项 | 结果 |
|---|---|
| A1 别名等价性证据测试 | `tests/reference_parity/test_track_a_alias_equivalence.py`，8 项断言 |
| A2 唯一真源模块 | `src/statspai/_parity_taxonomy.py` |
| A3 无法证明的别名降级 | `wooldridge_did` 撤销（见下） |
| B1 registry 从索引派生 tier | 完成；wheel 与源码树结果 0 差异 |
| B2 非估计量永不晋升 | 完成；基础设施已验证数 6 → 0 |
| B3 双向零分歧调和测试 | 完成；白名单删除 |
| C1 `docs/parity.md` 拆分证据档 | 完成，另加"诚实分母"与自动生成的家族覆盖表 |
| C2 `parity_summary()` 扩展 | `by_evidence_kind` + `denominators` |
| D 论文同步 | `generated_claims.tex` 重生；§5 加两段 + 一个新小节；`make audit` 全绿 |

### 计划外发现（三项）

1. **`sp.wooldridge_did` 的 certified 建立在假别名上。** 同字节同参数下与
   `sp.etwfe` 差 6.1%（点估计）/ 12.1%（SE），两种控制组设定都对不上——
   是两个不同的估计量。撤销别名与 certified 等级，写入 CHANGELOG ⚠️ 与
   MIGRATION，并把反证钉成断言。

2. **`sp.bibtex` 被公开为 `external-replication`。** 一个引用解析器被扫描
   噪音赋予了"复现已发表数字"的等级；`sp.describe_function` 同理被标为
   `analytical-only`。已连同 DGP/数据集加载器一起从**索引和 registry 两侧**
   排除。

3. **两行 certified 的参考实现是 Python 而非 R/Stata。** `sp.metalearner`
   （对 `econml`）与 `sp.dml_sensitivity`（对 `DoubleML`）。按 §5.1 应跟方法
   作者维护的实现，证据成立；但论文原句"every certified symbol links to a
   named R or Stata parity module"对这两行为假。已改写论文句子、在
   `_parity_taxonomy.PYTHON_REFERENCE_ROWS` 枚举、并用测试钉死该集合。

### 口径变化

| | 之前 | 之后 |
|---|---:|---:|
| certified | 81 | 168 |
| validated | 315 | 226 |
| api_stable | 783 | 785 |
| 跨语言（T2） | 未单独报告 | 169 |
| 已知真值（T1）+ 论文复现 | 未单独报告 | 226 |
| 估计量分母 | 未报告 | 773（跨语言 21.9%）|
| registry ↔ 索引分歧 | 7（2 高估 + 5 低估，白名单遮蔽 2） | 0（1 项枚举例外）|

### 未做（有意）

按既定判断，`network` / `spatial` / `mendelian` 等家族的 parity 回填**未**动。
它们不是论文阻塞项，且留作审稿轮次的弹药比投稿前补光更有说服力。
`docs/parity.md` 的家族覆盖表现在自动生成，可随时看到这些缺口。


## 过程中的一次返工（记录以免重犯）

跑 `pre-commit run --all-files` 时，black / isort 是**修复型** hook，在
`--all-files` 下对全仓做了重排，改了 619 个与本次工作无关的文件。已按
"本次意图改动清单"逐文件 `git checkout --` 还原，还原后：

- 主仓剩 21 个文件（18 改 + 3 新增），Paper-JSS 剩 10 个源文件 + 28 个审计产物
- `git diff src/statspai/registry.py` 的导入行改动数 = 1（只有我加的那条），
  确认保留文件未被全仓 isort 污染
- 八道 pre-push 闸门、`make -C Paper-JSS audit`、42 项 parity 测试重跑全绿

**教训**：验证闸门要用 `--hook-stage pre-push --all-files`（那批是只读的
`--check` 型），或对修复型 hook 用 `--files <本次改动>`；不要对 black / isort
跑 `--all-files`。
