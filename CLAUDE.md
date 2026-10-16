# CLAUDE.md — StatsPAI

> Claude Code / AI agent 在本仓库工作的指引。动手前请通读。

---

## 1. 项目定位

**StatsPAI** 的目标是**面向 Agent 设计，适合人类以及 agent 进行调用，并致力于超越 Stata、R、以及老的 Python 生态，成为全世界最好的因果推断与实证分析工具**。

做法有三条：

1. **一次 `import statspai as sp`** 覆盖 DiD / IV / RD / 合成控制 / DML / Meta-Learner / 贝叶斯因果 / 因果发现 / 结构计量 / 面板 / 空间 / 时序。
2. **Agent 原生**——所有函数返回结构化结果、带自描述 schema，人和 Agent 用同一入口。**v1.6 起进入 P1 阶段**：`sp.causal_question`（estimand-first DSL）、`sp.llm_dag_propose / validate / constrained`（LLM-DAG 闭环）、`sp.paper()`（自动论文）、`sp.causal_text`（文本因果 MVP）。
3. **数值对齐 Stata / R**——已有参考实现的方法，先对齐再扩展。

| | |
| --- | --- |
| 版本 | 以 [`pyproject.toml`](pyproject.toml) / `sp.__version__` 为准（不在此处写死，2026-09 审查发现此行停在 1.24.0） |
| Python | 3.10 – 3.13（2026-10-04 起不再支持 3.9；Windows 暂不在测试矩阵里，见 `docs/dev/2026-10-04-windows-test-gaps.md`） |
| License | MIT |
| 作者 | Biaoyue (Bryce) Wang · <brycew6m@stanford.edu> · CoPaper.AI / Stanford REAP |
| PyPI | <https://pypi.org/project/StatsPAI/> |
| 导入别名 | `import statspai as sp` —— 所有示例、docstring、测试一律 `sp.xxx` |

---

## 2. 仓库结构

```text
StatsPAI/
├── src/statspai/          # 主包：90 子模块 / 1,566 个注册符号，其中约 400 个是结果类与异常类（实时数 `python scripts/registry_stats.py`；此行 2026-10-12 核对）
│   ├── __init__.py          # 对外 API 入口
│   ├── registry.py          # 函数注册表（sp.help / sp.list_functions 依赖）
│   ├── help.py              # sp.help / sp.describe_function / sp.function_schema
│   ├── cli.py
│   └── <领域模块>/          # did iv rd synth dml metalearners …
├── rust/statspai_hdfe/    # HDFE Rust 后端（PyO3）
├── tests/                 # pytest 套件 + reference_parity/ + external_parity/
├── docs/ · mkdocs.yml     # MkDocs 文档
├── paper.md · paper.bib   # JOSS
├── benchmarks/            # 性能基准
└── pyproject.toml
```

领域分组（对外 API 按下列七类组织）：

- **因果 / 处理效应**：`causal did rd iv synth dml metalearners tmle bcf bayes mcmc causal_impact policy_learning ope dtr multi_treatment qte principal_strat proximal mediation mendelian assimilation bridge`
- **面板 / 结构**：`panel fixest structural frontier multilevel gformula gmm msm longitudinal`
- **空间 / 时序**：`spatial timeseries bartik`
- **因果发现 / ML**：`causal_discovery dag neural_causal deepiv conformal_causal matrix_completion bunching causal_llm causal_rl causal_text fairness`
- **设计 / 抽样 / 推断**：`matching power mht survey bounds dose_response interference selection censoring imputation transport target_trial epi survival surrogate doe`
- **分解 / 诊断 / 回归**：`decomposition regression nonparametric diagnostics robustness postestimation inference smart`
- **基础设施**：`core utils compat fast datasets output plots workflow agent experimental question`

---

## 3. 设计原则

1. **一次 import，统一 API**。能力通过 `sp.<function>` 暴露，不需要二级 import。
2. **Agent 原生**。`sp.list_functions()` / `sp.describe_function()` / `sp.function_schema()` 必须对所有对外符号有效。
3. **统一结果对象**。优先 `CausalResult`（或领域结果类），实现 `.summary()` `.plot()` `.to_latex()` `.to_word()` `.to_excel()` `.cite()`。
4. **家族方法用 dispatcher**。`sp.synth(method=...)` / `sp.decompose(method=...)` / `sp.dml(model=...)`——一个入口，多种估计器。
5. **先对齐 Stata / R**。`fixest` / `did` / `rdrobust` / `gsynth` / `MatchIt` / Stata 已有的，先对齐 API 和数值再扩展。
6. **证据优先**。数值正确性是底线，每个估计器必须有参考对齐或解析测试。
7. **失败要响亮**。假设违背 → 抛异常或 `warnings.warn` + 写入结果 `diagnostics`；不吞异常返回 `None` / `NaN`。orchestration / best-effort 路径（`workflow/` `smart/` `paper`）catch `Exception` 时**必须**调用 `statspai.workflow._degradation.record_degradation(target, section=..., exc=..., detail=...)`——发 `WorkflowDegradedWarning` + 把 `{section, error_type, message}` 追加到 `target.degradations`。**禁止** bare `except Exception: pass`——静默降级是隐藏正确性回归的最便宜方式。
8. **弃用走流程**。`DeprecationWarning` + [`MIGRATION.md`](MIGRATION.md) 登记 + 至少一个小版本缓冲期。
9. **引用零幻觉**。任何文献引用必须现场核验 DOI / 作者 / 年份 / 期刊，**禁止凭 LLM 记忆补全**。详见 §10《引用与文献》——这是 StatsPAI 的红线。

---

## 4. 代码规范

### 对外 API

- 新对外函数**必须注册** → [`src/statspai/registry.py`](src/statspai/registry.py)，否则 `sp.help` / `sp.list_functions` 看不到。
- 给已注册函数**加参数也要登记**：在它的 `params=[...]` 里补一条 `ParamSpec`（描述就是 agent 看到的文档），再 `python scripts/dump_schemas.py`。漏了的话 `sp.function_schema()` 里没有这个参数，agent 不知道它存在；pre-push hook `registry-params`（2026-10-05 起）会拦。
- docstring 用 NumPy 风格，包含 `Parameters` / `Returns` / `Examples` / `References`。
- **agent 卡片里能从代码读出来的内容不要手写（2026-10-14 起）。** 三处各有一个来源和一道闸门：(1) 选择型参数的 `enum`：注解是 `Literal`、或 docstring 里的列表被模块源码印证时自动得到；只由代码陈述的（函数体里的校验、模块常量、原样转交给另一个函数）登记在 `src/statspai/_enum_seeds.py`，`tests/test_enum_seeds.py` 会回到那个位置重新核对，**不要**把记忆里的取值写进去，语法型参数（`vce='cluster firm'`、lavaan 模型串）不给枚举。(2) 没有任何失败模式的函数，取自它自己抛出的带 `recovery_hint` 的类型化异常：改了这类 `raise` 之后跑 `python scripts/gen_source_failure_modes.py`（`tests/test_source_failure_modes.py` 查漂移），所以**新函数用 taxonomy 异常并写 `recovery_hint`，卡片就自动有内容**。(3) `sp.search_functions` 的中文词表在 `src/statspai/_discovery_zh.py`，条目写成 `\uXXXX` 转义、上一行注释写汉字（JSS 归档会把源码里的非 ASCII 转写掉），加词时两处都要写，测试会比对。工具层（MCP / `execute_tool`）对未传的 `seed=None` 类参数注入默认种子并在返回里写 `seed_injected`（`agent/_default_seed.py`）；库函数自己的默认值不动。
- `References` 段只写 **bib key**（对应 [`paper.bib`](paper.bib)）或**经过核验的规范引用**——禁止在 docstring 里手写未核验的引用字符串，详见 §10。
- 示例一律 `import statspai as sp` + `sp.xxx`。
- 破坏性改动 → [`MIGRATION.md`](MIGRATION.md) + `DeprecationWarning`。
- **核心层签名冻结（2026-10-17 起）。** `CORE_FUNCTIONS`（`sp.list_functions(core=True)` 的 32 个）的签名记在 `tests/core_api_snapshot.json`，删参、改默认值、可选变必选、调换位置参数、新增无默认值的参数都会被 `tests/test_core_api_freeze.py`（pre-push hook `core-api-freeze`）拦住。新增带默认值的参数不算破坏。有意的破坏先走弃用流程，再 `python scripts/core_api_snapshot.py --write --ack "fn.param=<MIGRATION.md 里的原文>"`。
- **`sp.*` 统一接受 polars / Arrow（2026-10-17 起）。** `statspai/__init__.py` 末尾用 `_frame_input.install` 包装每个公开函数（惰性导出在 `__getattr__` 里包装），并在定义模块里同名换绑，所以 `模块.函数 is sp.函数`、pickle 按引用可用、`__module__` 不变。polars / pyarrow / `__arrow_c_stream__` / `__dataframe__` 对象在调用时转成 pandas；pandas 参数**以同一对象**直通（`tests/test_frame_input.py` 守着）。正因为直通，`_frame_input.py` 被两个 provenance tracer 排除在估计路径之外；**改这个文件时不得让 pandas 输入走任何计算**，否则排除就不成立。新函数不用做任何事；不要在估计器里再写 polars 分支。

### 模块内部

- 共享基元放模块级 `_core.py` / `_common.py`（参考 `rd/_core.py`、`decomposition/_common.py`）。**不要**在多个文件重复实现 kernel / WLS / sandwich / 影响函数。
- 私有函数 `_` 前缀。
- 单文件 ~800 行以内，按关注点拆（estimator / inference / diagnostics / plots），不是单纯按行数拆。

### 依赖

- 核心依赖精简（见 `pyproject.toml`）。重依赖归入 extras：`dev` / `performance` (jax) / `bayes` (pymc) / `neural` `deepiv` (torch) / `fixest` (pyfixest) / `plotting`。
- `torch` / `jax` / `pymc` 必须**惰性 import**，用户没装 extra 不应触发 `ImportError`。
- 禁止引入 GPL / AGPL 依赖（和 MIT 冲突）。

---

## 5. 测试

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

### 5.1 Parity 规则（JSS 论文的核心断言，改动前必读）

规则全文在 `tests/r_parity/compare.py` 顶部 docstring、[`docs/dev/r_parity_tolerances.md`](docs/dev/r_parity_tolerances.md)、[`docs/guides/stability.md`](docs/guides/stability.md)；这里只列硬约束。

- **默认门槛**：同一份 CSV 字节、同一估计量，StatsPAI 与 R / Stata 参考的相对误差 **≤ 1e-6**（点估计与 SE 各自计）。这是 T2「同字节严格 parity」的定义，87 个 Track A 模块里绝大多数实际落在 1e-15 到 1e-9。
- **四个证据等级**：T1 已知真值回收（解析 DGP，无外部参考）；T2 严格跨语言 parity；T3 随机估计器的**种子复制等价**（固定数据、两侧各跑多个种子，比较种子分布；等价界按抽样 SE 事先写明，用 TOST 判定——单次各跑一次、拿抽样 SE 当 Monte Carlo 误差做分母，不算 T3；见 `tests/reference_parity/test_grf_seed_mc_equivalence.py`）；T4 有记录的约定差异或参考实现之间自身分歧（如 Basque SCM，R `Synth` 与 Stata `synth` 自己就不一致）。**只有 T2 可以写成「对齐 R / Stata」**，T3 / T4 必须按原等级表述，不得记为 parity 胜利。**达不到 T3 标准的随机比较**（单种子或少数几个种子、MC 容差、一个网格步长——如 CS bootstrap、fect/interflex CV、GRF 家族的已知真值筛查、HonestDiD C-LF）一律记为 **S（stochastic screen）**：只报告、不评级，不得写成 T3（2026-09 JSS v2 审稿指出论文把这几类都叫 T3）。
- **放宽只有两种合法理由**：两边计算的是有文档的不同量（自由度除数、ssc 小样本簇修正、解析 SE 对影响函数 SE），或方法论上不可能确定性一致。「算法相同但对不上」不是理由，是 bug。
- **放宽必须登记**：容差写入 `tests/r_parity/compare.py::TOLERANCES`（预注册预算，按模块一条）；≥ 5e-2 的条目在 `docs/dev/r_parity_tolerances.md` 打 A（机制明确）/ B（经验）/ C（未能解释）等级并写明机制；Stata 侧超预算的模块登记到 `compare.py::STATA_HEADLINE_GAP_EXCEPTIONS`；没有 Stata 参考的模块在 `STATA_SKIP_REASON` 写实测过的原因。**禁止**为了让某个测试通过而单独放宽一个模块。
- **Stata 与 R 共用一个预算**，不为 Stata 另设更松的门槛。
- **Track A 的 Python 侧必须跑原生实现**：`backend="honestdid"` / `backend="r"` 这类委托参考实现本身的调用，拿来和 R 比等于 R 对 R，**不得**作为 parity 行（10/21 号模块 1.31 前就是这样，JSS 审稿抓出）。官方 Python 端口（如 `bwselect="cct"` 的 rdrobust 包）只能作不参与 join 的旁证行；第三方 Python 库（linearmodels / statsmodels / pyfixest）服务的模块登记在 `compare.py::IMPLEMENTATION_PROVENANCE`，附录表打 † 标记。分类由 `scripts/trace_parity_provenance.py` 实跑调用追踪核验（`tests/test_parity_implementation_provenance.py`）。trace 绑定入口脚本、**估计路径上每个被执行的 StatsPAI 源文件**与已提交结果文件的 SHA-256：**改了任何 Track A 模块估计路径上的源码（不只是模块 .py）都要重跑受影响模块的 trace**（`python scripts/trace_parity_provenance.py NN ...`，全量约 15 分钟）；名单外的包记为 `unclassified`，要人工审查后加进名单。**ledger 有两个**：Track A（`tests/r_parity`）和原始数据（`tests/orig_parity`，重录要加 `--ledger orig`）；改了 `regression/ols.py`、`regression/iv.py`、`matching/`、`synth/scm.py`、`inference/ipw.py` 这类两边都执行的文件，两个都要重录。**重录之前先实跑模块、把结果与已提交的 `*_py.json` 做 diff**：trace 只绑定哈希，不检查已提交的结果还能不能从当前代码复现（末位抖动在 1e-9 复现容差内就 `git checkout` 还原；真变了就是冻结产物变了，按下文走重生成 + JSS 记录）。pre-push hook `parity-traces`（2026-10-05 起）检查两个 ledger 是否描述当前树——在此之前没有任何推送前检查，原始数据 ledger 在两条并行线下过期而无人察觉，LaLonde 的 `psm_att` 一行已经复现不出来。
- **每一行 SE 都被闸门盯着**：`tests/test_parity_harness_contract.py::test_every_r_se_row_is_inside_budget` 对每个 PASS 模块的**所有** R 侧 SE 行按注册预算门控；Stata 侧超预算的 SE 行必须在 `compare.py::STATA_SE_GAP_NOTES` 登记机制（且要能从我们自己的量重建出对方的数字）。GitHub-only 的 Stata 参考（如 `fect_stata`）在 do 文件里 `net install` 到 `tests/stata_parity/_ado_fect/`（已 gitignore），不要装进用户的 PLUS。
- **golden 文件不得手改**：`tests/r_parity/results/*_R.json` 与 `tests/stata_parity/results/*_Stata.json` 由 `tests/r_parity/TIER_A_FIXTURE_LOCK.json` 哈希锁定，只能通过 `verify_reproduce.py` / `verify_reproduce_stata.py` 实跑重生成；复现性容差为 **1e-9**，与 parity 容差无关，parity 容差从不豁免复现性漂移。
- **新增 Track A 模块：不要照抄下面这段清单，跑闸门。** 权威定义是 `tests/test_parity_harness_contract.py`（42 条断言，其中 `test_parity_artifact_inventory_has_explicit_contracts` 直接断言 `py_modules == set(TOLERANCES) == set(HEADLINE)`、`set(STATA_SKIP_REASON) == py_modules - stata_modules`），加上 `python scripts/tier_a_fixture_lock.py`（哈希锁）与 `cd Paper-JSS && make audit`。**流程是：写完模块 → 跑这三个 → 按报错补齐**，而不是对着清单打勾。

  下面这份是给人看的概览，**不是**验收标准，可能滞后于闸门：`NN_<method>.py` + `.R`（有 Stata 参考再加 `tests/stata_parity/NN_<method>.do`）；CSV 入 `tests/r_parity/data/`；`compare.py` 的 `TOLERANCES` **和** `HEADLINE` 各登记一条；三侧 reproducibility report 各补一行（`REPRODUCIBILITY_REPORT.md` / `_PY.md` / `_STATA.md`，且**必须跑全量重生**——`verify_reproduce.py <单模块>` 会用那一个模块覆盖整份报告，删掉其余几百行）；`TIER_A_FIXTURE_LOCK.json` 重生；`python scripts/build_parity_index.py` 重生 `docs/parity.md`；schema 包若因签名变动而漂移则 `python scripts/dump_schemas.py`。registry 证据备注**不用手写**，由 index 自动派生。

  为什么改成这样：2026-08 的 82–85 号模块漏了 Stata report 让 JSS 审计整体变红，于是有了上面那份清单；2026-09 加 88 号时**照着清单做仍然漏了四项**（`HEADLINE`、`_PY.md`、fixture lock、schema），闸门抓出 7 个失败。清单是照"上次漏了什么"写的，闸门是照"实际断言什么"写的——只有后者会自己更新。

#### 对不上怎么办：决策树（按顺序走，不得跳步）

StatsPAI 承诺的不是「和 Stata / R 数字一样」，而是**每一个数字要么对上参考实现，要么对上已知真值，要么带一份写清楚为什么对不上的说明**。「对不上又说不清」是唯一不被允许的状态。

1. **先假定是我们错了。** 同一份 CSV 字节做二分：设计矩阵、权重、残差、meat 矩阵逐个对照，定位第一个分叉点。绝大多数「对不上」在这一步终结于修 bug，走 CHANGELOG / MIGRATION 的 ⚠️ correctness fix。CS-DiD 的 `weights=` 从未传进 staggered 分支、静默返回未加权估计量，就是靠 `did::att_gt(weightsname=)` 对照抓出来的，单元测试没抓到。
2. **定位到了，是约定差异**（自由度除数、ssc 小样本簇修正、解析 SE 对影响函数 SE）。两边都对。处理：默认值跟随原方法作者的实现，能便宜暴露的加参数让用户切换，docstring 写明，`TOLERANCES` 登记为约定差异档并写出机制，差距大小必须被容差限住。
3. **定位到了，是参考实现自身有问题或解不唯一。** 不复制别人的 bug。保留我们的实现，记 T4「参考分歧」，**必须附独立证据**证明我们是对的（第二个参考、解析恒等式、或唯一识别的 DGP，如 `52_scm_unique` 之于 `07_scm`；`74_cic` 的分位数 tie-break 亦属此类）。Stata 侧登记到 `STATA_HEADLINE_GAP_EXCEPTIONS`，理想上向上游报告。
4. **方法论上不可能确定性一致**（forest 随机数、bootstrap、placebo）。走 T3：固定数据、两侧多种子重拟合，报算法 Monte Carlo SD 与种子均值差，按事先写明的等价界（相对抽样 SE）做 TOST，不写成 parity。**抽样 SE 不是算法 Monte Carlo 误差**——2026-09 JSS 审稿指出森林行曾拿 `sqrt(se_py²+se_R²)` 当分母，改用种子复制后两引擎在 500 / 2000 / 8000 棵树、两个数据集上都在 0.1 抽样 SE 内等价，种子间 SD 相当（2000 棵树时约抽样 SE 的 6–8%）。**种子必须拉开间隔**（两侧都用 `1000 + 100000k`）：grf 相邻种子长出的森林共享大部分随机抽样，连续种子会把 grf 的 MC SD 低估数倍——中途一版因此误报「StatsPAI 森林比 grf 抖 2.5–34 倍」，已撤回。StatsPAI 旧引擎（1.31 前）相邻 random_state 也共享大部分树，已修。
5. **定位不到。** 不允许称为 aligned / certified。`docs/dev/r_parity_tolerances.md` 打 C 级「未能解释」，论文里以 open item 出现，证据等级停在 `validated`（已知真值回收）或更低。Sun-Abraham 聚合方差与 `fixest` 差 0.7% 到 2.2% 曾经就是这么处理的：点估计对到 1.6e-9，方差钉住并标为未决，不把容差放宽到 3e-3 让它变绿。2026-09 回到第 1 步二分后发现是我们的 bug（设计矩阵含 9 个全零的幽灵 cohort×event 列、时间固定效应没按 nested 规则计入 K），修掉后与 Stata `eventstudyinteract` 对到 8e-12，与 `fixest` 只差有文档的 Prop. 3 份额方差项（`share_variance=False` 可复现 fixest 到 1e-9）——「未决」是诚实的中间状态，不是终点。

任何一格里都禁止：为了变绿悄悄放宽容差、手改 golden JSON、删掉模块让问题消失。

#### R 与 Stata 自己不一致时跟谁

- **跟原方法作者维护的实现**，那个是 canonical reference；移植版是 bridge。CS-DiD 跟 R `did`（Callaway & Sant'Anna 自己写的），Stata `csdid` 是移植；HonestDiD 跟 Rambachan & Roth 的 R 包；DoubleML 跟 R / Python `DoubleML`，Stata `ddml` 是 bridge；`rdrobust` / `rddensity` 两边都是 Cattaneo 团队维护，两边都必须对。
- 实践中 R 侧是主参考（`compare.py` 称 "canonical R reference"），Stata 侧是 "canonical or audited bridge"；两侧共用同一个 `TOLERANCES` 预算。
- 两个参考彼此都超预算时（如 Basque SCM 的 R `Synth` 对 Stata `synth`），该行自动降为 T4，`methodological_gap_ledger` 会要求给出分类和晋升路径；不得挑对得上的那一边写成 T2。
- 只有 Stata 实现、没有 R 实现的方法，Stata 就是 canonical；只有一侧实现的模块在 `STATA_SKIP_REASON` 写实测理由（`ssc describe` 返回码、目标估计量不同等），不得写「未安装」这类未经核实的理由（2026-08-06 曾因此错过 5 个可对的模块）。

---

## 6. Rust 组件

路径 [`rust/statspai_hdfe/`](rust/statspai_hdfe/)，HDFE 高性能后端，PyO3 打包，在 Python 侧通过 `sp.fast.*` / `sp.fixest.*` 暴露。

```bash
cd rust/statspai_hdfe && maturin develop --release   # 本地装入 venv
```

- **可选**：Python 侧检测到 Rust 不可用会回退 numpy / pyfixest，不报错。
- **CI 跳过 Rust**：`STATSPAI_SKIP_RUST=1`。

---

## 7. 发布

PyPI 凭据在 `~/.pypirc`——**不要**提交仓库、不要写进 memory。完整流程见 `memory/reference_pypi_publish.md`。

简流程：

1. Bump `pyproject.toml` + `__version__`。
2. 更新 [`CHANGELOG.md`](CHANGELOG.md)（`Added / Changed / Fixed / ⚠️ Correctness`）。
   版本号一改、tag 还没打，就进入**发版窗口**：`python scripts/release_gate.py --fix-census` 把 `docs/guides/stability.md` 与 `docs/jss_source_audit_dossier.md` 里的 registry 统计改成当前值（同时把根目录 `server.json` 的两处版本号改成新版本），再 `python scripts/release_gate.py --check` 确认（pre-push hook `release-gate` 与 `tests/test_release_gate.py` 在窗口内强制统计一致，窗口外只查另外两项）。这三项是 1.33.0 / 1.34.0 漏掉、靠构建论文复制包才发现、各赔了一个补丁版本的问题：统计过时、正则被 ASCII 转写改了语义（模式里的非 ASCII 字符一律写 `\uXXXX`）、包内文本出现宣传措辞。
3. `pytest -q` 全绿 + `pytest tests/reference_parity/ -q` 必过。
4. `rm -rf dist/ && python -m build && twine check dist/*`。**再把 sdist 的文件清单对照 `git ls-files`**：`MANIFEST.in` 的 `recursive-include tests *` 会把工作树里被 gitignore 的本地产物一起打包——1.32.0 发版时先后抓到复现检查的临时目录、Track C 的 38 MB 共享输入、以及可由包内 `sp.datasets.nhefs()` 逐字节重生、因而不入库的 NHEFS 副本，sdist 一度从 25 MB 涨到 63 MB。新出现的 gitignore 产物目录要在 `MANIFEST.in` 里 `prune`。
5. 干净 venv 装 wheel 冒烟测试。
6. `git tag vX.Y.Z && git push && git push --tags`。
7. `twine upload dist/*`。

**默认不发 GitHub Release（2026-09-26 起）。** 发版 = TestPyPI + PyPI（本地 twine）+ 打 tag，到此为止；**不要**跑 `gh release create`，也不要在网页上 Publish Release。原因：Publish Release 会同时触发两个不可逆动作——Zenodo 铸出永久不可删的 version DOI，以及 `ci-cd.yml` 的 `release: published` 再上传一次 PyPI（与本地 twine 重复）。只推 tag 两者都不触发。只有用户在**当前会话**明确要求（如期刊需要 Zenodo version DOI）时才发，并先按下文「Zenodo 归档」一条做核对；发版总结里要写明「未创建 GitHub Release」。

---

## 8. 文档

- MkDocs（[`mkdocs.yml`](mkdocs.yml)），源在 [`docs/`](docs/)，`mkdocs serve` 本地预览。
- 当前 guide：`synth` / `choosing_did_estimator` / `choosing_iv_estimator` / `choosing_rd_estimator` / `choosing_matching_estimator` / `callaway_santanna` / `cs_report` / `honest_did` / `repeated_cross_sections` / `robustness_workflow` / `bayesian_econometrics` / `migration-from-r` / `mixtape_ch09_did` / `stock_watson_4e` / `design_based_econometrics` / `ding_first_course` / `xu_lan_causal_econometrics` / `facure_causal_inference_in_python` / `schuler_vanderlaan_modern_causal_inference` / `forecasting_fpp` / `time_series_econometrics` / `design_of_experiments` / `wager_causal_inference` / `regression_and_other_stories` / `yuksel_aydede_causal_ml` / `conditional_inference_trees` / `figure_editor`。相关估计器改动要同步对应 guide。
- JOSS 论文 [`paper.md`](paper.md) + [`paper.bib`](paper.bib)——对外 API 或项目范围变更时同步。

---

## 9. Git 协作

- **🚨 提交闸门（本节最高优先级，压过下面所有条款）**：**2026-09-28 起，用户已给出常设授权**：agent 判断合适时可以直接 `git commit` + `git push origin HEAD:main`，不必每次再问。"合适"指同时满足：(1) 相关测试与 pre-commit / pre-push 闸门全绿（不得 `--no-verify`）；(2) 只暂存自己这条线的改动，逐文件 add，绝不 `-A` / `-a`（§9.2）；(3) 改动完整、自洽，不是半成品；(4) 触及 JSS 冻结产物时已在 `docs/dev/jss_review_changes.md` 记录。拿不准就先问。**以下仍须用户在当前会话明确授权**：打 tag、发 PyPI / TestPyPI、GitHub Release、`--force` / 改写已推送历史、删除远程分支。每段工作结束时照常用中文总结，写明推送了哪些 commit。本地 `PreToolUse` hook（[`.claude/hooks/block-git-commit-push.py`](.claude/hooks/block-git-commit-push.py)）仍会拦截，判断合适后在命令前置 `STATSPAI_ALLOW_GIT=1` 放行，这是一道有意识的确认，不是绕过。
- **获得授权之后**才适用以下条款：默认分支 `main`，**直推 main**、默认不开 PR，除非明确要求（见 `memory/feedback_no_pr.md`）。
- Commit 风格：`feat:` / `fix(<area>):` / `docs(<area>):` / `chore:`，摘要 ≤ 72 字符。
- **禁止**：`--no-verify` / `--no-gpg-sign` / `--force`（除非明确授权）；对已推送 commit `--amend`。出错用 `git revert`。
- **push 前必须让 `python3` 指向本仓库 venv**，否则 pre-push 必然红。`.pre-commit-config.yaml` 里那批闸门（`flake8-count` / `registry-drift` / `schema-drift` / `error-taxonomy` / `orchestration-assertions` / `examples-coverage` / `parity-traces` / `registry-params` / `cold-import budget`）都是 `language: system` + 裸 `python3`，会按 PATH 解析；在没激活 venv 的 shell 里解析到系统 Python（如 Homebrew 3.14），直接 `ModuleNotFoundError: No module named numpy`，看起来像代码坏了，其实是解释器不对。

  ```bash
  source .venv/bin/activate && git push origin HEAD:main
  # 或（worktree 里不想污染环境时）
  PATH="/path/to/StatsPAI/.venv/bin:$PATH" git push origin HEAD:main
  ```

  `flake8-count`（2026-10-03 起）就是 CI 的 `python scripts/quality_gate.py flake8`：`src/statspai` 的违规总数不得超过 `DEFAULT_FLAKE8_MAX`。超长的字符串拆成相邻字面量，black 不会替你拆。这道闸门以前只在 CI 跑，结果 main 一周涨了 191 条没人察觉，而它在 mypy 和 pytest 之前失败，CI 从 09-28 到 10-03 一个测试都没跑。

  **不要**因此改 hook 的 `entry`：CI 里 `python3` 本来就是对的，把它钉死到 `.venv/` 反而会让 CI 红。这是本地环境问题，不是配置问题。**更不要**用 `--no-verify` 绕过——这六道闸门是真在挡东西。

### 9.2 并发：多窗口同时开工必须各自占一个 worktree

**开工前先看这条。** 如果另一个 Claude 窗口（或同事）正在本仓库作业，**不要**两边都在主工作树的 `main` 上改。

实测代价（2026-08-01，两条线并行一晚）：多个文件进入永久争用。每次提交要手工做「备份共享文件 → 还原到 HEAD → 重生成派生产物 → 提交 → 还原」五步，做了四轮；推送闸门按*已提交*状态检查而工作区混着两边改动，两个视角每次都打架。更糟的是有一次 `-A` 式全量提交把另一条线未提交的工作整个扫了进去，代码上了 main 却挂在毫不相关的 commit message 下。

**争用点清单**（2026-09-24 更新；原先只列了八个，实测不止）：

| 类别 | 文件 | 为什么争 |
| --- | --- | --- |
| 手工追加点 | `registry.py`、`__init__.py`（三处：import 块 / `__all__` / `_register_lazy`）、`CHANGELOG.md`、`MIGRATION.md`、`CLAUDE.md` 本身 | 两边都往同一段尾部追加 |
| **计数行** | `README.md`（英文，默认，PyPI 长描述用它）、`README_CN.md`（中文）、`docs/index.md`、`docs/reference/index.md` | 四处手写的「N 个注册函数」，由 `registry_stats.py --check` 门控。**对方加一个函数，你这四行同时作废** |
| 生成产物 | `schemas/*` **与** `src/statspai/schemas/*`（包内镜像）、`_parity_index.json`、`docs/parity.md`、`docs/stats.md`（两类冲突：at-a-glance 行 + 按模块行） | 纯粹因为两边都重新生成 |
| 证据视图（2026-10 起） | `docs/evidence_inventory.{json,md}`、`docs/reproduction_manifest.json`、`docs/dev/agent_card_audit.{json,md}`、`docs/dev/2026-10-02-review-status.md` | 都是生成的，各有 `--check`。**何时重生**：改了 `validation_scope.py`、任何 `tests/stata_parity/option_parity/` 或 `tests/stata_translation_holdout/` 下的文件、Track B 结果 → `python scripts/build_evidence_inventory.py`（在 `build_parity_index.py` 之前）；重推导了 R / Stata golden 或改了 `r-parity.yml` 的模块列表 → `python scripts/build_reproduction_manifest.py`；给那 60 个被审查的入口（`scripts/agent_card_audit.py::AUDITED`）加了 schema 枚举值 → **不必立刻重跑**：`tests/test_agent_card_audit.py` 会把报告没见过的新值现场真实调用一遍，只在函数拒绝该值时失败；方便时再 `python scripts/agent_card_audit.py`（约 10 分钟）刷新报告，新值积压超过 25 个时测试会要求刷新；改了 `docs/dev/review_backlog.json` → `python scripts/build_review_status.py` |
| 字节同步对 | `paper.bib` ↔ `src/statspai/paper.bib` | 必须逐字节一致 |
| 棘轮基线 | `scripts/signature_house_style_baseline.json`、`quality_gate` 的 mypy / flake8 基线 | 只降不升，两边都想动 |

**计数行不要自己算。** 别拿「我加了 2 个函数所以 1,190 → 1,192」去改——并行期间对方也在加。跑 `python scripts/registry_stats.py --check`，**它报什么数字就填什么**，再用 `--table` 重生 `docs/stats.md` 的对应模块行。2026-09-23/24 一晚上这个数被推了四次（1,189 → 1,190 → 1,192 → 1,197 → 1,198）。

**派生产物有生成顺序：`build_parity_index.py` 必须在 `dump_schemas.py` 之前。** registry 的证据备注由 parity index 派生（§5.1 末句），schema 包又把备注嵌进去；顺序反了，推送时 `schema-drift` 闸门会拦，而报错只说 schemas 陈旧，不会提示是索引的锅。

**rebase 撞到生成产物时不要手工合并冲突**——取对方的版本再重生（`checkout origin/main -- <那批派生文件>`，然后按上面的顺序重跑两个脚本，最后按 `registry_stats.py --check` 报的数字改那四行计数）。只有手工追加点（`CHANGELOG.md` / `CLAUDE.md` / `registry.py`）才逐块合。注意对方**发版**时会把你的 `## [Unreleased]` 段整个提升成 `## [X.Y.Z]`——你的条目要另起一个新的 Unreleased 段，不要塞回已发布的版本里。

**提交被 hook 改文件而失败之后，不要顺手 `--amend`。** pre-commit 的 black / isort 会重写文件并让本次提交失败；此时最自然的下一步「重新 add 再 `--amend --no-edit`」会把新改动并进**上一个、很可能已经推送过的** commit，挂在毫不相干的 message 下。2026-09-23 真发生过，靠 `reset --soft <那个已推送的 sha>` 才救回来。正确做法是重新暂存后**新起一次**提交。

**做法**：

```bash
git worktree add .claude/worktrees/<线名> -b wt/<线名>
cd .claude/worktrees/<线名>
```

完成后**不必切回主树**即可并入 main（主树可能压着别人未提交的工作，绝不要在那里 merge/checkout）——推送时用 `HEAD:main` 引用即可快进；非快进先 `git rebase origin/main`。

**必须带 PYTHONPATH。** 仓库是 editable 安装，`statspai` 被钉死在**主**工作树，裸跑 `import statspai` 仍会加载主树代码——测试跑在别人的改动上，隔离形同虚设：

```bash
PYTHONPATH="$(pwd)/src" python3 -m pytest ...
PYTHONPATH="$(pwd)/src" python3 scripts/dump_schemas.py
```

自检：worktree 内 `len(sp.list_functions())` 必须等于 `python scripts/registry_stats.py --check` 报的数字，不等就是没生效。**不要**为此往 `pyproject.toml` 加 pytest `pythonpath`——主树的 editable 安装对另一条线是正确的，改共享配置等于把刚消除的争用造回去。

**如果只能留在主树**：提交前务必 `git status` 确认哪些改动不是自己的，**逐文件 / 逐 hunk** 暂存（`git add <file>` 或 `git apply --cached`），绝不用 `-A` / `-a` 全量提交；派生产物要先把共享源文件还原到 HEAD 再重新生成，否则会把别人未提交的内容一起固化进去。

### 9.1 例外：远程 runtime（Colab / Lambda / RunPod / CI）回传结果走 PR

直推 main 的前提是"操作在本地，作者审过"。从远程 runtime（**最典型的是 [`Paper-JSS/colab_gpu_bench.ipynb`](Paper-JSS/colab_gpu_bench.ipynb)**）自动回传 benchmark 结果时，本地审视环节缺失，**必须改走 PR**：

- **允许的 payload**：
  - `tests/perf/results/05_*.json`（或对应 bench 编号的 JSON）
  - `tests/perf/results/_provenance_*.txt`（git SHA / JAX 版本 / GPU 型号）
  - `tests/perf/results/_log_full_*.txt`（subprocess stdout/stderr）
  - 可选：同步更新的 `paper.md` / `Paper-JSS/manuscript/sections/06-performance.tex` 表格数字
- **禁止的 payload**：源码改动、依赖变更、新增其他模块——这些走常规直推 main，不能搭 benchmark PR 顺风车。
- **分支命名**：`bench/<bench-name>-<gpu-tag>-<yyyymmdd>`，例：`bench/05-feols-t4-20260518`。
- **PR 标题**：`bench(<bench-name>): <gpu-tag> results — n=..., B=..., commit=<short-sha>`。
- **PR body 必填**：
  - 跑这次 benchmark 用的 git commit SHA（应与 `_provenance_commit.txt` 一致）
  - GPU 型号 / JAX 版本（与 `_provenance_gpu.txt` / `_provenance_jax.txt` 一致）
  - 验收点：`cell 16` 输出的 speedup 表格 + 自动生成的 LaTeX snippet 贴在 body 里
- **身份**：远程 runtime 用 `GITHUB_TOKEN`（fine-grained，仅 `contents:write` + `pull-requests:write`，scope 限本仓库），不要复用 user PAT。Token 通过 Colab Secrets 或 runtime env 注入，**禁止**写进 notebook 源码或 commit message。
- **合并策略**：本地审视过 JSON 数字合理（speedup 不离谱 / 没退化）后 **squash merge**；不合理则 close + 排查。
- **频率**：同一 bench 同一 GPU 同一天**只允许一个 open PR**，避免噪音。重跑覆盖旧 PR 用 force-push 同分支（这一条是 §9 "禁止 force" 的例外，因为 PR 本身就是审视点）。

---

## 10. 引用与文献（零幻觉红线）

> 捏造一条引用，代价是整个包数值正确性的可信度。一位计量经济学读者点开 DOI 发现不存在，会立刻怀疑 StatsPAI 的所有估计器——**这是最便宜的质量杀手，必须零容忍**。

### 四要素核验

任何**新增**引用（docstring / README / `CHANGELOG.md` / `MIGRATION.md` / `paper.md` / `docs/guides/` / commit message 均适用）落地前必须独立核验：

1. **作者**——完整名单、拼写、顺序、大小写
2. **年份**——正式发表年（若引预印本，显式标注 `arXiv preprint, YEAR`）
3. **标题**——完整标题，不省略副标题
4. **期刊 / 会议 / 出版方** + **DOI 或 arXiv ID**

核验来源至少 **2 个独立渠道**（Crossref / doi.org / arXiv / 期刊官网 / Google Scholar 取其二），单一二手来源不作数。**禁止凭 LLM 或训练语料记忆补全**——哪怕是 Abadie (2003)、Callaway & Sant'Anna (2021)、Chernozhukov et al. (2018) 这类你"非常确定"的论文，也一律现场核验后再写入文件。

### `paper.bib` 单一来源

- 核验通过的引用一律落到 [`paper.bib`](paper.bib)，bib key 用 `lastnameYEARkeyword` 规范（例：`abadie2003economic`、`callaway2021difference`、`chernozhukov2018double`）。
- docstring 的 `References` 段、`docs/` 教程、`paper.md` 正文**只引 bib key 或引用一份规范字符串**；禁止在多处手写同一条引用，避免格式漂移。
- 发现 `paper.bib` 已有条目信息错漏 → 修一处全仓受益；同时 `git grep` 扫手写副本一并修正，防止旧字符串残留。

### master 与派生子集（2026-09-04 起）

- 根目录 `paper.bib` 是 **master**：全仓唯一可手工编辑的 bib。JOSS 论文（`paper.md`）直接引用它；docstring、`sp.bibtex()`、MCP `bibtex` 工具也读它。
- **各篇论文的 bib 一律是派生子集，禁止手改**。JSS 手稿的 `Paper-JSS/manuscript/jss-bib.bib` 由 `python tools/bib_subset.py extract --roots Paper-JSS/manuscript/main.tex --out Paper-JSS/manuscript/jss-bib.bib --keep ...` 从 master 抽取（`make -C Paper-JSS bib-split`）；手稿要引新文献，先按四要素核验加进 master，再重新抽取。长版章节独占的引用走第二个派生文件 `jss-bib-archival.bib`（`--roots main.tex sections/*.tex --minus jss-bib.bib`）。`bib_subset.py check` 是漂移闸门（pre-push hook `bib-subset-jss` / `bib-subset-jss-archival`）。
- **wheel 里带一份 master 副本** `src/statspai/paper.bib`，必须与根目录字节一致（`python tools/bib_subset.py packaged --sync` 同步；pre-commit hook `bib-packaged-sync` 和 citation-audit 的 Gate 1b 校验）。运行时通过 `statspai._bibpath.master_bib_path()` 解析：源码树优先，其次包内副本，两者都没有则抛 `FileNotFoundError`——**不再**回退到当前目录的 `paper.bib`。
- **条目必须 BibTeX 安全**：字段内**禁止原始非 ASCII 字符**（`Ørregaard` 要写 `{\O}rregaard`，`é` 写 `{\'e}`，破折号写 `--`，`R²` 写 `$R^2$`），否则 BibTeX 做姓名缩写会切断多字节序列，PDF 里出现乱码。**核验记录写在 `annote`**（所有 .bst 样式都忽略它），`note` 只放读者可见的短句（如 "arXiv preprint, first posted 2025-06-21"），因为 `note` 会被打印进参考文献，而且里面的 `_` `&` `%` 会直接让 LaTeX 报错。`tests/test_bib_subset.py` 有守卫测试。
- 同一文献只能有一个 key。发现某篇论文用了不同 key 引同一文献，改论文的 key，不加重复条目（`audit_bib_duplicates.py --strict` 会挡）。

### 交付前自检

- Commit / PR 触及新引用 → 在 message / 描述里注明 "refs verified via `<source1>`, `<source2>`"。
- Review / self-review 对任何陌生引用默认按"未核验"处理——要求 DOI 可点开、arXiv 可访问，或 Crossref 能直接搜到。
- **宁缺毋滥**：拿不准时写 `（citation needed）`、只引 bib key 占位、或干脆不引——**捏造一条引用比缺失糟糕一百倍**。

---

## 11. 分领域须知

各领域的约定（参考实现的口径、不要改回去的写法、证据怎么做、未决项）在 [`docs/dev/modules/`](docs/dev/modules/)，按领域一个文件。**改某个领域的代码之前，先读它的文件**；领域交叉时（例如 DiD 估计器的协变量）两个都读。新约定写进对应文件，不再写进本节。下表是索引，第三列是各文件里的条目标题，用来判断该读哪一个。

| 文件 | 领域 | 条目 |
| --- | --- | --- |
| [`rd.md`](docs/dev/modules/rd.md) | 回归断点（RD） | `rd/`；`rd/optimized.py`（`sp.rd_optimized`，Imbens-Wager 2019） |
| [`synth.md`](docs/dev/modules/synth.md) | 合成控制 | `synth/`；`synth/_gsynth_multi.py`（`sp.gsynth(treat=)`，1.39 起） |
| [`did.md`](docs/dev/modules/did.md) | 双重差分与事件研究 | `did/did_forest.py`；`callaway_santanna(notyet_cutoff=)`；`did/es_inference.py`；CS 的聚类变量；`did/_arp.py::rm_confidence_set`；`sp.drdid` 的面板路径；`sp.did_multiplegt`；`did/few_treated.py`；dCDH 家族（`did/did_multiplegt.py` / `did_multiplegt_dyn.py` / `twowayfeweights.py`）；`sp.sun_abraham`；DiD 的日期型时间列（`did/_core.py::calendar_time_aware`） |
| [`decomposition.md`](docs/dev/modules/decomposition.md) | 分解 | `decomposition/` |
| [`panel_multilevel.md`](docs/dev/modules/panel_multilevel.md) | 面板、多层模型、前沿、动态面板 GMM | `multilevel/` / `frontier/` / GLMM；`gmm/_dynpanel/`；`panel/` 的随机效应方差分量只有一份实现；`mackinnon1994_pvalue`（`panel/unit_root.py`）在拟合范围之上返回 1 |
| [`bayes_mcmc.md`](docs/dev/modules/bayes_mcmc.md) | 贝叶斯（PyMC 与 mcmc/） | `bayes/`；`mcmc/`（1.39 起）；`mcmc/` 的第二批（1.39 起）；`mcmc/` 的回归工作流（2026-10 Gelman-Hill-Vehtari 一轮） |
| [`forest.md`](docs/dev/modules/forest.md) | 因果森林与 GRF 家族 | `forest/`；`forest/` 的 GRF 家族 |
| [`regression.md`](docs/dev/modules/regression.md) | 回归族（似然模型、GEE、GAM、选择、非参数树） | `regression/mprobit.py`；`nonparametric/conditional_tree.py`（`sp.ctree`，2026-10-12 起，来自 Varian 2014 的复现）；`selection/elastic_net.py`（`sp.glmnet`，1.39 起）；共线回归元（`core/_collinear.py`，2026-10 起）；`regression/gee.py`（`sp.gee`，1.39 起）；`regression/gam.py`（`sp.gam`，1.39 起）；教材驱动的小估计器 |
| [`iv.md`](docs/dev/modules/iv.md) | 工具变量与 MTE | `regression/iv.py`；`iv/mte.py` |
| [`formulas_covariates.md`](docs/dev/modules/formulas_covariates.md) | 公式写法、协变量与处理变量的输入约定 | 公式里的非标识符列名；公式的 R 写法（同一轮）；协变量列表里的分类变量（`core/_covariates.py`，2026-10-05 起）；二值处理的校验；公式里的 `I(x^2)` |
| [`treatment_effects.md`](docs/dev/modules/treatment_effects.md) | 处理效应（TMLE、IPW/AIPW、匹配、反驳检验） | `tmle/`（2026-10 Schuler-van der Laan 一轮）；`matching/full.py`（`sp.full_match` / `sp.match(method='full')`，1.39 起）；`robustness/refute.py`（`sp.refute`）；`matching/`；`sp.ipw(se_method='sandwich')`；`sp.aipw(trim=)`（1.39 起）；`matching/`；`matching/residual_balance.py`（`sp.residual_balance`） |
| [`dml.md`](docs/dev/modules/dml.md) | 双重机器学习 | `dml/`；`dml/did.py`（`sp.dml_did`，1.39 起） |
| [`causal_discovery_dag.md`](docs/dev/modules/causal_discovery_dag.md) | 因果发现与 DAG | `causal_discovery/mmhc.py`（`sp.mmpc` / `sp.mmhc`）；`causal_discovery/`；`dag/`（图这一层） |
| [`structural.md`](docs/dev/modules/structural.md) | 结构模型（BLP、路径分析、生产函数） | `structural/blp.py`（`sp.blp` 的随机系数部分，2026-10 按 BLP 1995 复现文章修订）；`structural/path_analysis.py`（`sp.path_analysis`）；`structural/production/op_lp_acf.py`（OP / LP / ACF）；`structural/blp.py`（`sp.blp` / `sp.blp_instruments`，2026-10-12 起含 Berry 1994） |
| [`stata_translation.md`](docs/dev/modules/stata_translation.md) | Stata 翻译（sp.from_stata / sp.stata） | `agent/_translation/`（`sp.from_stata` / `sp.stata`）；`sp.stata` 的数据管理 / 描述统计 / 抽样 / 编程层（2026-10 Kohler-Kreuter 一轮新增） |
| [`timeseries.md`](docs/dev/modules/timeseries.md) | 时间序列与预测 | `timeseries/`；`timeseries/dlm.py`（1.39 起）；`timeseries/` 预测工具箱（2026-10，来自 Hyndman FPP 审查）；`timeseries/` 的 2026-10 Neusser 一轮 |
| [`survival.md`](docs/dev/modules/survival.md) | 生存分析 | `survival/models.py` |
| [`experiments_interference.md`](docs/dev/modules/experiments_interference.md) | 试验设计、干扰、自适应试验、离策略评估 | `doe/`（试验设计，1.39 起，来自 Joseph 2025 教材）；`interference/network_exposure.py`（2026-10 Wager 教材第 12 章重建）；`ope/mdp.py`（1.39 起）；`experimental/adaptive.py` |
| [`agent_mcp.md`](docs/dev/modules/agent_mcp.md) | MCP 服务器与随包 skill | `agent/mcp_server.py`（`statspai-mcp`，2026-10-14 一轮）；随包 skill（`agent/_skill/`，`statspai skill install`） |
| [`evidence.md`](docs/dev/modules/evidence.md) | 证据映射（validation_scope） | `sp.validation_scope`（`validation_scope.py`）是按配置 × 输出的证据映射 |
| [`plots.md`](docs/dev/modules/plots.md) | 图形编辑器 | `plots/` 的图形编辑器（`sp.interactive` / `FigureEditor`，2026-10-16 重建） |

跨领域的通用约定留在这里：

- **2026-10 月度审查留下的约定**（第三轮选项扫描 + 对本月约 210 个新函数的边界探测；清单与未决项在 CHANGELOG 该节）：
  - **权重的读法**：`weights_as=`（`core/_agent_summary.resolve_weights_as`）加在 13 个回归入口上。`'sampling'` 只在用户没另选方差时换成稳健方差，与手动指定逐位一致；每个入口只接受自己原生的那一种基于模型的读法。新增回归入口接受 `weights=` 时接上它，不要另起参数名。
  - **预测时没见过的因子水平必须拒绝**：检查在 `core/results.py::_prediction_design`（`postestimation/_design.reject_unseen_levels`），它读 `data_info['factor_levels']`。参照水平没有哑变量，所以只能靠拟合时记录（`fitted_factor_levels`）。已接 regress / logit / probit / glm / poisson / nbreg；**新增支持 `predict(newdata)` 的公式型估计器要在拟合时写这一项**，否则拼错的水平静默得到参照水平的预测。
  - **整数时间有间隔**：`sun_abraham` / `stacked_did` 的参照期是相对时间 −1，间隔面板上它不存在；两者经 `calendar_time_aware(numeric_gaps=True)` 按观测期的秩编号。新增以"相对时间 −1"为参照的估计器照此办理，并加"间隔编码等于连续编码"的测试。
  - **默认先验是否够 vague 的事后检查只有一份**：`mcmc/_core.note_default_prior`，所有带固定默认先验的采样器都要调用（1.39 时只有 `bayes_regress` 调了，IV 系数被压到真值的六分之一而无告警）。告警不得消耗随机数。
  - **并列**：凡是"按得分排序后取前若干"或"取 k 近邻"的统计量，并列的单位必须一起进入（`cate_gain_curve`、`factor_importance`），否则结果随行序变。`sp.match(ties='all')` 的**标准误**（2026-10-15 起）同臂邻居也全取并列者，即 Stata `psmatch2 , ai(1) ties`：Lalonde 1178.209472851439、NSW 640.4053566440037，任意行序都逐位一致（`test_psm_ties_se_parity.py`；Stata 要 `set type double`，默认 float 存条件方差时差 1e-8）。`ties='first'` 仍是不带 `ties` 的 psmatch2（随行序变，Stata 本身也如此）。此前记的 1186.871 是 J/(J+1) 换成 K/(K+1) 的错误候选。
  - **`.loc[y_df.index]` 是 bug**：索引标签重复时（`pd.concat` 不带 `ignore_index`）按标签取行会把行数翻倍。按位置取行；这一轮在 gam / gee / threshold / series / xthtaylor / abnormal_returns / dlm / mcmc 各入口都修过一遍。
  - **`sp.ljungbox(lags ≤ model_df)` 的 p 值缺失是约定**（R `Box.test(fitdf=)`、statsmodels 相同），只告警、不拒绝。
  - **探测代理的简报里要写明**：zsh 下未加引号的 `$(grep -rl ...)` 会作为**一个**参数传给 pytest，跑 0 条测试而不报错；用 `files=("${(@f)$(...)}")`，并核对报告里的测试条数。
- **签名 house style 是 ratchet**：`scripts/signature_house_style.py --check`。新函数用规范名（`id` / `time` / `treat` / `covariates` / `weights` / `vce`），旧拼写用 `@accepts_aliases` 收。
- **`fast/` / `fixest/` / HDFE**：性能关键路径先 Rust，再 numba / JAX。

---

## 12. 不要做的事

- 不要悄悄改现有估计器的数值输出。必要的正确性修复 → CHANGELOG + MIGRATION 用 **⚠️ correctness fix** 标注。
- **不要凭记忆写引用**——任何 citation 都必须按 §10 核验四要素（作者 / 年份 / 标题 / DOI 或 arXiv ID），未经 Crossref 或 DOI 核验的引用不得进入 docstring / 文档 / `paper.bib` / commit message。捏造引用 = 数值正确性信任的直接破产。
- 不要把 `torch` / `jax` / `pymc` 塞进核心 `dependencies`——放 optional extras，惰性 import。
- 不要绕过 registry 添加对外函数。
- 不要在没对齐既有 dispatcher（`sp.synth` / `sp.decompose` / `sp.dml`）的情况下另起一个。
- 不要把教程 / 长文档写进代码注释——放 [`docs/guides/`](docs/guides/)。
- 不要 mock 估计器的数值路径。参考对齐测试必须跑真 R / Stata 输出或公开论文数字。
- 不要吞异常返回 `None` / `NaN`。
- 不要把凭据、token、内部数据提交到仓库或 memory。

---

## 13. 参考 Memory

`~/.claude/projects/-Users-brycewang-Documents-GitHub-StatsPAI/memory/`：

- `user_bryce.md` — 用户画像（计量经济学背景，期望精准技术语言）
- `project_statspai_vision.md` — P0–P3 路线图
- `feedback_sp_alias.md` — 始终 `import statspai as sp`
- `feedback_no_pr.md` — 直推 main
- `reference_pypi_publish.md` — 发布流程

外部指针：[GitHub](https://github.com/brycewang-stanford/StatsPAI) · [PyPI](https://pypi.org/project/StatsPAI/) · [CoPaper.AI](https://copaper.ai)。

---

## 14. 速查

```bash
pip install -e ".[dev]"                       # 开发安装
pytest                                        # 测试
black src tests && flake8 src tests && mypy src   # lint / format / type
mkdocs serve                                  # 文档预览
python -m build && twine check dist/*         # 打包
python benchmarks/run_all.py                  # 性能基准
python -c "import statspai as sp; print(len(sp.list_functions()))"   # registry 自检
python scripts/registry_stats.py                # canonical 数字（README/docs/stats.md 同步用）
python scripts/registry_stats.py --check        # CI 漂移检查（function/submodule 计数）
python scripts/registry_stats.py --table        # 重生 docs/stats.md 的按模块表
(cd rust/statspai_hdfe && maturin develop --release)   # Rust 后端
```

---

*最后更新：2026-10-14。过期信息会蔓延到每一次 agent 会话——持续维护本文件。*

## 其它关键事项
- **论文的版本锚定（2026-10-02 起，所有会话遵守）。** StatsPAI 一天可能发几个版本，论文不追版本。**只在三个时点改锚：投稿、返修、接收后定稿。** 平时发新版时论文一个字都不改，只回答"新版本还能不能复现论文的数字"：
  - DiD 论文：`python Paper-DiD-JAE/scripts/check_release_compat.py --statspai-root <检出> [--full]` 在临时副本里重跑 Python 端、与已提交归档逐行比对，结论记入 `Paper-DiD-JAE/parity/RELEASE_COMPAT.{json,md}`。"不复现"不是论文的错，是下次改锚时要并入的变化清单。
  - JSS 论文：等价的机制已经有了，就是下面的审稿期冻结（`tests/jss_review_freeze.json` + `docs/dev/jss_review_changes.md`）：冻结产物变了就逐条登记对论文的影响，改锚时统一并入。投稿时把冻结设为启用即可，不需要另写脚本。`Paper-JSS/replication/scripts/check_headline_numbers.py` 已改为调用 `scripts/trace_perf_path.py --check`（Paper-JSS `d3bb3f2`）：检出在所钉 tag 上时哈希结论有约束力；main 走在 tag 前面时只打 NOTE，仍由 tag-diff 规则裁决，因为论文描述的是所钉版本；v1.34.2 及更早的 tag 没有这份记录，沿用白名单。**改锚 JSS 的做法（2026-10-08 按此做过一遍）**：先在目标树上把冻结产物全部重推导（`verify_reproduce.py` 要带 `STATSPAI_DIDM_LIB=~/.statspai-rlib-didm014`，否则 81 号模块报 r_error；`verify_reproduce_stata.py`；`verify_reproduce_py.py`；`tests/coverage_monte_carlo/run_*.py` 与 `mechanisms/*.py`，约 80 分钟，SDID 一行占 40 分钟；森林种子研究的 `_generate_grf_seed_mc_py.py`），末位抖动的文件还原、真变了的才提交并记账；Track C 只在 `trace_perf_path.py --check` 报过期时重测（`run_when_idle.sh` 每步前等 5 分钟空闲，开着 VS Code 时约 80 分钟；期间不要跑任何占 CPU 的东西）。计时路径记录自 `7c3f6d3d` 起不含 `src/statspai/__init__.py`（它只在热身时跑一次惰性属性钩子），所以新增导出函数不再让 HDFE 计时过期。Paper-JSS 的审计脚本里有随包变化的硬编码要同步：`methodological_gap_ledger.py` 的 Basque 求解器诊断值、`data_provenance_audit.py` 的打包数据集名单、三个脚本里的公共数据集个数、`jss_formal_compliance_audit.py` 的 `PAGE_CEILING`。全新克隆里 `make submission-ready` 第一遍会因投稿包尚不存在而在审计处失败，先 `make submission-package` 再跑。跑这些重推导会在工作树里留下被忽略的产物（`tests/orig_parity/data/0[7-9]*.csv`、`1[01]*.csv`、`tests/stata_parity/.pdf` / `.png`、两个 `_repro_check/`），打包和跑数据溯源审计前要删掉。
  - Track C 计时绑定的是**代码**不是版本号：`tests/perf/results/_timed_path.json` 记录每个计时模块执行到的源文件哈希（版本行已屏蔽），`python scripts/trace_perf_path.py --check` 回答"已提交的计时对这棵树还有效吗"，过时会点名文件。文档版本、无关模块的改动都不需要重测；过时了才在安静的机器上重跑 `tests/perf/run_when_idle.sh`，并在**测量所用的那棵树**上重跑 `trace_perf_path.py`（它会拒绝在别的版本上追踪）。
  - **不要为了让论文的检查通过而发补丁版本**，也不要因为出了新版本就去改论文的锚点、表格或归档。论文周期内确实需要修包（如审稿人要求）时，开维护分支 `paper/<名>-X.Y.x` 只合入论文需要的修复并从该分支发补丁，main 照常前进。
  - 同一时间只能有一个会话改某篇论文的仓库；改锚前先 `git fetch` 看对方有没有在动（2026-10-01 两个会话各自改锚 JSS，互相覆盖过一次）。
- **JOSS 论文已发表（2026-09-03）**：Wang & Rozelle, *Journal of Open Source Software* 11(125), 10604, DOI `10.21105/joss.10604`（review issue：https://github.com/openjournals/joss-reviews/issues/10604，已 `accepted` + `published`）。审稿阶段的"不要影响审稿"约束**解除**；`paper.md` / `paper.bib` 现在是已发表版本的存档，**不要再改动其内容**（勘误走 JOSS 的 erratum 流程）。对外引用一律用 `sp.citation()` / `CITATION.cff` 的 `preferred-citation`（JOSS 文章），软件条目用 `sp.citation(which="software")`。
- **下一篇：JSS**（Journal of Statistical Software），核心是 **Stata / R 数值 parity**（`tests/reference_parity/`），拟邀 Yiqing Xu 合作（2026-09-04 已邮件征询 Scott 意见）。
- **JSS 审稿期冻结（投稿即生效，至编辑决定为止）。** **当前状态（2026-10-08）：未投稿，冻结已暂停**（manifest `"active": false`，锚点是 **1.39.1**；2026-10-08 从 1.34.2 改锚，改锚前把全部冻结产物在该树上重新推导了一遍：R 89/89、Stata 85/85、Python 侧 89/89、原始数据、森林种子研究、Track B 均复现，只有 CS-DiD 异质性稳健行是 1.31.0 之前留下的过期值（0.946，实际 0.954）；Track C 重测。JSS 稿件与复制包在 v1.39.1 上 `make submission-ready` 全绿，改动在 Paper-JSS 分支 `reanchor-1.39`；1.34.2 的归档 `Paper-JSS/submissions/2026-10-0{1,2}-*` 已被取代），计划约一周后投稿 JSS、一个月内上传 arXiv 预印本（这两个时间只记在这里和 submissions README，论文、预印本、投稿信都不写日期）；投稿时按实际提交的版本 `python scripts/jss_review_freeze.py --write --release X.Y.Z` 重新冻结，下面的规则届时才生效。 稿件锚定 `tests/jss_review_freeze.json` 里的 tag；该文件哈希了稿件表格读取的全部冻结产物（Track A / 原始数据 parity 结果、Track B 覆盖率与机制实验、森林种子研究、Track C 计时）。审稿期间**照常开发、照常发版**，但凡改动其中任何一个文件（修 bug 后重生成 parity、重跑计时、加新模块），都必须在 `docs/dev/jss_review_changes.md` 记一条：日期、commit、原因、**对论文的影响**（哪张表哪个数字从多少变成多少），路径用反引号列出——否则 `tests/test_jss_review_freeze.py` 红。**2026-09-28 起按提交核对**：自冻结 tag 以来每一个改动冻结产物的提交（含合并提交），都要在**同一条**记录里用反引号写出它的 sha（≥7 位）和路径；以前只要路径出现过一次就算登记，第二次改同一文件会蹭第一条记录，当天就查出两处漏记。提交不知道自己的 sha，所以记录放在紧随其后的提交里、一起推送；检查挂在 pre-push hook 和 CI `parity-guards` 的 `jss-review-freeze` job（浅克隆看不到历史，pytest 版会 skip，CI 那一步拉全量历史）。**不要为了表格好看重生成冻结产物；不要改稿件**——记录下来的改动在下一轮修改稿时统一并入，届时锚到新 release 并 `python scripts/jss_review_freeze.py --write --release X.Y.Z` 重新冻结。审稿期内 Paper-JSS 的脚本要对着冻结 tag 的 worktree 跑（`STATSPAI_ROOT=<tag worktree>`），对着 main 跑会因版本号不同而让 release-boundary 审计变红，那不是论文的问题。编辑决定后把 manifest 的 `"active"` 设为 `false`。
- **Zenodo 归档：只有 "Publish 一个 GitHub Release" 会触发。** commit / push / 打 tag / 发 PyPI 新版都**不触发**归档。Publish Release 会同时触发 GitHub↔Zenodo 自动归档（铸出**永久不可删的 version DOI**）和 `ci-cd.yml` 的 `publish-prod`（上传 PyPI），一次两个不可逆动作——发之前务必先手动跑一轮 `ci-cd.yml` 的 `test_scope=full`，并核对 `.zenodo.json` / `CITATION.cff` 的标题、作者、ORCID、单位、license 与 `paper.md` 逐项一致。
- **JOSS #10604 的归档已完成（2026-08-24）。** 时序上要注意：JOSS 要求**在 `recommend-accept` 之前**交出 archive DOI，不是接收之后——编辑 8/18 的 post-review checklist 明确索要 DOI 和版本号。本次交付的是 **v1.23.0**，version DOI `10.5281/zenodo.22085759`；concept DOI 仍是 `10.5281/zenodo.19933900`（永远指向最新归档）。此后再发 Release 会继续铸新的 version DOI，**不影响本次投稿**（JOSS 锚的是已提交的那一个）。
- **提交闸门（同 §9 顶部"提交闸门"，二者是同一条规则、最高优先级）**：2026-09-28 起常设授权，agent 判断合适（闸门全绿、只含自己的改动、改动完整）即可直接 commit + push 到 main；tag / PyPI / GitHub Release / force push 仍须当次明确授权。每段工作结束照常用中文总结并列出已推送的 commit。


- Please think in Egnlish but summarize the coversation and discuss with me in Chinese.
