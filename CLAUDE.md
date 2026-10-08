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
├── src/statspai/          # 主包：87 子模块 / 1,167 函数（实时数 `python scripts/registry_stats.py`）
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
- `References` 段只写 **bib key**（对应 [`paper.bib`](paper.bib)）或**经过核验的规范引用**——禁止在 docstring 里手写未核验的引用字符串，详见 §10。
- 示例一律 `import statspai as sp` + `sp.xxx`。
- 破坏性改动 → [`MIGRATION.md`](MIGRATION.md) + `DeprecationWarning`。

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
   版本号一改、tag 还没打，就进入**发版窗口**：`python scripts/release_gate.py --fix-census` 把 `docs/guides/stability.md` 与 `docs/jss_source_audit_dossier.md` 里的 registry 统计改成当前值，再 `python scripts/release_gate.py --check` 确认（pre-push hook `release-gate` 与 `tests/test_release_gate.py` 在窗口内强制统计一致，窗口外只查另外两项）。这三项是 1.33.0 / 1.34.0 漏掉、靠构建论文复制包才发现、各赔了一个补丁版本的问题：统计过时、正则被 ASCII 转写改了语义（模式里的非 ASCII 字符一律写 `\uXXXX`）、包内文本出现宣传措辞。
3. `pytest -q` 全绿 + `pytest tests/reference_parity/ -q` 必过。
4. `rm -rf dist/ && python -m build && twine check dist/*`。**再把 sdist 的文件清单对照 `git ls-files`**：`MANIFEST.in` 的 `recursive-include tests *` 会把工作树里被 gitignore 的本地产物一起打包——1.32.0 发版时先后抓到复现检查的临时目录、Track C 的 38 MB 共享输入、以及可由包内 `sp.datasets.nhefs()` 逐字节重生、因而不入库的 NHEFS 副本，sdist 一度从 25 MB 涨到 63 MB。新出现的 gitignore 产物目录要在 `MANIFEST.in` 里 `prune`。
5. 干净 venv 装 wheel 冒烟测试。
6. `git tag vX.Y.Z && git push && git push --tags`。
7. `twine upload dist/*`。

**默认不发 GitHub Release（2026-09-26 起）。** 发版 = TestPyPI + PyPI（本地 twine）+ 打 tag，到此为止；**不要**跑 `gh release create`，也不要在网页上 Publish Release。原因：Publish Release 会同时触发两个不可逆动作——Zenodo 铸出永久不可删的 version DOI，以及 `ci-cd.yml` 的 `release: published` 再上传一次 PyPI（与本地 twine 重复）。只推 tag 两者都不触发。只有用户在**当前会话**明确要求（如期刊需要 Zenodo version DOI）时才发，并先按下文「Zenodo 归档」一条做核对；发版总结里要写明「未创建 GitHub Release」。

---

## 8. 文档

- MkDocs（[`mkdocs.yml`](mkdocs.yml)），源在 [`docs/`](docs/)，`mkdocs serve` 本地预览。
- 当前 guide：`synth` / `choosing_did_estimator` / `choosing_iv_estimator` / `choosing_rd_estimator` / `choosing_matching_estimator` / `callaway_santanna` / `cs_report` / `honest_did` / `repeated_cross_sections` / `robustness_workflow` / `bayesian_econometrics` / `migration-from-r` / `mixtape_ch09_did` / `stock_watson_4e` / `design_based_econometrics` / `ding_first_course` / `xu_lan_causal_econometrics` / `facure_causal_inference_in_python` / `schuler_vanderlaan_modern_causal_inference` / `forecasting_fpp` / `time_series_econometrics` / `design_of_experiments` / `wager_causal_inference` / `regression_and_other_stories` / `yuksel_aydede_causal_ml`。相关估计器改动要同步对应 guide。
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
| **计数行** | `README.md`（中文，默认）、`README_EN.md`（英文，PyPI 长描述用它）、`docs/index.md`、`docs/reference/index.md` | 四处手写的「N 个注册函数」，由 `registry_stats.py --check` 门控。**对方加一个函数，你这四行同时作废** |
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

- **`rd/`**：kernel / 局部多项式 / sandwich 走 `rd/_core.py`，不要重新实现。
- **`rd/optimized.py`（`sp.rd_optimized`，Imbens-Wager 2019）**：权重必须**精确**满足四个矩条件（两侧各自和为 ±1、与 running variable 正交，`_project`），否则最坏偏差无界；报告的 `max_bias` 是对*实际使用的权重*按 `M·∫|G|` 精确积分得到的（`worst_case_bias`），不是优化问题里离散化后的那个数——对任何权重都成立，对局部线性权重退化成 RDHonest 的闭式。求解用「单位跳跃、曲率 ≤ kappa 的最不利函数」的有界最小二乘（scipy `lsq_linear(method='bvls')`），再对 kappa 做一维搜索；**不要**换成乘子形式上的 L-BFGS-B 或自写的投影牛顿：Hessian 是二次积分算子，条件数太差，两者都试过、都不收敛。与 `optrdd` 是同一规划的两种离散化，只能对到约 0.01 个标准误（估计）和 0.3%（偏差），按 S 记，不要写成 parity；optrdd 的权重对 running variable 的正交只到 5e-5。连续 running variable 下相对 `sp.rd_honest` 的增益只有千分之几（三角核本来就接近最优），不要在文档里夸大。`fuzzy=` 用结果方程选出的同一组权重，区间是 Anderson-Rubin 式的集合 `{t: |γ'(Y − tD)| ≤ cv·se(t)}`，偏差界为 `(M + |t|·M_fuzzy)·∫|G|`；弱第一阶段时区间无界是正确行为，不要改成 delta 法区间。完全依从时必须精确退化为 sharp 结果（测试守着）。二维 running variable 没做：需要一般线性不等式约束的 QP，scipy 没有合适的求解器。
- **`synth/`**：20+ 估计器全部经 `sp.synth(method=...)` 分发。新增方法要同时加到 dispatcher 和 `synth_compare()`。`method='classic'` 的 `ci` 是与安慰剂秩检验 p 值对偶的常数效应反演区间（`_core.placebo_inversion_ci`，Firpo-Possebom 2018；供体少于 `1/alpha - 1` 个时为 `(-inf, inf)`），旧的 `estimate ± z·sd(placebo ATT)` 只留在 `model_info['ci_normal']`——**不要**再让 p 值和区间出自两套程序（2026-10 Gaillac-L'Hour 教材第 10 章查出）。
- **`decomposition/`**：影响函数 / statistic-value / WLS 在 `_common.py`。RIF / FFL / inequality / Oaxaca 都委托到该文件。
- **`multilevel/` / `frontier/` / GLMM**：v0.9.3–v0.9.4 有含正确性修复的大重构——用户引用旧数值时主动提示。
- **`bayes/`**：默认 NUTS (`draws=2000 tune=1000 chains=4 target_accept=0.9`)；必带 `rhat` / `ess_bulk` / `ess_tail` / `divergences`；`rhat > 1.01` 或 `ess < 400` 发 `ConvergenceWarning`；HDI 94%（arviz 约定）。
- **`mcmc/`（1.39 起）**：不依赖 PyMC 的贝叶斯计量——`sp.bayes_regress(model=...)`（dispatcher，10 个似然）、`sp.bayes_mixed`、`sp.bayes_ivreg`（单内生变量的 Gibbs IV）、`sp.bma`、`sp.bayes_factor` / `sp.savage_dickey`、`sp.bayes_bootstrap`、以及对任意链可用的诊断（`sp.geweke_diag` 等，与 R `coda` 对到 1e-9）。新增似然按 `_models.py` 的模式加一个类（`names` / `sample` / `to_u` / `log_kernel`），**不要**另起入口；众数、Laplace、Gelfand–Dey 都只依赖 `log_kernel`。MCMC 没有跨语言逐位 parity：**每个采样器的证据是小模型的精确后验**（`tests/reference_parity/test_bayes_regress_exact_posterior.py` 用 `scipy.stats` 独立写被积函数做网格积分，均值 4 个 MC 标准误内、sd 5% 内、边际似然对归一化常数），新增采样器必须加一条同类测试；与 MCMCpack / bayesm 的长链对比只算 S。先验约定跟教材与 MCMCpack：`sigma2 ~ IG(alpha0/2, delta0/2)`。默认先验方差 1000 只在回归元尺度适中时才算无信息，拟合后会检查并告警，不要去掉这个检查。协方差矩阵的逆 Wishart 先验**不用单位阵尺度**（教材 / MCMCpack / bayesm 的做法在对数因变量面板上让方差分量偏大 14 倍）：默认只给平方和加 0.02，并报告先验占比（`re_prior_share`），超过 25% 告警。**PyMC 那边的教训**（`bayes_iv` / `bayes_hte_iv` 1.39 前）：把第一阶段 OLS 残差当数据代入结构方程，后验均值是 2SLS，但后验 sd 偏小 `sqrt(1 - corr^2)`，强内生时 95% 区间覆盖率只有一半；任何“两步”贝叶斯模型都要问第一步的不确定性进没进后验。本机 pytensor 的 C 编译会挂（`ld: library 'd64' not found`），用 `PYTENSOR_FLAGS="cxx=,mode=NUMBA"` 跑 PyMC 测试。`bayes_synth` 的效应后验必须含被处理单位处理后自身噪声 `N(0, σ²/T1)`（1.39 前只有权重不确定性，覆盖率 83%）。`bayes_mte` 的 `polynomial` 模式（1.39 起）是局部 IV：`E[Y|p] = α + Σ b_k ∫₀ᵖ a(u)^k du`（`integrated_mte_powers`，不含 D）；旧式 `Y = α + D·g(p)` 在未处理结果也被选择时不是 MTE（真斜率 −0.8 估成 −0.14），不要改回去。证据是不需要 PyMC 的 `tests/reference_parity/test_mte_local_iv_identity.py`。
- **`forest/`**：`sp.causal_forest` 默认是自研 GRF 引擎（`_grf_engine.py`，numba），拟合流程在 `_grf_fit.py` / `_panel_forest.py`，推断在 `_grf_inference.py`。训练行上的一切统计量（ATE、校准、RATE、BLP）必须用 **OOB** 预测，不得用 `effect(X_train)`。grf 是 GPL-3：只按论文与文档行为独立实现，**不得**移植其源码；与 grf 的森林对比只能是 T3，推断算子（给定森林）才可做到 1e-14 级对齐。`average_treatment_effect(subset=)` 是 grf 的同名参数（给定森林输出时四个 target 对到 1e-14），但**用森林自己的 OOB 预测划分子集再比较两半不是检验**，文档里要指向 `sp.rate_split`。面板重复观测必须 `clusters=`；`fe=` 森林没有倾向得分，不做 AIPW，改用**插补分数**（`_fe_imputation.py`：未处理格拟合单位+期效应 [+时变 controls]，处理格 `Y - α̂ - γ̂`）给出 ATT / BLP / 校准 / 分组效应（`sp.forest_group_effects`），ATT 与 `sp.did_imputation` 逐位一致；`target_sample='all'` 仍拒绝。**`sp.rate` 现在支持 `fe=` 森林**（1.30.0）：秩权重与插补权重都是线性的，复合后 RATE 就是又一个 `v'y`，方差沿用同一套精确线性权重（`variance` 在此默认 `'bjs'`——`'forest'` 标定的是*本样本*的 RATE，对总体 RATE 只覆盖 90.8%）。但**用森林自己的 OOB 排序去评自己不是检验**：插补分数带着 `-gamma_hat_t`，与森林见过的期共用数据，零异质性下 AUTOC 均值 −0.025、5% 检验拒绝 17.5%；用 `sp.rate_split`（按单位/dyadic member 分裂，两侧各自拟合）回到 +0.0008 / 7.5%，功效 99.5%。两侧少于 ~30 个单位时 `rate_split` 会告警——那个规模上分裂测的是它自己。**`sp.forest_policy_tree`（1.31.0）**给出规则本身：在插补分数上长策略树（深度 ≤2 精确搜索，复用 `policy_learning` 的 `exact_policy_tree` / `PolicyTree`，不另起一套），同样**强制**按单位分裂拟合与定价——零异质性且 `cost` 等于常数效应时真实增益恒为 0，同样本版均值 +0.048（自身 se 0.037）、13.0% 声称显著，分裂版 +0.005 / 3.5%，有真异质时分裂版 0.3189 对 oracle 0.3191、功效 99.5%。`value` / `value_treat_all` / `gain_over_treat_all` 是同一 design 的三个线性泛函，gain 的点估计恰为差值，但**方差不是两者之差**——BJS 块中心化按各泛函自己的 v² 加权。**两个函数都默认 `n_splits=21`，按 CDDF (2025) 的 VEIN 聚合**（中位数点估计 / 条件区间按 1−α/2 构造后取中位数 / p 值取中位数再乘 2）——单次分裂只是一个抽样，而 `random_state` 又在用户手里，正是 CDDF 指出会让推断失效的做法。无法插补的分裂按「不可容许的划分」跳过并计数（小面板上 21 次里必有），方差非正的分裂贡献点估计但不贡献区间（用 nanmedian，否则一个 NaN 污染整个区间）。规则本身无法取中位数：报中位增益那次分裂的树，并给 `split_stability`——**根分裂变量在各次分裂间跳动时，再窄的价值区间也不构成对那条规则的证据**。`calibrate_cate` 默认用插补回归（`method='within'` 是 1.29.0 的旧回归，其斜率不是去衰减因子）。`split_rule="legacy"` 仅供复现旧数字，已弃用。
- **`forest/` 的 GRF 家族**（`iv_forest` / `multi_arm_forest` / `lm_forest` / `causal_survival_forest` / `regression_forest` / `multi_regression_forest` / `probability_forest` / `quantile_forest` / `survival_forest` + `variable_importance` / `best_linear_projection` / `get_scores`）：全部是引擎的 tree *kind*（`_grf_ext.py`：relabel+分裂 / 叶统计 / 局部求解），共用 `_grf_family.py` 的输入解析、`ForestOptions`、OOB 冗余参数、得分推断。**新增家族成员也按这个模式加 kind，不要另起 sklearn 森林**——之前三个"森林"就是 sklearn 替身（IV 用 Y 训练邻域森林 + 全局 Wald 比；multi-arm 样本内 AIPW；CSF 的 CATE 来自可以不在 W 上分裂的 RF），名不副实，1.31 起重建并记 ⚠️。约定：grf 的 `alpha` 叫 **`split_alpha`**，`alpha` 永远是显著性水平；**每个辅助森林走独立随机流**（`_grf_family.with_stream`），引擎的组种子来自 `SeedSequence(seed)`——两者缺一，同种子森林会抽到相同子样本（IV 的 CATE RMSE 因此比 grf 高 4.6%），相邻 `random_state` 会共享几乎全部树。证据：森林之后的一切算子（局部解、DR 得分、ATE/SE、BLP、KM/NA、加权分位数、变量重要性）喂 grf 自己的权重对到 ≤7e-14（`test_grf_family_operator_parity.py`，T2）；CSF 的"冗余参数→得分"映射按 grf 的离散化（删失积分对 `c_k <= min(U,h)` 求和，`[log S^C(c_{k-1}) - log S^C(c_k)]/S^C(c_k)`，分母用总体值 `(W-e)^2`）对到 1.2e-15（`test_csf_psi_operator_parity.py`，grf 内部 `compute_psi` 只作黑盒比输出）；森林本身只有随机筛查 S（`test_grf_family_statistical_parity.py`，三个 grf 种子 + 比值门限，不是 TOST 等价检验）。CSF 的冗余参数生存森林**必须用完整时间网格**：截到 horizon 会把之后的事件舍入到网格末点当作 horizon 前失败（生存概率目标偏差 0.023，4 个 MC se）。
- **`gmm/_dynpanel/`**（`sp.xtabond` / `sp.xtdpdsys`）：求逆一律走 `_estimate.safe_inv`，不要直接 `np.linalg.inv`——`inv` 只在主元恰为 0 时报错，数值奇异的权重矩阵会原样返回量级 1e15 的噪声（Hansen 表 17.3 的两步权重秩 191/199，修复前标准误是 1e14 且无告警）。奇异时用的广义逆是 `sweep_ginv`（每步取剩余对角最大者为主元、共线矩条件置零），即 Mata 的 `invsym`；两步估计量在权重奇异时**依赖广义逆的取法**，Moore-Penrose 给出的是另一组数，换掉它就对不上 `xtdpd` / `xtabond2`。Windmeijer 校正里的一步协方差要去掉被置零的矩条件（`_fit.py` 的 `V1_step`）。`time_dummies=True` 的取舍规则跟 Stata `xtdpd`：没有变换方程行的期不建虚拟变量；有常数项时再从回归元里去掉最后一个、但保留在工具变量里。证据在 `test_hansen_methods_stata_parity.py`（abdata 上一步 / 两步 / 奇异权重三组，1e-9）。Stata 报的 instrument 数是秩，我们报列数。
- **`regression/mprobit.py`**（`sp.mprobit`）：选择概率是按 GHK 的逐维条件形式写出、再用 Gauss-Legendre 求积积分的（备选项 ≤ 4），**不是模拟**。求积节点做了 `u = s^3(10 - 15s + 6s^2)` 的变量代换，不要拿掉：被积函数经过正态分位数函数，在单位区间两端导数无界，朴素的 Gauss 规则只有 O(m^-2) 的收敛（24 个节点误差 2e-4，代换后 5e-8），对数似然会差 0.1。导数是先对效用求差分再乘设计矩阵，不要退回对每个系数做差分（慢 4 倍以上）。归一化与 Stata `cmmprobit` 相同：与基准相减后的误差协方差，scale 备选项的方差为 2。证据：对 Stata `mprobit`（同样是求积）T2；对 `cmmprobit`（模拟）只能到 3–4 位，回放里按「系数在 Stata 标准误的 2% 以内」判定，不要写成严格对齐。
- **`tmle/`（2026-10 Schuler-van der Laan 一轮）**：`sp.tmle` 的 `estimand` 必须校验（1.39 前除 `'ATE'` 以外的任何字符串都静默返回 ATT，`'ate'` 也是）。三条约定：(1) EY1 / EY0 / RR / OR 只在 `fluctuation='per_arm'` 下才是 TMLE（单个 clever covariate 只解差值那一个方程），所以这四个 estimand 强制 per-arm，`result.detail` 也只在 per-arm 时给出；RR / OR 的区间在对数尺度上构造，`se` 是自然尺度的 delta 法。(2) ATT 只扰动 Q、g 不动，写成估计方程的形式，但因为扰动恰好解了 `sum H (Y - Q*) = 0`，它**就是**处理组上 `Q*(1,W) - Q*(0,W)` 的 plug-in 均值，EIF 均值恰为零（`test_r2_teffects_parity.py` 的恒等式）；ATC 是同一计算把两臂对调。**不要因为"它不更新 g"就以为它不是 TMLE 而去改默认算法**（这一轮中途这么做过，查到上面那条恒等式后撤回，ATT 的数值没有动）。R `tmle` 的 ATT / ATC 是另一种同样合法的 TMLE（小步长路径同时更新 g、先丢掉 `g < min(g | A=1)` 的对照、按似然停止），两者只差零点零几个标准误，不是 parity 行。(3) 每个结果的 `model_info['influence_function']` 均值必须为零，这是 `tests/test_tmle_targeting_properties.py` 对所有 estimand 的断言，新增 estimand 要过这条和"饱和模型下等于分层估计量"那条。与 R `tmle` 2.1.1：均值、差、RR、OR 在共享初始拟合下对到 1e-11（含权重与聚类）；带权重时它的 log OR 影响曲线没有中心化（方差偏大 0.3%–2%，可由我们的量加回 `w(1/(1-EY1) - 1/(1-EY0))` 重建到 1e-9），那是参考实现的问题，不要去对。弱重叠下（9% 的真倾向得分在 [0.025, 0.975] 之外）点估计无偏，是影响函数标准误偏小 15%–21%，ATT 的 95% 区间只覆盖 81%；**换截断界没有用**（试过 1e-6 到 0.1，0.74–0.87），交叉拟合到 0.84，整体重拟合的 bootstrap 到 0.90，所以补救办法是 `se_method='bootstrap'`（`tmle/_bootstrap.py`，按行或按簇重抽、每次连 Super Learner 一起重拟合，不接受外部给定的 Q / g1W；与 `fold_indices` 同用时同一行的各个副本留在该行原来的折里），不要去加自适应截断。**也不要加"把倾向得分拟合在结果模型预测值 (Q0, Q1) 上"这条捷径**：结果模型正确时它能把标准差降 17%–21%，结果模型漏掉一个混杂时偏差 1.47、覆盖率 0（双稳健性整个丢掉）；正规的 C-TMLE 是 `sp.ctmle`（`tmle/collaborative.py`，按 van der Laan-Gruber 2010 独立实现，R `ctmle` 是 GPL、只当黑盒）：贪心与交叉验证的准则都是 `RSS + var(IC)`（CV 里再加 `n·bias²`）——**贪心那一步也必须带方差惩罚**，只用 RSS 时进入顺序在损失平坦处与 R 不同；带上之后 6 个夹具的进入顺序与每一步的估计量对到 3e-9（T2）。CV 选第几步无法逐行对照，原因已定位：`ctmleDiscrete` **完全忽略 `folds` 参数**（固定种子时传什么划分结果都一样），所以只能比分布——60 个随机划分上各步的 CV-RSS 均值一致，200 个数据集上偏差 / 标准差 / 平均选中步数一致，记为 S。R 默认的 CV 准则只有 RSS（它报告的 `penlikelihood` 里没有惩罚项），对应 `penalty='search'`；我们默认 `'variance+bias'`，在正确设定下标准差 0.093 对 0.103。`order=` 是预排序（scalable）变体，6 条序列对到 8e-10。**重启规则**：一步必须同时降低带惩罚的准则**和**裸 RSS，任一不降就从当前靶向拟合重启（只看准则时，方差略小而 RSS 略大的候选会被误判为改进，两条序列因此差 5e-6；是穷举 8 种重启模式定位出来的，不要改回只看准则）。模块叫 `tmle/collaborative.py`，**不要改回 `ctmle.py`**（与函数同名后测试取不到私有部分）。三条不要动的结论：(1) 影响函数标准误偏小 20%–40%（R 也一样），推断用 `se_method='bootstrap'`；(2) **只做 ATE**，协同式 ATT 试过，结果模型漏掉效应异质性时偏 −0.17，已撤回，不要加回来；(3) 序列上的 RSS 不单调（扰动最大化的是 Bernoulli 似然），不要写成单调。`sp.aipw(outcome_model='logit' | 'probit' | 'poisson')` 与 Stata `teffects aipw (y x, logit | probit | poisson)` 对到 5e-9（堆叠三明治里每一臂用三样东西：得分乘子 `s`、Hessian 权重 `h`、`v = dmu/deta`；logit / poisson 是典则联结，`s = Y - mu`、`h = v`；probit 用它自己的得分 `lam = (Y - mu) phi / (mu (1 - mu))` 和**观测** Hessian `lam (lam + eta)`，换成期望 Hessian 就对不上 Stata）。已知的处理概率走 `sp.tmle(g1W=标量)` / `sp.aipw(propensity=)` / `sp.ipw(propensity=)`；`sp.dml(model='irm')` 有意没加。给 `tmle` 维护者的说明在 `docs/dev/2026-10-07-tmle-weighted-odds-ratio-note-draft.md`，2026-10-07 已由 Bryce 邮件发出，尚无回复；对方若修了，重生 `tmle_parameters_R.json` 后要删掉“带权 OR 方差不一致”那条测试的豁免。`sp.causal_gap` 是书里第 2.3 节的 critical causal gap（`diagnostics/causal_gap.py`），只做区间端点到 null 的距离，不要往里加任何混杂模型。
- **`dml/`**：`sp.dml(model=...)` 是截面 DML 的 dispatcher（plr / irm / pliv / iivm）；面板另开两个函数，不进 dispatcher（签名要 unit/time）。`sp.dml_panel` 是带单位 FE 的静态 PLR；**`sp.dynamic_dml`（1.31.0）是时变处理的序列效应**（Lewis & Syrgkanis 2021）——处理会改变后续状态时，静态估计量不是低效而是错的。三角矩条件按**堆叠 GMM** 一次解出，点估计与倒推一致但附带跨期联合协方差（总效应 se 0.032 对独立相加 0.062）。给定相同的折与一阶段学习器，与 `econml.panel.dml.DynamicDML`（方法作者写的参考实现）在点估计**和** SE 上都对到 1e-15，是 T2。`lags=1` 是默认且不可省：状态里没有处理史时估计量不是变噪声而是**自信地错**（三期偏 −15% / −15% / +37%，无一覆盖真值）。
- **`dml/did.py`（`sp.dml_did`，1.39 起）**：两期 DML-DiD，面板（ΔY 或长表 `time=` + `id=`）与重复截面两种布局，不进 `sp.dml` dispatcher（签名不同）。canonical 参考是 Python `DoubleML` 的 `DoubleMLDID` / `DoubleMLDIDCS`：同折、同学习器下点估计与 SE 对到 1e-15（`test_dml_did_doubleml_parity.py`，夹具只收确定性学习器，森林版依赖 sklearn 版本）。处理组份额用**全样本**均值（DoubleML 的做法）；教材和 Chang (2020) 按训练折估，`in_sample_normalization=False` 时份额在估计量里约掉。倾向得分是截断不是删行，截断数进 `model_info` 并告警。**已知真值测试不要用随机森林当学习器**：二次趋势设计上 n=600 时森林的冗余参数偏差让估计停在 2.69（真值 2），n=20,000 仍有 2.12，这是学习器的收敛速度，不是估计量的 bug。
- **`matching/full.py`（`sp.full_match` / `sp.match(method='full')`，1.39 起）**：最优全匹配按「二部图最小费用边覆盖 = 约化收益 `min_i + min_j - c_ij` 上的最大权匹配 + 未覆盖点各取最便宜的边」精确求解，一次 `linear_sum_assignment`，**不要**换成网络流库或启发式。零距离并列时要删掉两端都已被覆盖的边（否则连通分量不是星形）。证据是小问题上穷举全部边覆盖；optmatch（GPL，黑盒）把距离按容差取整，默认容差下总距离比我们大，收紧到 1e-9 后总距离一致但匹配集合仍不唯一，估计值差 1e-3——**只能写「总距离不高于 optmatch」，不能写估计值与 MatchIt 对齐**。给定匹配集合时估计量与按集合聚类的 SE（`matched_set_effect`）等于 `lm(weights=)` + `sandwich::vcovCL`，1e-9。
- **`synth/_gsynth_multi.py`（`sp.gsynth(treat=)`，1.39 起）**：多处理单位、交错处理、带协变量的广义合成控制（Xu 2017）。对照组上的 IFE 是「β 的最小二乘」与「残差矩阵双向去均值后截断到秩 r」的交替最小二乘；固定 r 时与 R `gsynth` 对到 1e-12（r = 0..3、有无协变量、交错）。`covariates=` 的调用（含单处理单位）一律走这条路径：旧的 `_partial_out_covariates` 是不含截距和固定效应的混合 OLS，书中数据上 6.68 对 5.70，已在 1.39 记 ⚠️。因子数用论文的留一处理前期准则；`gsynth` 1.4 转调 `fect`，交叉验证方案不同，**不要声称选 r 的规则与 R 一致**。无协变量的单处理单位路径（Track A 19 号模块）没有动，它的选 r 规则是随机遮盖对照矩阵的单元格，与两者都不同，改它会动冻结产物。gsynth / fect 是 MIT。
- **`did/did_forest.py`**：每个 (g, t) 一片森林，干净对照组规则与 CS 相同；聚合 SE 由单元级影响函数跨格求和。
- **`causal_discovery/mmhc.py`（`sp.mmpc` / `sp.mmhc`）**：局部搜索（MMPC、半交错 HITON-PC）加受限爬山。bnlearn 是 GPL，只按论文独立实现、黑盒比输出。骨架是边集，可以也应当精确对照（24 个数据集全等）；混合搜索两边都停在局部最优，测试只断言 BIC 不低于 bnlearn，**不要**写成 parity。
- **`structural/path_analysis.py`（`sp.path_analysis`）**：解析器在 `path_analysis.py`，估计只有一个引擎 `_sem_engine.py`（观测变量路径模型、潜变量 `=~`、均值结构、增长曲线都是同一个 `v = alpha + B v + zeta`，载荷就是 `B` 的一项）——新增模型类型往引擎里加参数种类，不要另起一套似然。参考是 lavaan；潜变量模型的容差 5e-6 是 lavaan `nlminb` 的停止准则，不是方法差异。**`auto_cov_y=False` 是有意的默认**：`lavaan::sem` 会自动让终端结果变量的扰动相关，Stata `sem` 和这里都不会，对照 lavaan 时两个以上终端结果要传 `auto_cov_y=True`。`missing='fiml'` 按缺失模式分组求似然，标准误用观测信息（对梯度做数值差分），饱和模型矩用 EM；外生变量缺失的行仍然删除。未实现：多组、分类指标。
- **`causal_discovery/`**：`pc.py::stable_skeleton` 是 PC 与 FCI 共用的骨架搜索，按 PC-stable 写（层内邻接集冻结；条件集取自 `adj(x)\{y}`，再取自 `adj(y)\{x}`；有序对与子集顺序同 `pcalg::skeleton`，所以分离集也一致）。不变式：**骨架里的每条边都必须出现在 CPDAG 里**。两个对撞结构争同一条边时按节点顺序先到先得，冲突写进 `orientation_conflicts`，不许静默删边（旧实现 200 次抽样里 85 次丢边）。`collider_conflict='last'` 是 pcalg 的规则（后到覆盖），在 `test_pc_pcalg_parity.py` 的 24 个小样本参考图上与 `pcalg::pc` 逐边一致；默认 `'first'` 只在报告了冲突的 14 例上与 pcalg 不同。两种规则没有哪个更对，改默认前先看这份 fixture。`sp.fci` 是完整 FCI（`_fci_core.py`：Possible-D-SEP 二次删边 + Zhang R1–R10），36 个参考数据集上与 `pcalg::fci` 逐标记一致（`test_fci_pcalg_parity.py`）；标记矩阵 `P[i, j]` 是边 i–j 在 j 端的标记，与 pcalg 的 `amat` 同约定。`sp.hill_climb` 的 BIC 与 bnlearn 一致到 1e-12，搜索路径不必与 bnlearn 相同（都是局部最优）。新增发现算法复用这份骨架搜索，不要另写一份带提前终止的循环（旧 FCI 在"某层没删边"时就停，三个共因的结构永远留一条虚假边）。
- **`dag/`（图这一层）**：`sp.dag` 的解析器在 `graph.py::_parse`，**读不懂的语句必须抛异常，不许跳过**（2026-10 前 `X <- Z` 整条被丢，混杂边消失，空集被当成合法调整集）。`adjustment_sets` 先用祖先集判定"是否存在"（任何规模都精确），再枚举 ≤6 个变量的集合，更大的由祖先集逐个剪枝得到；**不要**再用"枚举到上限没找到"推出"不存在"。路径分三类：`causal` / `backdoor`（从箭头进入处理变量）/ `noncausal`（从处理变量出发、途中遇到对撞点），`confounder` 只指无对撞点的后门路径上的非对撞点。潜变量只认 `latent=` 声明和 `<->`，**不认名字前缀**。分类数据的条件独立检验只有一份实现 `dag/_ci_tests.py`（各层卡方 / G 统计量求和，自由度按层内实际出现的水平数，等于 dagitty `cis.chisq` 与 bnlearn `x2-adf` / `mi-adf`），`DAG.test_implications` 与 `sp.pc_algorithm` 共用。`sp.bayes_net` 是离散因果贝叶斯网（变量消元精确推断、`do` 即截断分解）；**条件概率表决定不了反事实**，`counterfactual` 只在非根节点全为确定性函数时放行。有潜变量时走 `sp.identify(...).estimate(data)`：两者遇到"答案依赖于数据里不存在的父配置 / 条件事件"都要报正性失败，前者警告，后者把受影响的处理取值置缺失。`sp.identify` 输出的估计式会按 d-分离化简条件集，表达式树的叶子永远是观测分布的边缘或条件概率，所以化简在完整图上做。`sp.identify_counterfactual`（`counterfactual_id.py`）是按论文独立写的 ID* / IDC*；R 的 `cfid` 是 GPL，**只当黑盒比输出，不读源码**，CRAN 的 0.1.8 把必要性概率判成 0（错），GitHub 上的 0.1.9 已修；与 0.1.9 在 1,118 个随机查询上 95% 一致，剩下 55 个分歧未裁决，其中 38 个是它判可识别而我们不可识别，可能是我们不完备。**对照外部包前先看它的开发版**。改这个文件后必须重跑 `tests/test_counterfactual_identification.py` 里的真值核验（随机结构模型上枚举外生变量），论文图里没写明、靠它查出来的四处细节记在评审文档第二轮。
- **公式里的非标识符列名**：反引号 `` `a b` `` 在 `core/utils.py::r_formula_idioms` 里统一翻译成 `Q("a b")`（`backticks_to_q`），所有走 patsy 的入口自动支持；只读列名的解析器（IV、计数模型、固定效应、gam 的平滑项）用 `unquote_name` 取出列名。**新增公式入口时调用 `r_formula_idioms`，不要另认反引号**；对公式字符串做文本替换（如把 `- 1` 当去截距）必须避开引号内的内容，`sp.iv` 曾把 `Q("x-1")` 改成 `Q("x+ -1")`。
- **`robustness/refute.py`（`sp.refute`）**：估计器无关的反驳检验，按 `estimator(data, y=, treat=, covariates=, **kw)` 调用。p 值问的是"重跑结果是否以应有的值为中心"，不是"是否接近"；`n_simulations < 2/alpha - 1` 时永远拒绝不了，直接报错。打乱结果变量会把混杂一起打掉，是弱检验；能抓出漏调整的是 `outcome_function=`（保留混杂的哑结果）。它们都查不出未观测混杂，文档里必须指向 `sp.sensemakr` / `sp.evalue`。
- **`matching/`**：`sp.match` 有放回匹配默认 `ties='all'`（1.39 起；Stata `teffects`、R `Matching::Match` 的做法，估计不依赖行序，Lalonde 上与 `teffects psmatch` 逐位一致）。`ties='first'` 是 `psmatch2` 不带 `ties` 选项时的规则，`sp.psmatch2` 显式传它；新写的内部调用要想清楚自己要哪一个，不要依赖默认。
- **`callaway_santanna(notyet_cutoff=)`**：默认 `'period'` 跟 R `did`（`G > max(t, base) + anticipation`）；`'asinr'` 是 Stata `csdid, asinr`（`G > t`），`'cohort'` 是 csdid 默认。三者只在 universal 基期的前期格或 `anticipation > 0` 时分歧，不要再把 asinr 写成"R 约定"。
- **`did/es_inference.py`**：`sp.event_study_vcov` 是**所有**事件研究估计量联合协方差的单一入口（CS/aggte、event_study、SA、Gardner、BJS、stacked、LP-DiD、dCDH、ETWFE），`sp.uniform_bands` 在其上做 sup-t 同时置信带，`honest_did` 的 FLCI 也走它。新增事件研究估计器时把联合协方差放进 `model_info['event_study_vcov']`（DataFrame，index/columns = 相对时间；若前后期分属不同回归则设 `attrs['block_diagonal']=True`），抽取器会自动接上。**不要**再各自重建协方差：main 上曾用固定份额重建，非对角块与 R `did` 差 8%，FLCI 因此偏窄。
- **`did/few_treated.py`**：处理组极少时（1 个或几个簇）簇稳健 SE 会大幅过度拒绝（实测 30 簇 AR(1) 设计上名义 5% 实际 74%）。`sp.did_few_treated` 用对照组构造安慰剂分布并反演，`method='ferman_pinto'` 额外按 `Var(W)=A+B/M` 校正组规模异方差（拟合为负时退 NNLS 并警告）。点估计仍是 TWFE 系数，**不声称一致**。
- **dCDH 家族（`did/did_multiplegt.py` / `did_multiplegt_dyn.py` / `twowayfeweights.py`）**：处理变量可以是离散非二值的（报纸数、税率档），三者都按**期初处理水平**匹配对照。`sp.did_multiplegt` 的 DID_M 是每单位处理的效应：每个切换者带自己变化的符号，除以总绝对变化量（2026-10 前把基线非 0 的切换者一律当 switch-off、不除变化量，在 Gentzkow 报纸面板上给出 −0.00082，`did_multiplegt_old` 是 0.0057791）。`sp.did_multiplegt_dyn` 内部把时期换成**秩**（`_tidx`，四年一期的面板等同年度面板）、丢掉已同时高于和低于期初水平的 (g,t)（Design Restriction 2）、事件按 (F, 方向, 期初水平) 划分、切换者方差单元按 (期初水平, F, F 期处理值) 中心化；`se_method='analytic'` 时联合检验直接用解析协方差做 Wald，不再跑 bootstrap；`aggregation='switchers'` 是 `Av_tot_eff`（Σ N_l δ_l / Σ N_l δ^D_l）；`same_switchers` 要求每个 horizon 的效应都**可估计**（有对照），不只是被观测到。**与 Stata 的一处有意不同**：非平衡面板上参考实现会因中间某期缺对照而把一个尚未切换的组整个删掉（它切换后的处理均值恰好回到期初水平时），我们保留它作对照；把该组从输入里删掉即可逐位复现 Stata（模块 docstring 与 `docs/dev/2026-10-05-dcdh-did-textbook-review.md`），不要为了对上而复制这个行为。`controls=` 的回归按期初水平（及 `trends_nonparam` 单元）在尚未切换的 (g,t) 上加权拟合，调整后的结果变量以**水平**形式写出（`Y − Xθ_d − Σλ`，面板有缺口也成立），解析方差带斜率估计项（`_residualise_on_controls` 返回的 `b`，事件上累加的 `m_x`），六种设定与 Stata 对到 5e-7。`placebo_sign='r'` 才是不带 `robust_dynamic` 的 `did_multiplegt_old` 的符号，默认 `'stata'` 是带 `robust_dynamic` 的。证据在 `tests/reference_parity/test_dcdh_textbook_stata_parity.py`（合成面板 + 162 个 Stata 数）。
- **`selection/elastic_net.py`（`sp.glmnet`，1.39 起）**：glmnet 口径的 lasso / ridge / elastic net（gaussian、binomial），与 R `glmnet` 4.1-10 的路径对到 1e-10、系数 1e-6、`lambda.min` / `lambda.1se` 同一格点。glmnet 是 GPL，**只当黑盒，不读源码**；文档没写明、靠输出反推并各有一条测试的四个约定，改之前先看 `test_glmnet_r_parity.py`：(1) gaussian 的 y 先标准化到单位方差，所以原尺度下岭惩罚是 `lambda / sd(y)`；(2) 自动路径的首点是无穷大惩罚下的拟合（岭回归时系数约 1e-36，不是标签上那个 λ 的解）；(3) 自动路径提前停止，gaussian 按相对增益、binomial 按绝对增益，阈值 1e-5；(4) `cv.glmnet` 的每个训练折**自建路径**，再按 λ 线性插值到全样本的 λ 上（给定 `lambda_` 时才在这些 λ 上直接拟合）。收敛阈值比 glmnet 默认严得多；p > n 且 λ 接近 0 时 glmnet 自己没收敛（差 1e-3），那里的证据是次梯度条件，不是对 R。与 `sp.shrinkage` 并存：后者的惩罚写在 RSS 上、用 n−1 标准差，换算式在指南里。坐标下降是纯 NumPy，**不要**为提速引入 numba 的非惰性 import。
- **`sp.ipw(se_method='sandwich')`**：1.39 起也覆盖 `normalize=False`（HT）和 `trim > 0`。被截断的得分对系数的导数为零（估计量里它就是常数），倾向得分方程的 score / Hessian 用**未截断**的拟合值。没有第三方包算这个方差，证据是测试里独立写的数值微分堆叠 M 估计量（12 种组合 1e-10）。
- **`sp.aipw(trim=)`（1.39 起）**：倾向得分截到 `[trim, 1 - trim]`，默认 0.01；`trim=0` 是教材公式。截断数必须在 `_fit_propensity` **之外**统计（1.39 前在拟合函数里先截了一次，外层再数，`n_propensity_clipped` 恒为 0，警告永不触发）。新写的估计器若截断倾向得分，要报告截断数并告警。
- **`sp.sun_abraham`**：交互权重是各队列在该相对时间的**观测份额**（加权时是权重份额），不是单位份额——只有平衡面板上两者相等，非平衡面板必须用前者才与 `eventstudyinteract` 一致。`event_window` 默认只决定**报告**哪些相对时间（`window_rule='report'`，回归仍对全部相对时间饱和）；`'bin'` 把窗口外的相对时间并入端点；`'reference'` 是旧行为（窗口外的已处理观测进入参照组，系数被它们的效应污染），只为复现旧数字保留。
- **`sp.validation_scope`（`validation_scope.py`）是按配置 × 输出的证据映射**：每行对每个维度列出*实际运行过*的取值（禁止通配符），并列出比较过的输出（estimate / se / coverage / diagnostic）；只有输出可证明不依赖某维度时才在 `invariant` 里豁免并写理由。新增或改动核心估计量的选项、默认值或证据行时同步这张映射；行所归功的入口点必须出现在证据产物源码里（`test_each_artifact_calls_the_entry_point_it_is_credited_to`）。2026-09 建映射时抓出 `sp.iv(vce=)` 静默丢参、`sp.fast.feols` 默认 ssc 未被任何 parity 行覆盖、LIML 行其实跑的是 `sp.liml`。
- **`agent/_translation/`（`sp.from_stata` / `sp.stata`）**：处理函数只按**全称**读选项；缩写（`r` / `cl()` / `a()` / `vce(cl id)`）、前缀（`qui` / `eststo:` 剥掉，`by` / `svy` 拒绝；`jackknife` / `bootstrap` 前缀与 `vce(jackknife)` / `vce(bootstrap)` 自 2026-10 起由 `sp.stata` 在 `_stata_resample.py` 里重抽样执行——命令自带这类方差的如 `sdid` 不经过它；`sp.from_stata` 单行翻译仍拒绝前缀）、宏（拒绝）由 `_stata_options.py` 在处理函数之前统一处理，**不要**在各处理函数里再认缩写。任何选项只有三种去处：进调用、进 `untranslated_options`（附 note）、进 `ignored_display_options`；`sp.stata` 遇到未翻译的选项就拒绝执行。`if` / `in`（2026-10 起）由 `sp.stata` 按 Stata 的缺失值规则应用（缺失值大于任何数，`if x > 0` 会保留 `x` 缺失的行），算不出来的表达式（`e(sample)`、字符串函数）才拒绝；`sp.from_stata` 只翻译单行、手里没有数据，仍把限定条件放进 `unapplied_sample`。数据步（`generate` / `replace` / `keep` / `drop` / `sort` / `mvdecode` / `encode` / `preserve` / `restore` / `predict` / `scalar` / `display`）在 `_stata_datastep.py`，表达式在 `_stata_expr.py`（封闭文法的递归下降解析器，**不许**用 `eval`；新增函数要按 Stata 文档的缺失值语义写并补 `tests/test_stata_expr.py`），时间序列算子 `L.` / `F.` / `D.` 在 `_stata_tsops.py`（按 `tsset` / `xtset` 的时间变量在面板内取滞后，不是行位移）。`generate` 默认存单精度，与 Stata 一致，这是回放 log 能对到最后一位的前提。新增或改动处理函数时，读到的选项值若解析失败要记入 `untranslated_options`（禁止 `except: pass`），且 `python_code` 必须与 `arguments` 是同一个调用——`tests/test_stata_translation_grammar.py` 的两条不变式（加选项必留痕迹；代码与参数一致）会拦。目标函数默认值与 Stata 不同时把 Stata 的默认写出来（`csdid` 的 `base_period='varying'`、`xtabond` 的 `robust=False`）。`replace` 按 Stata 的顺序语义执行（2026-10 起）：表达式引用被替换变量更早的行（`x[_n-1]`、`L.x`、`x[1]`）时看到的是已更新的值，由 `DataSteps._replace_in_order` 处理：能把表达式改写成"只依赖本行各列"的（`_row_local`：`_n` / `_N`、其他变量的下标、随机数各物化成一列）走 `_replace_row_by_row`，只重算读到了变动行的那些行，链多长都是每环一次小求值；含"对自身早先行的 running `sum()`"的走 `_replace_in_row_order`（按行顺序，先判条件，未选中的行不推进累加）；改写不了的（`sum()` 套 `sum()`）才整列逐轮求不动点，链长超过 20,000 行拒绝。**`sum()` 只在命令选中的行上累加**（`gen s = sum(x) if d`）：数据步先定下 `if` / `in` / `by` 选中的行，经 `stored["_sum_rows"]` 交给求值器，新增会求值表达式的数据步要沿用 `_evaluate_assignment`。`L.x` 生成的列登记在 `DataSteps.ts_derived`，每次被引用都按当前数据重算，**不要**再按"列已存在就复用"处理。跨行的语法（`///` 续行、注释、`#delimit`、文本定义的 `global` / `local` 宏、`xtset` 的面板声明）在 `_stata_script.py`，只由 `sp.stata` 使用；只有"读得出来"的才解析：未定义宏、扩展宏函数（`local n : word count ...`）、`syntax`、`mata` 拒绝。**2026-10 起循环与计算宏会执行**（`_stata_flow.py`：`forvalues` / `foreach` / `while` / `if … else` 块、`local x = exp`、`` `=exp' ``、`tempvar` / `tempfile`、带 `args` 的 program），做法与 Stata 相同——把块体收到配对的 `}`，每轮设好循环宏后逐行交回会话，所以块体里的命令走的仍是同一套逐行翻译与拒绝规则，循环本身不引入新的语义。多数据集命令（`save` / `use` / `append` / `merge` / frames）在 `_stata_multi.py`，**不读写磁盘**：`save` 只在会话里留一份，外部数据集经 `sp.stata(..., files={名: DataFrame})` 传入；矩阵（`J()` / `e(b)` / 逐格赋值 / `svmat`，以及和、积、转置、`inv()` 这类表达式）在 `_stata_matrix.py`，其余矩阵函数（`cholesky` 等）拒绝；`egen` 在 `_stata_egen.py`。这些模块里每条语义都对照过真 Stata（见 `docs/dev/2026-10-04-clarke-applied-microeconometrics-review.md`），新增函数照此办理：先在 Stata 里跑出答案，再写实现。`python scripts/stata_corpus_scan.py <目录>` 是探测器：拿一批 do 文件看翻译覆盖率，按"出现在几个项目里"排序（它只回答"翻译了没有"；手里有 Stata log 时用 `python scripts/stata_log_replay.py <log> --data <dta 目录>` 回答"数字对不对"，逐个对照 log 里打印的每个系数、标准误、检验统计量——2026-10 用 Stock & Watson 4E 的 13 份 log 查出语料扫描报告 97.6% 忠实翻译的同一批文件里，稳健 probit / logit 的标准误和其后的 Wald 检验全是错的小样本因子）；修复要依据 Stata 文档、用合成命令测试，并在没参与修复的 do 文件上复查，**不要**照着某篇论文打补丁，也不要为只出现在一篇论文里的用户命令新增 `sp.*` 函数。2026-10 用 12 篇复现包的语料查出：修复前只有 12.9% 的估计语句能原样正确翻译，28% 是静默错误。
- **`sp.stata` 的数据管理 / 描述统计 / 抽样 / 编程层（2026-10 Kohler-Kreuter 一轮新增）**：数据管理命令在 `_stata_manage.py`（`recode` / `xtile` / `destring` / `levelsof` / `assert` …），描述与抽样在 `_stata_describe.py`（`tabulate` 全套选项、`mean` / `proportion` / `total` / `ratio`、`svyset` + `svy:`），表达式函数在 `_stata_functions.py`（字符串按**字节**计，`u*` 函数按字符），`margins` 的网格与因子水平在 `_stata_margins.py`，估计后命令在 `_stata_postrun.py`（`predict` 的影响统计量、`estat gof`、`lrtest`、`e(sample)`），宏与 `syntax` 在 `_stata_macro.py`。新增命令往对应模块加，不要再塞进 `_stata_session.py`。四条容易踩的约定：(1) **扩展缺失值**——数据里 `.a`–`.z` 与 `.` 都是 NaN，session 用 `stored['ext_missing']` 记录哪些变量可能含扩展缺失（来源：`attrs['_ext_missing']`、`__miss` 伴随列、缺失码标签、session 内赋值），在这些变量上 `x == .` / `x != .` / `x > .`、`by`、`tabulate, missing`、`collapse, by()` 一律拒绝（Stata 里 `.a != .` 为真、每种缺失各成一组），`missing(x)` / `x < .` / `x >= .` 照跑；新增会按缺失值分组或比较的命令时要接上这道检查。(2) **`mean` 族的方差**——无权重无聚类时 Stata 把 `over()` 各组当作独立的简单随机样本（`s/√n`，`proportion` 用 `√(p(1-p)/n)`，`total` 用 `√n·s`）；有 pweight / cluster / svy 时才是线性化方差，后者与 `sp.svydesign` 共用 `_design_vcov`。svy 默认 `singleunit(missing)`：存在单 PSU 层时标准误为缺失，不要替用户改成 certainty。(3) **`e(sample)`** 按「内存里那个 DataFrame 对象」绑定，`sort` / `drop` 之后即失效并拒绝，不要改成按行号猜。(4) `levelsof` 对非整数只写 16 位有效数字（`%18.0g`），`float` 变量的 16.1 因此在循环里匹配不到自己——这是 Stata 的行为，照做。经典检验（`sp.ranksum` / `kwallis` / `spearman` / `ktau` / `ksmirnov` / `median_test` / `robvar` / `oneway` / `signrank`）在 `inference/rank_tests.py`，回归与 logit 的影响诊断（`sp.influence_measures` / `sp.logit_influence` / `sp.logit_gof`）在 `diagnostics/influence.py`；Stata 18 参考数由 `tests/reference_parity/_fixtures/kk_syllabus_reference.do` 在合成数据上生成，改动这些函数或对应翻译后重跑该 do 文件再跑 `test_kohler_kreuter_stata_parity.py`，**不要手改 `kk_syllabus_stata.txt`**。回放脚本新增了「表格逐数比对」（`BAG`）与 `display` 文本比对，新命令没有专用比较器时把命令名加进 `BAG` 即可。
- **`matching/`**：倾向得分的 logit / probit 拟合只有一份，在 `matching/_binary_fit.py`（标准化设计上做 Newton，冗余协变量按 Stata 的做法剔除、系数记 0，未收敛发 `ConvergenceWarning`）。**不要**在别处再写一个直接对原始列 `np.linalg.solve` 的 IRLS：Hessian 奇异时它不报错，旁边有 `re74^2` 这种 1e9 量级的列时误差会漏进得分（2026-10 邱嘉平教材第 6 章，ATT 偏 4%；包内 Lalonde 数据加一个重复虚拟变量可复现，605.80 对 Stata 468.10）。匹配用的得分**不截断**，相同协变量的行用 `_index_by_row` 只算一次（并列靠位级相等判断）；比较"距离相同"时两边必须出自同一个数组——`cutoff**2`（C 库 pow）与 numpy 的 `d**2` 约六百个数里有一个差最后一位。`sp.pscore` 是 Becker-Ichino 的分块与平衡性检验（Stata `pscore`），`attnd` 翻译成 `sp.psmatch2(ties=True)`，`common_support='treated'` 是它们的 `comsup`（按处理组得分范围剔对照），与 psmatch2 的 `common`（`'minmax'`）不是一回事。`atts` 是 `sp.match(method='stratify', strata=<块列>)`，`attk` 是 `sp.psmatch2(method='kernel', kernel='normal', bwidth=0.06)`；`attr` 是 `radius_weights='pairs'`（每个半径内的配对算一次，按"半径内有几个处理个体"给对照加权）：只为复现已有结果而提供，**默认值必须保持 `'treated'`**（psmatch2 的半径匹配），翻译时带说明。`sp.heckman` 两步法的 ρ̂ 出了 [-1, 1] 时按 Stata 截断并告警，不要把 ρ̂²>1 直接代进方差公式。Stock-Yogo 临界值在 `diagnostics/_stock_yogo.py`，是从 Stata `estat firststage` 的 `r(mineigcv)` 实跑导出的，**不要手改**。
- **`timeseries/`**：`sp.regress(robust='hac')` 的滞后阶数用 `hac_lags=`（默认 Newey-West 1994 规则 `floor(4(T/100)^(2/9))`），小样本因子用 `hac_small=`——默认不乘，对应 `sandwich::NeweyWest(adjust=FALSE)` 和 statsmodels；Stata `newey` 乘 `N/(N-K)`，要 `hac_small=True` 才逐位一致（Track A 51 号模块对 Stata 的 1e-2 容差就是这个因子，不是数值误差）。`robust='ewc'` 是 Lazarus-Lewis-Stock-Watson (2018) 的等权余弦长期方差估计量（`ewc_df` 个余弦项，默认 `floor(0.4 T^(2/3))`），推断用 `t(ewc_df)`，联合检验在 `sp.test` 里按 `(B-m+1)/(Bm)·W ~ F(m, B-m+1)` 换算（`data_info['hotelling_df']`）；没有可对照的第三方实现，证据是按定义重算、取满 `T-1` 项时恰为 `HC0·T/(T-1)` 的恒等式、以及真零假设下的拒绝率。`sp.unitroot` 的 ADF 与 `statsmodels.adfuller` 在统计量、p 值、临界值、选阶上全等；DF-GLS 的临界值是 `_critvals.py::DFGLS_SURFACE`，由 `scripts/simulate_dfgls_critical_values.py` 对模拟的零分布拟合而来（分位数存在 `tests/reference_parity/_fixtures/dfgls_null_quantiles.json`），**不要手改表**：改拟合形式用 `--refit` 重拟合并让 `test_surface_is_the_fit_to_the_committed_simulation` 通过；渐近临界值在宏观数据常见的样本长度上会过度拒绝，所以不用。`sp.unitroot(test='pp' | 'kpss')` 与 Stata `pperron` / `kpss` 逐位一致；**KPSS 的原假设是平稳**，`reject=True` 是单位根的证据，结果对象用 `null` 标明方向。`sp.arima` 的 `trend=` 默认值跟 R `stats::arima` 和 statsmodels `ARIMA`：不差分时估计常数项、差分后不估计；Stata `arima` 永远带常数，翻译时写出 `trend='c'`（2026-10 之前默认的 `statespace` 路径不估计常数，均值非零的序列 AR 系数被推向 1）。`sp.garch` 的 `p` 是滞后条件方差阶数、`q` 是滞后残差平方阶数（ARCH(1) 是 `p=0, q=1`），`q=0` 不可识别、直接拒绝；Stata `arch` 默认 OPG 标准误，对应 `vce='opg'`。`sp.svar` 只做识别这一步（简化式取自 `sp.var`）：短期 AB 模型和长期约束走同一个似然（长期约束等价于把 A 固定为 `(I - ΣA_i)^{-1}`），标准误用**期望**信息阵（过度识别时与观测信息阵不同，Stata 用前者）；符号约束返回的是可容许旋转的集合，分位数带**不是**置信区间，文档和 `summary()` 都必须这么说。`sp.arima` 的差分模型用**精确扩散初始化**（`_exact_diffuse`，并把 `d + sD` 个扩散观测从似然里剔除），不要退回 statsmodels 默认的近似扩散先验（方差 1e6）：数据量纲一大它就成了信息先验，GDP 的 AR(1) 系数随单位在 0.02 到 0.22 之间变，而 R / Stata 与量纲无关（2026-10 Maitra 教材核查发现）。AIC / BIC / AICc 自己算，不读 statsmodels 的。`auto=True` 对齐 `forecast::auto.arima(stepwise=FALSE, approximation=FALSE)`：KPSS 定 `d`，常数 / 漂移项参与搜索，根离单位圆不到 1% 的候选丢弃。`sp.granger_causality` 的 `F_stat = W / df` 是 Stata `var, small` 的口径（Stata 18 实测），不是 statsmodels 的经典 F，后者要 `se_df='r'`，这是约定不是 bug。`sp.ardl`（AR / ADL 预测回归）的滞后项命名为 `<var>_L<k>`，`sp.stata` 的时间序列算子改写用同一套名字；`sample=` 固定估计窗口而滞后取自更早的行，BIC 选阶必须在公共样本上比。`sp.structural_break(method='sup-f', break_vars=, vce=)` 是教科书的 QLR（只让部分系数断裂、稳健 F），默认值不变。
- **`mcmc/` 的第二批（1.39 起）**：`sp.bayes_sur` / `sp.bayes_shrink` / `sp.stochvol` / `sp.bayes_arima` / `sp.bayes_mvprobit` / `sp.bayes_mnprobit` / `sp.bayes_mixture` / `sp.gp_regress` / `sp.bart` / `sp.abc`。新增采样器共用 `_results.posterior_table`，不要再各自拼汇总表。**模块名不要与函数同名**（`mcmc/sv.py`、`trees.py`、`simulation.py` 就是因此改名：`from .x import x` 之后 `statspai.mcmc.x` 变成函数，测试里取不到模块的私有核）。没有可网格积分的后验时用另外三种证据之一，写在 `docs/guides/bayesian_econometrics.md` 的 "How each sampler is checked"：**穷举**（混合模型的全部划分、BART 的全部树）、**先验重要性抽样**（probit 系统，格概率用求积）、**联合分布检验**（每次 sweep 后按当前状态重抽数据，参数的边际必须是先验；任何一个满条件写错都会让先验矩偏掉）。联合分布检验里若潜变量决定了数据（probit 的符号），数据步必须连潜变量一起重抽，否则链不可约。probit 系统在**未识别**的协方差上采样、逐次抽样归一化，先验因此只对未归一化参数成立；多项 probit 只有个体回归元时协方差近乎不可识别，链与 `bayesm::rmnpGibbs` 一样慢，这是模型的性质。`sp.bart` 是按论文独立写的引擎（BART / dbarts 是 GPL，**不得**移植），只有 birth / death 两种移动；`min_leaf` 靠拒绝提议实现，等价于把树先验截断到满足条件的树，联合分布检验的先验模拟也要做同样的拒绝。`sp.bcf` 不是建在 BART 上的（冗余函数是梯度提升），不要在文档里把两者连起来。
- **`mcmc/` 的回归工作流（2026-10 Gelman-Hill-Vehtari 一轮）**：`sp.loo` / `sp.waic` / `sp.kfold` / `sp.loo_compare` / `sp.loo_predict` / `sp.psis`（`mcmc/crossval.py`）、`sp.ppc` / `sp.bayes_r2` / `sp.loo_r2`（`mcmc/checks.py`）、`sp.bayes_regress(prior='weakly_informative', offset=, exposure=)` 与结果上的 `posterior_linpred` / `posterior_epred` / `posterior_predict` / `log_lik`（`mcmc/_workflow.py`）。`loo`、`rstanarm`、`arm`、`retrodesign` 都是 GPL，**只当黑盒比输出**：PSIS 按 Vehtari et al. (2024) 与 Zhang-Stephens (2009) 独立写，给定同一个对数似然矩阵与 `loo` 2.9.0 对到 1e-13（逐点 elpd、Pareto k、n_eff、平滑后的权重）；k 的阈值是 `min(1 - 1/log10(S), 0.7)`，**不是**书里固定的 0.7。`mcse_elpd` 用 delta 法，与 `loo` 的近似只对到 1%，不要去凑。弱信息先验照 rstanarm 默认（斜率 `2.5 sd(y)/sd(x)`，截距先验定义在**中心化**回归元上，所以传给采样器的是原始系数的**满协方差**先验；高斯模型的 `sigma ~ Exp(1/sd(y))` 不共轭，用"从平坦先验的逆伽马提议、按 `exp(-rate(σ'-σ))` 接受"的独立 Metropolis 步，接受率在 `_extras['sigma_accept']`，不要报成 `acceptance_rate`，否则触发 0.1–0.7 的告警）。**默认先验正在走弃用流程**：1.39 不传 `prior` 时仍是固定先验（方差 1000）并发 `DeprecationWarning`（只在模型有弱信息先验、且没传 `prior_mean` / `prior_var` 时），**1.40 改为 `'weakly_informative'`**；改默认时同步 MIGRATION 与 `test_default_prior_change_is_announced_only_where_it_applies`，refit（`kfold`）里把 `prior` 固定为实际用的那个，否则每折都告警。logit / poisson / negbin / mlogit / ologit 的采样核是 `_core.metropolis_mixture`（80% 以后验众数为中心的 t5 独立提议 + 20% 随机游走；`acceptance_rate` 只报随机游走那部分，另一部分在 `_extras['independence_accept']`），**不要**退回单一随机游走（ESS 只有抽样数的一成）。分组二项用 `trials=` 或 `cbind(成功, 失败)`（`_TrialsSpec`，对新数据预测时由它重建试验数），`bayes_r2` / `loo_r2` 对分组拟合直接拒绝。regularized horseshoe（`slab_scale=`）的尺度参数不共轭，在对数尺度上做 Metropolis；全局尺度要靠“τ 乘、所有 λ 除同一因子”的联合移动和每轮 5 次尺度更新才混得动，它的精确后验网格在尺度方向要取得很远（有 slab 时大尺度不损失似然，后验保留半柯西尾）。二值因变量的 horseshoe 是 `bayes_shrink(family='logit')`（`mcmc/_shrink_logit.py`），系数靠 Polya-Gamma 增广（`mcmc/_polyagamma.py`，按 Polson-Scott-Windle 2013 的精确拒绝算法独立写、已向量化）变成正态满条件，尺度更新与高斯版共用、残差方差取 1；它的精确后验参照**必须先把尺度从先验里数值积掉**再在（截距, 斜率）上用三次间距的网格（边缘 horseshoe 密度在 0 处有对数极点），在（斜率, 尺度）或（斜率/尺度, 尺度）上打网格都分辨不了。新增似然时在 `_workflow.py` 的 `pointwise_log_lik` / `predictive_draws` / `residual_variance` 各补一支，`test_pointwise_log_lik_sums_to_the_sampler_likelihood` 会核对逐点矩阵加总等于采样器自己的标量似然。结果里辅助参数不在最后一列的模型（`bayes_shrink` 的 `lam` / `tau` 排在 `sigma2` 后面）要在模型对象上设 `sigma2_index`。horseshoe（`bayes_shrink(prior='horseshoe')`）是 Makalic-Schmidt 的 Gibbs；它的精确后验测试必须把系数**解析积掉**，在系数上打网格分辨不了小局部尺度造成的零点尖峰（第一版测试因此误判采样器差 8 个 MC 标准误）。`sp.retrodesign(dof=)` 默认按真正的 t 检验算（功效与 S 型错误是非中心 t，夸大倍数对估计的标准误做一维求积）；Gelman-Carlin 论文里的函数是平移的中心 t（`method='shifted'`），`retrodesign` 包两者各用一半，三者在低功效小自由度下不同，证据是对 t 检验本身的模拟。
- **共线回归元（`core/_collinear.py`，2026-10 起）**：`sp.logit` / `probit` / `cloglog` / `glm` / `poisson` / `nbreg` / `ologit` / `oprobit` / `mlogit` / `tobit` / `qreg` / `svyglm` 在拟合前调用 `drop_collinear`（按列名建模的用 `drop_collinear_names`），按公式书写顺序扫描、常数项最先、剔除相关集里靠后的那个，发 `note: x omitted because of collinearity (...)` 并写入 `model_info['omitted']`（`[{variable, reason}]`，与 `sp.regress` 同形）。此前这些入口把秩亏设计直接交给优化器，返回 1e13 量级的系数、0 / 1e6 / NaN 的标准误，多数没有告警（书中 Childcare 例子把一个分类变量的全部虚拟变量和常数项一起放进倾向得分模型）。**新增基于似然的估计器时接上这一步**，不要依赖 `pinv` 或"Hessian 奇异"告警。判据是逐列的（单位化后对已保留列投影，剩余长度小于 1e-9 才剔除），不是对整个设计矩阵的秩容差，所以均值大、离散小的回归元（年份）不会被误删；`sp.regress` 仍用它自己的结构化检测（NIST 病态但满秩的设计）。二值模型的分离告警（`logit_probit._warn_if_separated`，`sp.glm` 的二项族共用）的判据是"线性指数把所有观测分对"或"拟合概率在机器精度上等于 0 / 1"，旧判据另要求 99% 的拟合概率贴边，小样本下会漏报。
- **公式的 R 写法（同一轮）**：`y ~ .` 由 `core/utils.expand_dot` 在 `create_design_matrices` 入口展开（含 `|` 的 IV / 固定效应公式不展开）；布尔型因变量 `(earn > 0) ~ x` 在同一入口折成一列 0/1（patsy 会编成 `[False]` / `[True]` 两列）；有序 / 多项模型因变量外的 `factor()` / `C()` / `ordered()` 由 `multinomial._bare_outcome` 去掉。`sp.glm` 认 `cbind(成功, 失败) ~ x`（折成"比例 + 试验数作权重"，对数似然是未分组数据的，比 R 少一个二项系数常数）、`family='quasipoisson' | 'quasibinomial'`（即 `scale='x2'`）、`link='robit(dof)'`（Liu 2004，`robit(dof,unit)` 是书里方差为 1 的参数化）。
- **`timeseries/dlm.py`（1.39 起）**：`sp.dlm` 是随机游走系数的动态线性模型（Kalman 滤波 / RTS 平滑 / MLE / FFBS Gibbs，numba 内核）。协方差更新用 Joseph 形式——朴素的 `R - KQK'` 在扩散先验（`C0 = 1e7`）下只剩一两位有效数字，不要换回去。证据：给定方差时滤波、平滑、似然与 R `dlm` 对到 1e-9（`dlmLL` 是不含 `2π` 常数的负对数似然），Gibbs 对照不经过 Kalman 滤波的精确后验（`tests/reference_parity/test_dlm_parity.py`）。缺失行直接拒绝，不静默拼接不相邻的日期。
- **协变量列表里的分类变量（`core/_covariates.py`，2026-10-05 起）**：接收 `covariates=[...]` 的估计器用 `@_expands_categorical("covariates")` 装饰（已接入 ipw / aipw / match / psmatch2 / sbw / genmatch / optimal_match / cardinality_match / ebalance / overlap_weights / cbps / tmle / g_computation / dml / metalearner / auto_cate / policy_tree / dose_response / drdid / callaway_santanna、倾向得分诊断，以及 DiD 各估计器的 `controls=` / `covariates=`；森林家族走自己的 `forest/_grf_family.py::one_hot_covariates`：每个水平一列、不设基准，预测时未见过的水平报错）。`category` dtype、字符串列、`C(col)` / `i.col` 一律展开成哑变量（首水平为基准），普通整数列仍按数值处理；全数值的调用原样穿透，不复制数据。展开记录在 `model_info['covariate_expansion']`，对新数据预测时用 `apply_covariate_expansion` 重建（`sp.predict_cate` 已接）。**新增这类估计器时加装饰器，不要在函数里 `.astype(float)` 硬转**：此前 `category` 列被静默当成连续变量（书中第 5 章 AIPW 0.2757 对 0.2712），字符串列在 numpy 里报 `could not convert string to float`。
- **二值处理的校验**：只对 0/1 处理有定义的函数入口调用 `core._validate.require_binary_treatment`。`sp.balance_table` 遇到三臂处理曾返回一张 N=0 的空表，`sp.cate_eval` 遇到连续处理曾返回带标准误的 RATE。
- **DiD 的日期型时间列（`did/_core.py::calendar_time_aware`）**：`callaway_santanna` / `sun_abraham` / `did_imputation` / `etwfe` / `gardner_did` / `stacked_did` / `wooldridge_did` / `twfe_decomposition` / `bacon_decomposition` / `did_forest` 接受 `datetime64` / `Period` 的时间列与 cohort 列，按观测到的日期顺序编号为 1..P，缺失或晚于样本末期的 cohort 记为 0（never treated），编号存 `model_info['calendar_time']`。新增 staggered 估计器时用同一个装饰器，并加一条「日期与整数编码结果相等」的测试：`gardner_did` 接入前在日期型面板上静默返回 3.13（整数编码为 1.93）。面板路径上 `callaway_santanna` 拒绝重复的 (unit, period) 行（旧代码 `aggfunc="first"` 静默取每格第一行，`sp.did(..., time='post')` 在日度数据上符号都反了）。
- **`regression/gee.py`（`sp.gee`，1.39 起）**：Liang-Zeger 矩估计。三家参考实现在三处约定上各不相同，**都已精确定位，不要"修"成一致**：(1) 矩的除数——R `gee` 用 `N - p`（`dof_correction=True`，默认），Stata `xtgee` 用 `N`（`nmp` 才是 `N - p`）；(2) 稳健协方差——我们与 R `gee` 都是裸三明治，Stata `vce(robust)` 再乘 `G/(G-1)`；(3) 尺度——R 对所有分布族估计 φ，Stata 对 binomial / poisson 固定为 1（`scale=1`，只影响 model-based SE）。AR(1) 参数有两个矩估计，两个都提供：`corstr='ar1'` 是相邻配对的合并矩（= Stata `xtgee, corr(ar 1)`，两种除数下都对到 1e-7）；`corstr='ar-m'` 是 R `gee` 的 `"AR-M"`（各簇的平均相邻乘积之和 ÷ 各簇的均方之和，对到 1e-12）。平衡面板上两者只差自由度项，簇大小不等时才分开（示例数据上 0.557 对 0.568）。这一条第一轮曾记为未解释，是用平衡面板做黑盒试验（隐含除数恰为 `(m-1)(N-p)/m`）定位出来的，GPL 源码没有读。`sp.from_stata("xtgee ...")` 会把 Stata 的四个默认值（`corstr='exchangeable'`、`vce='model'`、`dof_correction=False`、binomial / poisson 的 `scale=1.0`）显式写进调用。
- **`survival/models.py`**：`sp.cox` 的稳健 / 聚类方差必须用**与 `ties=` 相同规则**的得分残差（`_cox_score_individual(breslow=)`）。1.39 前 Efron 拟合用的是 Breslow 残差，无并列时两者相同所以单元测试没抓到，并列多的数据上稳健 SE 偏 2%（⚠️ 已修）。约定：`robust='hc0'` 是裸三明治（= R `coxph(robust=TRUE)`；Stata `stcox, vce(robust)` 再乘 `N/(N-1)`），`cluster=` 带 `G/(G-1)`（= Stata；R 不带）。`sp.kaplan_meier(conf_type=)` 的默认值正在走弃用流程：1.39 不传时仍是 `'plain'` 并发 `DeprecationWarning`，**1.40 改为 `'log-log'`**（Stata `sts` 的默认；R `survfit` 是 `'log'`）；改默认时同步 MIGRATION 与文档里的调用（现已全部显式写出 `conf_type`）。
- **`regression/gam.py`（`sp.gam`，1.39 起）**：单变量 P-spline 光滑项（三次 B 样条 + 二阶差分惩罚 + 样本内和为零约束），与 `mgcv::gam` 的 `s(x, bs="ps")` 同基、同惩罚，固定平滑参数下逐位一致（1e-9），所以**改基函数、节点或约束之前先想清楚会不会丢掉这个可检验性**。三条约定：(1) `lambda_` 乘的是原始 `D'D`，mgcv 的 `sp` 乘的是 `D'D / S.scale`，换算 `lambda_ = sp / S.scale`；(2) 默认 `method='reml'`（mgcv 默认是 `GCV.Cp`，但其作者推荐 REML）——在线性真值的夹具上 GCV / UBRE 会把 edf 选到 8.6 / 9，REML 给直线；(3) GCV / UBRE 曲面可能有多个局部极小，我们用网格 + 精修取全局较低者，与 mgcv 选的不同时比**准则值**而不是曲线。所有求解都走「设计矩阵叠惩罚平方根」的 QR，**不要**退回正规方程（λ≈1e13 时会丢 5 位）。mgcv 是 GPL，只当黑盒用（`smoothCon` / `S.scale` / `gcv.ubre`），源码没有读。`s(x, by=d)`（d 为数值列）不做中心化、自带水平项，与 mgcv 的 `by=` 逐位一致；`vce=` / `cluster=` 的三明治以「曲线退化成直线时等于 `sp.poisson` 的 hc0 / robust / cluster SE」这个恒等式为证据，**不要**把 R 的 `vcov(gam, sandwich=TRUE)` 当参考——它另有一个没查明的小样本修正（约大 1%）。第五轮（2026-10-06）补上了 `te(x, z)`、`s(x, bs="tp")`、`s(g, bs="re")`，基与惩罚的构造在 `regression/_gam_terms.py`，引擎改为「惩罚列表」（每个惩罚一个 λ，一个项可以有多个，`fit.lambdas` 是全量）。黑盒对出来的 mgcv 约定，改之前先看：张量积的边际要先换成「k 个等距点上的函数值」参数化，每个边际惩罚除以自己的最大特征值，再 `lambda = sp / S.scale`；惩罚重叠时 REML 的对数行列式按总惩罚的非零特征值算，不能逐项相加；thin plate 核是 `|r|^3 / 12`，按特征值截到 k 维，超过 2,000 个不同取值时等距抽样；随机效应是单位阵岭惩罚、`S.scale = 1`，`lambda = scale / sigma_b^2`。`smooth_terms` 的检验是 Wood (2013)：`ref_df` 是 `2 diag(F) - diag(FF)` 之和，秩取非整数，p 值是两种特征向量符号选择的平均（mgcv 打印的统计量只是其中一种，所以**统计量本身只对到 12%，p 值对到 5e-6**，不要去「修」统计量），混合卡方尾部用 Imhof 反演；随机效应不给检验（方差在边界上）。`partial(simultaneous=True)` 是后验模拟的同时置信带。仍未做：多于两个变量的张量积、`ti()`、二元 thin plate、随机斜率。
- **公式里的 `I(x^2)`**：`core/utils.r_power_in_identity` 把 `I()` 内的 `^` 改写成 `**`（R 用户的写法；patsy 会把它当按位异或）。**新增任何直接调用 patsy `dmatrix` / `dmatrices` 的入口都要先过这个函数**，否则同一个公式在不同估计器里一个能跑一个报 `xor`。`I()` 之外的 `^` 保持 patsy 的交互展开语义，不动。
- **`panel/` 的随机效应方差分量只有一份实现**：`panel/xt_tools._swamy_arora`，自由度按**秩**计（组内不变的回归元不占组内回归的参数），`panel/_cre.py::variance_components` 调它；每个回归元都随组内变化时仍用 linearmodels 自己的数。linearmodels 的 `RandomEffects` 按列数计，含行业虚拟变量这类回归元时 `sigma_e` / `theta` / 系数都会偏（2026-10 Hansen 第 17 章对照 Stata 查出）。**不要**再在别处重算这套分量。
- **`regression/iv.py`**：投影走 QR、2SLS 第二阶段走 `lstsq`，不要改回 `(W'W)^-1` / `(X'PX)^-1`（条件数平方两次，立方工具变量设计上丢 7 位）。与外生回归元和其它工具变量共线的排除工具变量由 `_independent_instruments` 丢弃并告警（`model_info['omitted_instruments']`），不再抛 numpy 的 `Singular matrix`。稳健 2SLS 之后 `sp.estat(result, 'overid')` 的 Hansen J 同时就是 Stata `estat overid` 的稳健 score 统计量（键 `score`）；`iid_errors` 里是 Sargan / Basmann（LIML 为 Anderson-Rubin / Basmann F）。Stata 在 `ivregress, perfect` 下的 score 统计量随工具变量书写顺序变化，那是参考实现的问题，不要去对它。
- **`mackinnon1994_pvalue`（`panel/unit_root.py`）在拟合范围之上返回 1**（含常数 2.74、含趋势 0.70），与 Stata、statsmodels 一致；plm 在范围外继续代三次多项式，带趋势时 +3.25 给出 0.06，不要为了对 plm 去掉截断。
- **教材驱动的小估计器**（2026-10 Hansen）：`sp.cnsreg`（约束最小二乘 / 有效最小距离）、`sp.nls`（参数写在花括号里，表达式由 `_stata_expr` 的封闭文法求值，不用 `eval`）、`sp.model_average`（Mallows / 刀切 / 平滑 AIC、BIC，单纯形上的 QP 在支撑集上解 KKT 得精确解）都在 `regression/`；`sp.pca` / `sp.factor` 在 `multivariate/`；`sp.jackknife` 在 `inference/_jackknife_general.py`，与 `sp.bootstrap` 对称。Stata 的 `factor, ml` / `ipf` 与 `nl` 默认容差较松，比对前先让 Stata 收紧容差；Stata 的 jackknife 把重复值存成单精度，标准误只能对到 1e-7。
- **`timeseries/` 预测工具箱（2026-10，来自 Hyndman FPP 审查）**：`sp.ets` / `sp.simple_forecast` / `sp.stl` / `sp.classical_decompose` / `sp.forecast_accuracy` / `sp.tscv` / `sp.ljungbox` / `sp.ndiffs` / `sp.nsdiffs` / `sp.boxcox_lambda` / `sp.fourier_terms` / `sp.seasonal_dummies` / `sp.ts_features` / `sp.bootstrap_series` / `sp.bagged_forecast` / `sp.hierarchy` / `sp.reconcile`。预测器的 `boxcox=` / `biasadj=` 统一走 `_forecast_common.py` 的 `resolve_boxcox` / `back_transform_frame`（区间端点直接反变换，点预测默认是中位数，`biasadj=True` 才是均值；`fitted_values` 在原尺度、`residuals` 在变换后尺度，与 R 一致），新预测器不要自己写反变换。`sp.ets` 允许序列中间有缺失：缺失期状态按零新息推进、不进似然（R 是截取最长连续段，我们故意不同）。`sp.reconcile` 的 top-down / middle-out 要求严格层级，分组结构直接拒绝；`proportions="forecast"` 不是线性映射，`G` 为 NaN，也因此不给 `sd=`。`sp.ts_features` 只实现能对上 R `tsfeatures` 的那些特征（1e-13），谱熵、Hurst、非线性、GARCH 特征没有做，**不要**凭印象补未对齐的特征。bagging 是随机的，证据只有结构性质，不要写成与 R 对齐。canonical 参考是 Hyndman 自己的 R 包（`forecast`、`hts`），**不是**教材 Python 版用的 statsforecast / hierarchicalforecast：后两者的 ARIMA / ETS 优化器会停在次优解，`mint_shrink` 用了中心化协方差。约定：所有拟合结果的 `.forecast(horizon, level=(80, 95))` 返回 `forecast` / `lower_<L>` / `upper_<L>` 列，共用 `timeseries/_forecast_common.py`（读序列、未来索引、区间布局），新预测器不要另起一套。`sp.ets` 的似然、参数域、选模、解析预测方差在 `_ets_core.py`（numba），**经 `_register_lazy` 惰性加载**，`import statspai` 不得引入 numba；在 R 的参数处逐位复现 R（1e-13），估计值不必与 R 相同——R 的 Nelder-Mead 2000 步就停，我们重启优化，似然只高不低，测试断言的是「R 自己的似然代码在我们的参数处给出我们的值，且不低于 R 的最优值」。R `forecast` 9.0.2 的 `forecast.ets` 对 ETS(A,N,A) / ETS(M,N,A) 的预测方差把 γ 错放了一期，我们按 Hyndman et al. (2008) 表 6.2，附模拟证据（T4，参考实现缺陷）；上游开发版已修（PR #1173），用更新的 `forecast` 重生夹具后要删掉记录这一差异的那条测试和 ANA / MNA 豁免。`sp.arima` 在**差分后的序列**上估计（Stata 与 R `arima(diff(y))` 的似然定义；`forecast::Arima` 把差分当近似扩散状态，双重差分时对数似然差约 1e-2），然后把参数代回**精确扩散初始化**（`_exact_diffuse`，Maitra 审查引入）的水平模型求拟合值、标准误与预测；搜索阶段的似然由 `timeseries/_arma_core.py` 的新息算法（numba，首次调用 `sp.arima` 时才 import）计算，与 statsmodels 卡尔曼滤波对到 1e-8、快约 15 倍，水平模型只为最终返回的那次拟合构建（月度 `auto=True` 从分钟级降到秒级，**不要**把候选模型的评估改回 `est.fit`）；量纲差的序列先缩放再搜索、标准误在缩放后的模型上算再映射回来，估计值、似然、预测、标准误对计量单位全部不变（两条线的测试都在守这一点，改 `_fit` 时两边的 `test_dynamic_modelling_parity.py` 与 `test_forecasting_r_parity.py` 都要跑）；每次拟合都从两个起点出发（statsmodels 默认起点 + R `CSS-ML` 式的条件平方和估计），再过「单纯形 → 拟牛顿」复核，**不要**退回单次 L-BFGS：教材九个序列的 32 个模型里它有 13 个停在次优解且报告已收敛。`auto=True` 是 Hyndman-Khandakar：先定 D（STL 季节强度 > 0.64）再定 d（KPSS），常数项由 AICc 决定，根模小于 1.01 的模型不返回（论文写 1.001，R 实际按 1.01 拒绝）。`sp.stl` 的默认值跟 R（季节窗 11、季节 loess 0 次、`ceil(window/10)` 插值步长），与 statsmodels 默认不同，docstring 给出复现 statsmodels 的参数。`sp.reconcile` 的协方差不中心化、除以 T，与 `hts` / `fabletools` 一致。夹具由 `tests/reference_parity/_fixtures/_generate_forecasting_{data.py,R.R}` 重生（先 Python 后 R，R 侧需要 `forecast` 与 `hts`）。
- **`timeseries/` 的 2026-10 Neusser 一轮**：`sp.garch` 的高阶模型曾停在边界（单次 simplex，`beta[2]=0`、似然低于嵌套的 GARCH(1,1)），现为多起点有界拟牛顿；`ar=` / `dist='t'` / `model='gjr'|'egarch'` 走数值导数路径，高斯常数均值仍走解析得分，**改搜索逻辑后两条路径都要过 `test_garch_extensions_stata_parity.py` 与 `test_garch_asymmetric_stata_parity.py`**。GJR 的门限项写在负冲击上（`gamma = -tarch`，`alpha = arch + tarch`），Stata 的预样本约定（门限项取满值、EGARCH 的 `|z|` 在 t 分布下仍以 `sqrt(2/pi)` 居中）是把我们的似然在 Stata 估计值处求值反推出来的，不要凭直觉改。脉冲响应的不确定性只有一个入口 `timeseries/irf_bands.py`（`sp.irf(ci=)` 的 delta 法与残差 bootstrap、`SVARResult.irf(ci='bootstrap')` 共用 `bootstrap_fits`），不要在别处再写一套；Stata 的 `.irf` 文件是单精度，对它的容差只能到 1e-6。bootstrap 的三种带（`efron` / `hall` / `kilian`）都从 `bootstrap_fits(..., boot)` 出，Kilian 偏差校正在根 0.92、T=60 时覆盖率 86%，百分位法只有 38%；**默认是 `kilian`**（2026-10-06 用户授权我决定：四个设计上它的覆盖率从不差于百分位法，这是"默认值跟参考实现"规则的一次有证据的例外），`efron` 是 Stata / `vars` 的做法，最大根超过 0.9 时告警。`sp.garch(in_mean=True)` 跟 Stata 的估计量：预样本方差在爬似然时是常数（外层交替到自洽），不是对参数求导的那个点。协整的秩检验与约束检验共用 `cointegration._johansen_residuals`。状态空间：自己写矩阵用 `sp.kalman_filter` / `sp.statespace`（时点约定是 `X_t = F X_{t-1} + V_t`，`(x0, P0)` 属于 `X_0`；statsmodels 与 KFAS 初始化的是第一个*预测*状态，对照时要映射），随机游走系数回归用 `sp.dlm`；diffuse 初始是大方差近似，不是精确 diffuse。`sp.xcorr(x, y)` 的 lag h 是 `corr(x[t+h], y[t])`（R `ccf`），Stata `xcorr` 是镜像。`sp.arima` 的最终拟合现有四类起点（默认、CSS、零系数、低一阶模型的估计补零），目的是"不比嵌套模型差"；改 `_fit` 后 `tests/test_arima_starts.py` 与前三轮的 ARIMA 测试都要过。`sp.mswitch` 的参数化与 Stata 相同（`p_ij = exp(-q_ij)/(1+Σexp(-q_ik))`、`lnsigma`），`theta` / `vcov` 可直接对 `e(b)` / `e(V)`；状态按常数递增排序，Stata 不排序。`sp.tvp_var` 的 Kalman 版逐方程等于 `sp.dlm`，不要再写第三个滤波。精确 diffuse 是 `init='exact'`，默认仍是大方差近似。`sp.tvp_var_sv`（Primiceri 的 TVP-VAR-SV）是随机估计器，只能记 S：证据是逐块精确条件检验加 Geweke 联合分布检验，对 `bvarsv` 有 1–4% 未解释的差距；指示变量的抽样顺序按 Del Negro-Primiceri 勘误，改顺序会让联合分布检验失败。`sp.mswitch_lrtest` 的每个 bootstrap 复制必须与观测数据走同一套搜索（size 靠这个成立）。**新增结果类时，推送前要跑 `tests/test_result_protocol_audit.py` 与 `tests/test_result_agent_contract.py`**：第一轮漏了 `ZivotAndrewsResult` 的结果协议，按文件名挑的测试子集没覆盖到，全量跑才发现，main 红了三轮。未决项见 `docs/dev/2026-10-06-neusser-time-series-econometrics-review.md`。
- **`doe/`（试验设计，1.39 起，来自 Joseph 2025 教材）**：`experimental/` 管 RCT（随机化、平衡、样本量），`doe/` 管“跑哪些因子组合”——析因 / 部分析因（`sp.factorial_design` / `factorial_effects` / `design_aberration`）、混料（`mixture_design`）、模型最优设计（`sp.doe_optimal`，**不要**和 RCT 样本量的 `sp.optimal_design` 混）、空间填充（`space_filling` / `design_augment` / `design_criteria`）、全局敏感性（`sobol_indices` / `morris_screening`）、代表点与数据划分（`support_points` / `split_data`）、序贯设计（`sequential_design`）。整个子包经 `_register_lazy` 惰性加载；模拟退火内核在 `_spacefill_core.py`（numba，首次调用搜索型 `space_filling` 时才 import，另有 numpy 慢路径）。参考实现（SFDesign / MaxPro / support / SPlit / twinning / sensitivity / FrF2 / DoE.base / AlgDesign / rkriging）全是 GPL / LGPL：只按论文实现、黑盒比输出。证据分三类，**不要混写**：给定输入的确定性量（判据、贪心增补、字长型、twinning、Sobol / Morris 统计量、给定超参的 kriging 预测）是逐位对齐；同一目标的两个优化器（精确最优设计、kriging 似然）比判据值；随机搜索（空间填充设计、支撑点、SPlit、Lenth 的模拟临界值）只是 S，写“判据值持平或更好”，不写 parity。四处已定位的约定：(1) `sensitivity::soboljansen` 的平方和除以 `2n-1`，Jansen 论文除以 `2n`，我们跟论文；(2) Lenth 的临界值默认用**模拟零分布**（作者自己的 `unrepx` 的做法），1989 论文的 `t(m/3)` 在 7 个效应时名义 5% 实际只有 2.1%，留作 `reference='t'`；(3) `rkriging` 插值时加 1e-6 的 nugget、最大化的是限制似然（与我们默认的 `likelihood='reml'` 一致，`'ml'` 是教材手写 `OKfit` 的 profile 似然）；(4) SFDesign 的 `uniform.crit` 返回的是开方后的 wraparound discrepancy，scipy 返回平方。`sp.doe_optimal` 的近似设计先在网格上跑乘法算法、合并支撑点、连续 polish，等价定理界不到 0.9995 时把最违反处的候选点加进支撑再 polish；精确设计是多起点交换 + 随机踢，**不保证全局最优**，靠 `efficiency`（相对近似最优）暴露坏的局部解。`sp.factorial_design` 的最小低阶混杂搜索是穷举，超出上限直接拒绝，不要换成贪心后还叫 minimum aberration。**`sp.gp_regress` 的教训**：长度尺度远小于数据间距时边际似然是平的（梯度为 0），优化器进去就出不来，重复观测的设计上曾返回“点与点之间回到均值”的拟合；现在先在长度尺度 × 噪声占比的粗网格上扫一遍、取最好的两个点作起点，不要去掉这一步。**第二轮（同日）**：`sp.sobol_indices` 默认改为中心化的 Saltelli 一阶估计量（惰性输入的误差 0.0001 对 Jansen 的 0.009，主导输入 0.008 对 0.005；必须按合并均值中心化，否则主导输入误差 0.012），对照 R 时显式传 `estimator='jansen'`。`sp.factor_importance` 是 FIRST（Huang-Joseph 2025）：集合 S 的条件方差 = 各观测在 S 上 k 近邻（含自身）的结果样本方差的均值，前向选择带 early dropping、重复 `n_forward` 次，再后向剔除，重要性是 `(v(S\i) - v(S)) / (Var(y) - v(S))`，数值数据上与 `first::first` 对到 1e-10；**只含分类因子的集合**里同格观测全是并列近邻，参考实现取树返回的任意 k 个，我们取格内方差，这是有意的不同，不要为对上 R 改回去。`sp.space_filling(qualitative=)` 的判据是 `1/∏(Δ²)·1/∏(d_k+1/L_k)²`（= `MaxProMeasure(p_nom=)`）。`sp.factorial_design` 另可出 Plackett-Burman（12 / 20 / 24，首行写在 `_PB_ROWS`，构造时现场验正交）、3 / 5 / 7 水平正则分式、L18（差集矩阵回溯搜索得到，只能比广义字长型，不能逐元素比）；二水平最小低阶混杂搜索在 `_factorial_core.py`（numba，穷举，上限按“组合数 × 2^m ≤ 4e9”）。`sequential_design(criterion='alc')` 不是默认，两条准则互有胜负。未做清单见 `docs/dev/2026-10-07-joseph-experimental-design-review.md`。
- **`interference/network_exposure.py`（2026-10 Wager 教材第 12 章重建）**：曝露概率对内置映射是**精确**的（Bernoulli 设计下由度数的二项分布给出），只有用户传入的 callable 映射才模拟；默认估计量是 Hájek（自归一化），`estimator='ht'` 才是 Horvitz-Thompson。方差是依赖图上的二次型 `v'Gv/n²`，`G_ij = 1{(N_i∪{i})∩(N_j∪{j})≠∅}`；`G` 一般不是半正定的，原始形式（`variance='hac'`）在重叠差时覆盖率只有 75–82%，小网络上还会为负，所以默认 `'hac_psd'`（取 `G` 的半正定部分，对随机化方差恒保守）。**不要**改回对角方差：旧实现对所有单位（含未处于该曝露水平的）求和 `Y²(1-π)/π²`、不含单位间协方差，在度数不均的网络上标准误是真实抽样标准差的十几倍，点估计因模拟概率被截断而有偏。证据是小网络上对全部分配的穷举（HT 无偏；`E[v'Gv/n²] = Var + δ'Gδ/n²`），不是模拟。不可能取到某个曝露水平的单位（如无邻居）会被排除并告警，`min_prob=` 是用户显式选择的重叠修剪。`sp.interference_test` 的焦点集必须与实现的分配无关（由种子抽取或用户给定），置换的是原假设允许变动的那部分处理：`no_spillover` 是非焦点单位；`no_higher_order` 是既非焦点也非焦点邻居的单位；`anonymous` 是在每个焦点的邻居集合**内部**置换（焦点的闭邻域必须两两不交，否则条件分布不是组内置换的均匀分布）。后两个检验固定了大部分分配，功效很低（试过的设计上约 0.2），文档里已写明，不要把「未拒绝」当证据。`design='complete'` 的曝露概率是超几何的、精确；方差沿用 Bernoulli 设计的估计量，只有模拟证据（默认区间覆盖 97–98%），没有定理。
- **`ope/mdp.py`（1.39 起）**：`sp.mdp_policy_value` 是教材式 (15.9) 的自归一化双重稳健估计，两个冗余函数（超额回报 Q、平稳分布比 ω）都由关于基的线性矩方程解出，Q 的水平方向不可识别所以基里去掉常数方向（`psi`），ω 用 `mean(ω)=1` 归一。表格基下估计值恒等于拟合 Bellman 方程的解，也等于拟合转移模型的平稳均值（1e-10，这是它的主要证据，改矩方程后先看这条恒等式）。标准误用「真值处求和项是鞅差」，**不做** HAC。`sp.marginal_policy_effect` 的求和项是向前累加的结果，相邻期重叠，**必须** HAC（滞后数至少为 `horizon`），每条轨迹末尾 `horizon` 期没有完整的向前和、直接丢弃。
- **`experimental/adaptive.py`**：Thompson 抽样的"最优臂概率"用求积算（两臂是闭式），**不要**换成后验抽样的频率——事后推断要用到的分配概率必须是实际使用的那个确定数。`sp.adaptive_inference(method='aw')` 是 `1/sqrt(e)` 加权的自归一化均值，两臂对比的方差是两臂方差之和（同一期不会给两臂都贡献结果，鞅差不相关）。没有概率下限时 Thompson 抽样不满足 Lindeberg 条件，覆盖率会掉到 88%，这是方法的性质（教材第 6 章末），函数在分配概率低于 `1/T` 时告警、对 UCB 这类确定性规则直接拒绝。`'mean'` / `'ipw'` 只为对比而留，`detail['valid_under_adaptivity']` 为 False。`sp.contextual_bandit` 的分配概率同样是精确的（各臂在该协变量处的后验均值与标准差喂给同一个求积函数）；**`sp.adaptive_inference` 对带协变量的数据不成立**（分配依赖协变量后，平方根加权的臂均值不再以臂均值为中心），文档指向用记录的概率做 IPW。
- **`matching/residual_balance.py`（`sp.residual_balance`）**：目标函数是 `(1-zeta)·Σγ² + zeta·‖失衡‖∞²`（从 balanceHD 的输出反推并核对过：zeta 越大失衡越小）。求解走对偶（每个协变量一个变量，L-BFGS-B，对偶变量以 `2a/n` 为单位，否则 zeta 接近 0 或 1 时病态），再用活跃集迭代精确解 KKT；活跃集不收敛时退回拟牛顿点。balanceHD 的 quadprog 路径没有解到最优（它的目标值在每个配置上都比我们高 1e-10 到 1e-8），所以权重只对到 1e-6，测试容差 2e-5 是参考求解器的精度，**不要**为了对上而放松我们的求解。balanceHD 是 GPL-3，只当黑盒。标准误以协变量为条件（样本平均效应），`outcome_model='none'` 时没有标准误（残差里还有协变量信号，且高维下纯加权估计有偏）。
- **`iv/mte.py`**：ATT / ATU 是系数的精确线性函数（`_aggregate_functionals`）：处理组是 `U < P` 的单位，所以对**全样本**按 `p_i` 加权积分 `∫_0^{p_i} MTE`，不是用处理组内的倾向得分分布（2026-10 前的做法，线性 MTE 下 ATT 1.22 对真值 1.30）。聚合量与曲线的标准误必须用系数的**完整**协方差（多项式各项系数强负相关，只取对角会把 ATE 标准误夸大三倍），解析标准误以拟合的倾向得分和协变量为条件，`bootstrap=` 才包含两者。bootstrap 必须重抽**修剪前**的数据并在每次抽样里重做修剪。
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

*最后更新：2026-10-07。过期信息会蔓延到每一次 agent 会话——持续维护本文件。*

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
