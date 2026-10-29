# CLAUDE.md — StatsPAI

> Claude Code / AI agent 在本仓库工作的指引。动手前请通读。本文件只放硬约束；每条规则的细节、来由和历史事故在 [`docs/dev/handbook/`](docs/dev/handbook/) 与 [`docs/dev/modules/`](docs/dev/modules/)，**做到相关的事之前先读对应文件**。

**工作语言**：用英文思考，用中文总结和讨论。

---

## 1. 项目定位

**StatsPAI** 的目标是**面向 Agent 设计，适合人类以及 agent 进行调用，并致力于超越 Stata、R、以及老的 Python 生态，成为全世界最好的因果推断与实证分析工具**。

1. **一次 `import statspai as sp`** 覆盖 DiD / IV / RD / 合成控制 / DML / Meta-Learner / 贝叶斯因果 / 因果发现 / 结构计量 / 面板 / 空间 / 时序。
2. **Agent 原生**：所有函数返回结构化结果、带自描述 schema，人和 Agent 用同一入口。
3. **数值对齐 Stata / R**：已有参考实现的方法，先对齐再扩展。

| | |
| --- | --- |
| 版本 | 以 [`pyproject.toml`](pyproject.toml) / `sp.__version__` 为准（不在此处写死） |
| Python | 3.10 – 3.13（Windows 见 `docs/dev/2026-10-04-windows-test-gaps.md`） |
| License | MIT（禁止引入 GPL / AGPL 依赖） |
| 作者 | Biaoyue (Bryce) Wang · <brycew6m@stanford.edu> · CoPaper.AI / Stanford REAP |
| 导入别名 | `import statspai as sp`，所有示例、docstring、测试一律 `sp.xxx` |

---

## 2. 仓库结构

```text
StatsPAI/
├── src/statspai/          # 主包（实时计数 `python scripts/registry_stats.py`）
│   ├── __init__.py          # 对外 API 入口
│   ├── registry.py          # 函数注册表（sp.help / sp.list_functions 依赖）
│   ├── help.py              # sp.help / sp.describe_function / sp.function_schema
│   └── <领域模块>/          # did iv rd synth dml metalearners …
├── rust/statspai_hdfe/    # HDFE Rust 后端（PyO3，单独发布）
├── tests/                 # pytest + reference_parity/ + r_parity/ + stata_parity/
├── docs/ · mkdocs.yml     # MkDocs 文档；docs/dev/ 是开发者笔记
├── paper.md · paper.bib   # JOSS（已发表，勿改）+ 全仓唯一 master bib
└── pyproject.toml
```

按领域的子模块分组、guide 清单、Memory 文件指针见 [`docs/dev/handbook/overview.md`](docs/dev/handbook/overview.md)。

---

## 3. 设计原则

1. **一次 import，统一 API**。能力通过 `sp.<function>` 暴露，不需要二级 import。
2. **Agent 原生**。`sp.list_functions()` / `sp.describe_function()` / `sp.function_schema()` 必须对所有对外符号有效。
3. **统一结果对象**。优先 `CausalResult`（或领域结果类），实现 `.summary()` `.plot()` `.to_latex()` `.to_word()` `.to_excel()` `.cite()`。
4. **家族方法用 dispatcher**。`sp.synth(method=...)` / `sp.decompose(method=...)` / `sp.dml(model=...)`：一个入口，多种估计器。
5. **先对齐 Stata / R**。`fixest` / `did` / `rdrobust` / `gsynth` / `MatchIt` / Stata 已有的，先对齐 API 和数值再扩展。
6. **证据优先**。数值正确性是底线，每个估计器必须有参考对齐或解析测试。
7. **失败要响亮**。假设违背 → 抛异常或 `warnings.warn` + 写入结果 `diagnostics`；不吞异常返回 `None` / `NaN`。orchestration / best-effort 路径（`workflow/` `smart/` `paper`）catch `Exception` 时**必须**调用 `statspai.workflow._degradation.record_degradation(target, section=..., exc=..., detail=...)`。**禁止** bare `except Exception: pass`。
8. **弃用走流程**。`DeprecationWarning` + [`MIGRATION.md`](MIGRATION.md) 登记 + 至少一个小版本缓冲期。
9. **引用零幻觉**。见 §10，这是红线。

---

## 4. 代码规范

细则（agent 卡片的三个自动来源、核心层冻结的 `--ack` 写法、polars 输入层的实现约束）见 [`docs/dev/handbook/api.md`](docs/dev/handbook/api.md)。

- 新对外函数**必须注册** → [`src/statspai/registry.py`](src/statspai/registry.py)。给已注册函数**加参数也要**在 `params=[...]` 补 `ParamSpec`，再 `python scripts/dump_schemas.py`（pre-push `registry-params` 会拦）。
- docstring 用 NumPy 风格（`Parameters` / `Returns` / `Examples` / `References`）；`References` 只写 bib key 或核验过的规范引用。
- **agent 卡片里能从代码读出来的不要手写**：`enum` 来自 `Literal` 或 `src/statspai/_enum_seeds.py`（不要凭记忆写取值）；失败模式来自带 `recovery_hint` 的 taxonomy 异常（改了 `raise` 跑 `python scripts/gen_source_failure_modes.py`）；中文词表 `_discovery_zh.py` 写 `\uXXXX` 转义 + 汉字注释。
- **核心层签名冻结**：`CORE_FUNCTIONS` 的 32 个签名由 `tests/test_core_api_freeze.py` 守着。新增带默认值的参数不算破坏；其他改动先走弃用流程再 `scripts/core_api_snapshot.py --write --ack`。
- **polars / Arrow 输入由 `_frame_input` 统一处理**：新函数什么都不用做，不要在估计器里写 polars 分支；改 `_frame_input.py` 时 pandas 输入不得走任何计算。
- 共享基元放模块级 `_core.py` / `_common.py`，不要在多个文件重复实现 kernel / WLS / sandwich / 影响函数。私有函数 `_` 前缀。单文件 ~800 行以内，按关注点拆。
- 核心依赖精简；`torch` / `jax` / `pymc` 归 extras 且**惰性 import**。
- 签名 house style 是棘轮（`scripts/signature_house_style.py --check`）：规范名 `id` / `time` / `treat` / `covariates` / `weights` / `vce`，旧拼写用 `@accepts_aliases`。

---

## 5. 测试

```bash
pytest                                        # 全量
pytest tests/test_did.py -q                   # 单文件
pytest tests/reference_parity/ -q             # 已知真值 / 解析 DGP 回收
python tests/r_parity/compare.py              # Track A：R / Stata 同字节 parity 汇总表
python tests/r_parity/verify_reproduce.py     # R 侧 golden 重推导（需 R）
python tests/stata_parity/verify_reproduce_stata.py   # Stata 侧（需 Stata 许可）
```

每条规则的来由见 [`docs/dev/handbook/testing.md`](docs/dev/handbook/testing.md)。

- 每个对外函数：**正确性测试 + 边界测试**各至少一个。新估计器有参考实现走对齐，没有走解析/仿真，容差就地标注理由。
- 覆盖率目标：核心（`did iv rd synth dml panel`）≥ 95%，整仓 ≥ 85%。核心棘轮**只在 nightly 跑**，给核心模块加代码后看次日 nightly（`python scripts/coverage_campaign.py gaps <模块>`）。
- 每条测试断言一个真命题（已知真值、恒等式、参考实现的数）。**「与另一个数不同」不是断言**；不为行数写冒烟测试。
- **判断退化不要和 0 比**，用相对阈值（如 `se <= 1e-10 * max|y|`），跨平台 BLAS 的舍入零不同。
- **改了行为后按「谁会调用它」找测试**：`grep -rlE '<参数名>|<函数名>' tests`，不要凭文件名挑。推送后看一眼 CI 的 Parity guards。
- `Path.read_text()` 必须传 `encoding="utf-8"`。

### 5.1 Parity 规则（JSS 论文的核心断言）

**改动 parity 相关的任何东西之前，先读 [`docs/dev/handbook/parity.md`](docs/dev/handbook/parity.md) 全文**（含「对不上怎么办」五步决策树、「R 与 Stata 不一致时跟谁」）。硬约束：

- **默认门槛**：同一份 CSV 字节、同一估计量，相对误差 **≤ 1e-6**（点估计与 SE 各自计）。
- **证据等级**：T1 已知真值、T2 严格跨语言 parity、T3 种子复制等价（TOST）、T4 有记录的约定差异或参考分歧、S 随机筛查（只报告不评级）。**只有 T2 可以写成「对齐 R / Stata」**。
- **放宽只有两种合法理由**：有文档的不同量，或方法论上不可能确定性一致。必须登记在 `compare.py::TOLERANCES`（≥ 5e-2 的在 `docs/dev/r_parity_tolerances.md` 打 A/B/C 级）。Stata 与 R 共用一个预算。T2 模块里 > 1e-6 的每一行都要在 `compare.py::ROW_GAP_NOTES` 登记机制。
- **Python 侧必须跑原生实现**，委托给 R 的 backend 不算 parity 行。
- **改了任何 Track A 模块估计路径上的源码，都要重录 provenance trace**（`python scripts/trace_parity_provenance.py NN ...`；两个 ledger，原始数据加 `--ledger orig`）。**重录之前先实跑模块，与已提交的 `*_py.json` 做 diff。**
- **golden 文件不得手改**，只能实跑重生成；复现性容差 1e-9。
- **新增 Track A 模块**：写完后跑 `tests/test_parity_harness_contract.py`、`python scripts/tier_a_fixture_lock.py`、`cd Paper-JSS && make audit`，按报错补齐，不要对着清单打勾。
- 对不上时**先假定是我们错了**，二分定位。禁止为了变绿放宽容差、手改 golden、删模块。

---

## 6. Rust 组件

[`rust/statspai_hdfe/`](rust/statspai_hdfe/) 单独发布为 PyPI 包 `statspai-hdfe`，在有 wheel 的平台是核心依赖。细则（谁用它、数值约定、CI、发布步骤）见 [`docs/dev/handbook/rust.md`](docs/dev/handbook/rust.md)。

- **唯一入口是 `statspai/_rust_backend.py`**，不要再写 `import statspai_hdfe`。入口语义一变就升 `lib.rs::API_VERSION` 并同步 `MIN_API_VERSION`。
- 结果必须与线程数无关（固定分块、按序相加）；Rust 与 NumPy 只在停止容差内一致，不要写逐位一致的断言。
- **Parity 一律在无 Rust 路径上生成与复现**（`STATSPAI_SKIP_RUST=1`）。
- 共线检测不能靠 `np.linalg.inv` 抛错，用 `fast/_validation.py::require_identified_regressors`。
- 发布 `statspai-hdfe` 只用 `~/.pypirc` token + 本地 twine，不用浏览器登录 PyPI，不配 trusted publishing。

---

## 7. 发布

完整步骤与历次踩坑见 [`docs/dev/handbook/release.md`](docs/dev/handbook/release.md) 与 memory `reference_pypi_publish.md`。PyPI 凭据在 `~/.pypirc`，**不要**提交仓库、不要写进 memory。

- 版本号一改就进入发版窗口：`python scripts/release_gate.py --fix-census` 再 `--check`。
- `pytest -q` 与 `pytest tests/reference_parity/ -q` 全绿；build 后把 sdist 清单对照 `git ls-files`，新的 gitignore 产物目录要在 `MANIFEST.in` 里 `prune`。
- **默认不发 GitHub Release**：发版 = TestPyPI + PyPI（本地 twine）+ 打 tag。Publish Release 会铸出永久的 Zenodo version DOI 并让 CI 再传一次 PyPI，只有用户在当前会话明确要求才发。发版总结写明「未创建 GitHub Release」。

---

## 8. 文档

- MkDocs（[`mkdocs.yml`](mkdocs.yml)），源在 [`docs/`](docs/)。改估计器时同步 [`docs/guides/`](docs/guides/) 里对应的 guide。
- 教程和长文档放 `docs/guides/`，不要写进代码注释。

---

## 9. Git 协作

细则（推送环境、并发争用点清单、派生产物生成顺序、远程 runtime 的 PR 规则）见 [`docs/dev/handbook/git.md`](docs/dev/handbook/git.md)。

- **🚨 提交闸门（最高优先级）**：2026-09-28 起常设授权，agent 判断合适时可直接 `git commit` + `git push origin HEAD:main`。「合适」= (1) 相关测试与 pre-commit / pre-push 闸门全绿；(2) 只暂存自己的改动，逐文件 add，绝不 `-A` / `-a`；(3) 改动完整自洽；(4) 触及 JSS 冻结产物时已在 `docs/dev/jss_review_changes.md` 记录。拿不准就先问。**仍须当次明确授权**：打 tag、发 PyPI / TestPyPI、GitHub Release、`--force` / 改写已推送历史、删除远程分支。本地 hook 会拦截 git commit / push，判断合适后前置 `STATSPAI_ALLOW_GIT=1`。每段工作结束用中文总结，列出推送的 commit。
- 直推 main，默认不开 PR。Commit 风格 `feat:` / `fix(<area>):` / `docs(<area>):` / `chore:`，摘要 ≤ 72 字符。
- **禁止** `--no-verify` / `--no-gpg-sign` / `--force`；不对已推送 commit `--amend`（hook 改了文件导致提交失败后，重新暂存并**新起**一次提交）。出错用 `git revert`。
- **push 前让 `python3` 指向本仓库 venv**（`source .venv/bin/activate`），否则 pre-push 闸门报 `No module named numpy`。不要改 hook 的 `entry`。

### 9.2 并发

- 另一个窗口在本仓库作业时，**各自开 worktree**：`git worktree add .claude/worktrees/<线名> -b wt/<线名>`，推送用 `HEAD:main`。worktree 里跑测试和脚本**必须带** `PYTHONPATH="$(pwd)/src"`。
- 只能留在主树时：提交前 `git status`，逐文件 / 逐 hunk 暂存。
- **计数行不要自己算**，跑 `python scripts/registry_stats.py --check`，它报什么就填什么。
- **`build_parity_index.py` 必须在 `dump_schemas.py` 之前。** rebase 撞到生成产物时取对方版本再重生，只有手工追加点（`CHANGELOG.md` / `CLAUDE.md` / `registry.py`）逐块合。

### 9.1 远程 runtime 回传结果走 PR

Colab 等远程 runtime 回传 benchmark 结果时不直推 main，按 [`docs/dev/handbook/git.md`](docs/dev/handbook/git.md) 末节的 payload / 命名 / 身份规则开 PR。

---

## 10. 引用与文献（零幻觉红线）

> 捏造一条引用，代价是整个包数值正确性的可信度。**这是最便宜的质量杀手，必须零容忍。**

### 四要素核验

任何**新增**引用（docstring / README / `CHANGELOG.md` / `MIGRATION.md` / `paper.md` / `docs/guides/` / commit message 均适用）落地前必须独立核验：

1. **作者**：完整名单、拼写、顺序
2. **年份**：正式发表年（预印本标注 `arXiv preprint, YEAR`）
3. **标题**：完整标题，含副标题
4. **期刊 / 会议 / 出版方** + **DOI 或 arXiv ID**

核验来源至少 **2 个独立渠道**（Crossref / doi.org / arXiv / 期刊官网 / Google Scholar 取其二）。**禁止凭 LLM 或训练语料记忆补全**，哪怕是你「非常确定」的论文。

全文（bib 子集抽取命令、包内副本的解析顺序、相关 hook）见 [`docs/dev/handbook/citations.md`](docs/dev/handbook/citations.md)。

### `paper.bib` 单一来源

- 根目录 [`paper.bib`](paper.bib) 是 **master**，全仓唯一可手工编辑的 bib。bib key 用 `lastnameYEARkeyword`。docstring / docs / 论文只引 bib key，不在多处手写同一条引用。
- 各篇论文的 bib（如 `Paper-JSS/manuscript/jss-bib.bib`）是 `tools/bib_subset.py extract` 派生的子集，**禁止手改**；要引新文献先核验进 master 再重新抽取。
- `src/statspai/paper.bib` 必须与根目录逐字节一致（`python tools/bib_subset.py packaged --sync`）。
- **条目必须 BibTeX 安全**：字段内禁止原始非 ASCII 字符（`{\O}rregaard`、`{\'e}`、`--`、`$R^2$`）。核验记录写在 `annote`，`note` 只放读者可见的短句。
- 同一文献只能有一个 key（`audit_bib_duplicates.py --strict`）。

### 交付前自检

- Commit 触及新引用 → message 里注明 "refs verified via `<source1>`, `<source2>`"。
- 陌生引用默认按「未核验」处理。
- **宁缺毋滥**：拿不准时写 `（citation needed）` 或不引。捏造一条引用比缺失糟糕一百倍。

---

## 11. 分领域须知

各领域的约定（参考实现的口径、不要改回去的写法、证据怎么做、未决项）在 [`docs/dev/modules/`](docs/dev/modules/)。**改某个领域的代码之前，先读它的文件**；领域交叉时两个都读。新约定写进对应文件，不写进本节。

| 文件 | 领域 |
| --- | --- |
| [`rd.md`](docs/dev/modules/rd.md) | 回归断点 |
| [`synth.md`](docs/dev/modules/synth.md) | 合成控制（含 gsynth） |
| [`did.md`](docs/dev/modules/did.md) | 双重差分与事件研究（CS、SA、BJS、dCDH、日期型时间列、staggered 共用检查） |
| [`decomposition.md`](docs/dev/modules/decomposition.md) | 分解 |
| [`panel_multilevel.md`](docs/dev/modules/panel_multilevel.md) | 面板、多层模型、前沿、动态面板 GMM、单位根 |
| [`bayes_mcmc.md`](docs/dev/modules/bayes_mcmc.md) | 贝叶斯（PyMC 与 `mcmc/`） |
| [`forest.md`](docs/dev/modules/forest.md) | 因果森林与 GRF 家族 |
| [`regression.md`](docs/dev/modules/regression.md) | 回归族（似然模型、GEE、GAM、glmnet、ctree、共线回归元） |
| [`iv.md`](docs/dev/modules/iv.md) | 工具变量与 MTE |
| [`formulas_covariates.md`](docs/dev/modules/formulas_covariates.md) | 公式写法、协变量与处理变量的输入约定 |
| [`treatment_effects.md`](docs/dev/modules/treatment_effects.md) | TMLE、IPW/AIPW、匹配、反驳检验 |
| [`dml.md`](docs/dev/modules/dml.md) | 双重机器学习 |
| [`causal_discovery_dag.md`](docs/dev/modules/causal_discovery_dag.md) | 因果发现与 DAG |
| [`structural.md`](docs/dev/modules/structural.md) | 结构模型（BLP、路径分析、生产函数、DDC） |
| [`stata_translation.md`](docs/dev/modules/stata_translation.md) | Stata 翻译（`sp.from_stata` / `sp.stata`） |
| [`timeseries.md`](docs/dev/modules/timeseries.md) | 时间序列与预测 |
| [`measurement_error.md`](docs/dev/modules/measurement_error.md) | 测量误差 |
| [`adverse_selection.md`](docs/dev/modules/adverse_selection.md) | 逆向选择（保险市场） |
| [`survival.md`](docs/dev/modules/survival.md) | 生存分析 |
| [`experiments_interference.md`](docs/dev/modules/experiments_interference.md) | 试验设计、干扰、自适应试验、离策略评估、实验报告工具 |
| [`agent_mcp.md`](docs/dev/modules/agent_mcp.md) | MCP 服务器与随包 skill |
| [`evidence.md`](docs/dev/modules/evidence.md) | 证据映射（`sp.validation_scope`） |
| [`plots.md`](docs/dev/modules/plots.md) | 图形编辑器 |
| [`datasets.md`](docs/dev/modules/datasets.md) | 数据集与测试的联网规则 |

跨领域约定（`weights_as=`、预测时拒绝未见过的因子水平、整数时间间隔、默认先验检查、并列、`.loc[y_df.index]` bug 等）见 [`docs/dev/handbook/cross_cutting.md`](docs/dev/handbook/cross_cutting.md)。新增回归入口、支持 `predict(newdata)` 的估计器、staggered 估计器、带默认先验的采样器、排序 / 近邻类统计量之前先读它。性能关键路径先 Rust，再 numba / JAX。

---

## 12. 不要做的事

- 不要悄悄改现有估计器的数值输出。必要的正确性修复 → CHANGELOG + MIGRATION 用 **⚠️ correctness fix** 标注。
- **不要凭记忆写引用**（§10）。
- 不要把 `torch` / `jax` / `pymc` 塞进核心 `dependencies`。
- 不要绕过 registry 添加对外函数。
- 不要在没对齐既有 dispatcher（`sp.synth` / `sp.decompose` / `sp.dml`）的情况下另起一个。
- 不要 mock 估计器的数值路径。参考对齐测试必须跑真 R / Stata 输出或公开论文数字。
- 不要吞异常返回 `None` / `NaN`。
- 不要把凭据、token、内部数据提交到仓库或 memory。

---

## 13. 论文与归档

细则（JSS 改锚的完整重推导步骤、审稿期冻结的登记规则、Track C 计时绑定）见 [`docs/dev/handbook/papers.md`](docs/dev/handbook/papers.md)。

- **论文不追版本**，只在投稿、返修、接收后定稿三个时点改锚。平时发新版论文一个字不改。不要为了论文的检查通过而发补丁版本。同一时间只能有一个会话改某篇论文的仓库。
- **JOSS 已发表**（2026-09-03，DOI `10.21105/joss.10604`）。`paper.md` / `paper.bib` 的已发表内容不要再改；对外引用用 `sp.citation()`。
- **JSS 审稿期冻结**：`tests/jss_review_freeze.json` 哈希了稿件读取的全部冻结产物。当前状态（2026-10-27）：未投稿，`"active": false`，锚点 1.40.2。启用后，改动任何冻结产物都要在 `docs/dev/jss_review_changes.md` 登记日期、commit sha、路径、对论文的影响。不要为了表格好看重生成冻结产物，不要改稿件。
- **Zenodo 只由 Publish GitHub Release 触发**（见 §7）。JOSS 的归档是 v1.23.0，version DOI `10.5281/zenodo.22085759`。

---

## 14. 速查与 Memory

```bash
pip install -e ".[dev]"                         # 开发安装
black src tests && flake8 src tests && mypy src # lint / format / type
python scripts/registry_stats.py --check        # 计数漂移检查（README / docs 计数行以它为准）
python scripts/registry_stats.py --table        # 重生 docs/stats.md 的按模块表
(cd rust/statspai_hdfe && maturin develop --release)   # Rust 后端
```

Memory 在 `~/.claude/projects/-Users-brycewang-Documents-GitHub-StatsPAI/memory/`（索引 `MEMORY.md`）。外部指针：[GitHub](https://github.com/brycewang-stanford/StatsPAI) · [PyPI](https://pypi.org/project/StatsPAI/)。

---

*最后更新：2026-10-30。过期信息会蔓延到每一次 agent 会话，持续维护本文件；新的细则写进 `docs/dev/handbook/` 或 `docs/dev/modules/`，这里只加一行指针。*
