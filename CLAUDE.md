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
| 版本 | `1.20.0`（见 [`pyproject.toml`](pyproject.toml)） |
| Python | 3.9 – 3.13 |
| License | MIT |
| 作者 | Biaoyue (Bryce) Wang · <brycew6m@stanford.edu> · CoPaper.AI / Stanford REAP |
| PyPI | <https://pypi.org/project/StatsPAI/> |
| 导入别名 | `import statspai as sp` —— 所有示例、docstring、测试一律 `sp.xxx` |

---

## 2. 仓库结构

```text
StatsPAI/
├── src/statspai/          # 主包：87 子模块 / 1,139 函数（实时数 `python scripts/registry_stats.py`）
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

- **因果 / 处理效应**：`causal did rd iv synth dml metalearners tmle bcf bayes causal_impact policy_learning ope dtr multi_treatment qte principal_strat proximal mediation mendelian assimilation bridge`
- **面板 / 结构**：`panel fixest structural frontier multilevel gformula gmm msm longitudinal`
- **空间 / 时序**：`spatial timeseries bartik`
- **因果发现 / ML**：`causal_discovery dag neural_causal deepiv conformal_causal matrix_completion bunching causal_llm causal_rl causal_text fairness`
- **设计 / 抽样 / 推断**：`matching power mht survey bounds dose_response interference selection censoring imputation transport target_trial epi survival surrogate`
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
pytest tests/reference_parity/ -q             # R/Stata 对齐
pytest tests/external_parity/ -q              # 论文数字对齐
```

### 新代码要求

- 每个对外函数：**正确性测试 + 边界测试**各至少一个。
- 新估计器：**有参考实现走对齐，没有走解析/仿真**，容差 `atol` / `rtol` 就地标注并说明理由。
- 核心估计器（`did iv rd synth dml panel`）目标覆盖率 **≥ 95%**；整仓目标 **≥ 85%**。
- **Windows CI 注意**：`Path.read_text()` 必须传 `encoding="utf-8"`（cp1252 默认会挂，见 commit `8755996`）。

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
3. `pytest -q` 全绿 + `pytest tests/reference_parity/ -q` 必过。
4. `rm -rf dist/ && python -m build && twine check dist/*`。
5. 干净 venv 装 wheel 冒烟测试。
6. `git tag vX.Y.Z && git push && git push --tags`。
7. `twine upload dist/*`。

---

## 8. 文档

- MkDocs（[`mkdocs.yml`](mkdocs.yml)），源在 [`docs/`](docs/)，`mkdocs serve` 本地预览。
- 当前 guide：`synth` / `choosing_did_estimator` / `choosing_iv_estimator` / `choosing_rd_estimator` / `choosing_matching_estimator` / `callaway_santanna` / `cs_report` / `honest_did` / `repeated_cross_sections` / `robustness_workflow` / `migration-from-r` / `mixtape_ch09_did`。相关估计器改动要同步对应 guide。
- JOSS 论文 [`paper.md`](paper.md) + [`paper.bib`](paper.bib)——对外 API 或项目范围变更时同步。

---

## 9. Git 协作

- **🚨 提交闸门（本节最高优先级，压过下面所有条款）**：默认**禁止** `git commit` / `git push`。必须等用户在**当前会话**里明确说"可以提交 / 可以 push"才放行；会话结束时也**不要**自动提交——先用中文总结、等用户指示。本仓库已配 `PreToolUse` hook 强制拦截（见 [`.claude/settings.json`](.claude/settings.json)），但**不要依赖 hook 兜底**，要主动遵守。
- **获得授权之后**才适用以下条款：默认分支 `main`，**直推 main**、默认不开 PR，除非明确要求（见 `memory/feedback_no_pr.md`）。
- Commit 风格：`feat:` / `fix(<area>):` / `docs(<area>):` / `chore:`，摘要 ≤ 72 字符。
- **禁止**：`--no-verify` / `--no-gpg-sign` / `--force`（除非明确授权）；对已推送 commit `--amend`。出错用 `git revert`。

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

### 交付前自检

- Commit / PR 触及新引用 → 在 message / 描述里注明 "refs verified via `<source1>`, `<source2>`"。
- Review / self-review 对任何陌生引用默认按"未核验"处理——要求 DOI 可点开、arXiv 可访问，或 Crossref 能直接搜到。
- **宁缺毋滥**：拿不准时写 `（citation needed）`、只引 bib key 占位、或干脆不引——**捏造一条引用比缺失糟糕一百倍**。

---

## 11. 分领域须知

- **`rd/`**：kernel / 局部多项式 / sandwich 走 `rd/_core.py`，不要重新实现。
- **`synth/`**：20+ 估计器全部经 `sp.synth(method=...)` 分发。新增方法要同时加到 dispatcher 和 `synth_compare()`。
- **`decomposition/`**：影响函数 / statistic-value / WLS 在 `_common.py`。RIF / FFL / inequality / Oaxaca 都委托到该文件。
- **`multilevel/` / `frontier/` / GLMM**：v0.9.3–v0.9.4 有含正确性修复的大重构——用户引用旧数值时主动提示。
- **`bayes/`**：默认 NUTS (`draws=2000 tune=1000 chains=4 target_accept=0.9`)；必带 `rhat` / `ess_bulk` / `ess_tail` / `divergences`；`rhat > 1.01` 或 `ess < 400` 发 `ConvergenceWarning`；HDI 94%（arviz 约定）。
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

*最后更新：2026-07-15。过期信息会蔓延到每一次 agent 会话——持续维护本文件。*

## 其它关键事项
- 目前的修改，默认不要影响到 joss 论文的审稿：https://github.com/openjournals/joss-reviews/issues/10604。万一有影响到，千万一定要特别提醒。
- **审稿期：自由迭代，不碰 Release。** 审稿进行中可以照常 commit / push / 发 PyPI 新版 / 打 tag —— 这些都**不触发 Zenodo 归档**、不打乱审稿（JOSS 锚的是 concept DOI `10.5281/zenodo.19933900`（永远指向最新归档）+ 仓库，不卡版本号）。**唯一开关是"Publish 一个 GitHub Release"**：只有它会触发 GitHub↔Zenodo 自动归档、铸出**永久不可删的 version DOI**。所以**审稿接收前不要随手发 GitHub Release**（PyPI 发版没关系）。等论文被接收那一刻再：定版 tag → 发 GitHub Release → 拿 version DOI → 回填 `paper`。当前 Zenodo 已归档到 **v1.17.0**（version DOI `10.5281/zenodo.20568882`）。
- **提交闸门（同 §9 顶部"提交闸门"，二者是同一条规则、最高优先级）**：默认**禁止** commit / push，必须等用户在本会话明确授权才放行；每次会话结束也**不自动提交**，先用中文总结、与用户讨论，等用户指示。
- 先不要 commit 和 push！
- 先不要 commit 和 push！
- 先不要 commit 和 push！

- 你的思考和推理过程请你使用英文进行。但是最后的总结请你使用中文和我讨论。

- 你的思考和推理过程请你使用英文进行。但是最后的总结请你使用中文和我讨论。

- 你的思考和推理过程请你使用英文进行。但是最后的总结请你使用中文和我讨论。