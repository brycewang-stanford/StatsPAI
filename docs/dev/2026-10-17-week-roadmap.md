# 2026-10-17 一周路线图（全库层面的改进）

Bryce 2026-10-17 授权：一周内完成"第一优先"到"第三优先"，再做"次一级但重要"。
worktree：`.claude/worktrees/week-roadmap`（分支 `wt/week-roadmap`），每个阶段闸门全绿后推 main。

本文件是进度锚点，上下文压缩后从这里恢复。

## 起点（2026-10-17 实测）

- 注册函数 1,569；估计量 941，有证据 939，跨语言 487。
- house style：SE 主题 83.5% 规范，遗留 21 处；covariates 遗留 40 处（`X`）。
- 有 `cluster=` 却不收 `vce=` 的函数 91 个；只收 `random_state=` 不收 `seed=` 的 89 个。
- 没有事后换标准误的入口。`regress` 的 provenance 不记 `weights=` / `hac_lags=`，按记录重拟合会静默丢参数。
- `validation_scope` 只覆盖 22 个函数。
- 真实模型评测（S2）按规则等 Bryce 的 OSF 预注册；本机无模型 API key。

## P1 统一推断语法

- [x] 1.1 `vce=` 覆盖所有产出标准误且有 `cluster=` 的函数（分类后约 70 个），共用一个解析器，不支持的种类响亮拒绝。
- [x] 1.2 `seed=` 覆盖只收 `random_state=` 的函数；house style 加 `seed` 主题与棘轮。
- [x] 1.3 `sp.vcov(result, vce=...)`：直接路径（OLS / IV / feols / 影响函数）+ 重拟合路径（先复现原结果再换方差，复现不了就拒绝）；`sp.vcov_compare`。恒等式测试：`sp.vcov(fit(), vce=s) == fit(vce=s)`。
- [x] 1.4 provenance 记全显式参数（`regress` 等漏记 `weights=`）。
- [x] 1.5 MCP 工具 `vcov_from_result`。

## P2 证据按配置细化

- [x] 2.1 盘点高频入口（前 50）的 scope 映射缺口。
- [x] 2.2 用本机 R / Stata 补参考，登记进 `validation_scope`。
- [x] 2.3 结果对象上直接报告"这个配置有没有参考证据"。

## P3 真实 agent 评测（预注册之前能做的全部）

- [x] 3.1 读 `tests/agent_bench` 现状。
- [x] 3.2 任务集、对照组（statsmodels / linearmodels）、确定性评分器。
- [x] 3.3 OSF 预注册草稿（假设、提示集哈希、分析计划），Bryce 提交。

## P4 次一级

- [x] 4.1 marginaleffects 式后估计：`predictions` / `comparisons` / `slopes` / `hypotheses`。
- [x] 4.2 稳定性分层（核心层冻结 API，`sp.help` 默认展示核心层）。
- [x] 4.3 Windows 进 CI 测试矩阵。
- [x] 4.4 polars / Arrow 原生输入。
- [x] 4.5 CLAUDE.md §11 迁到 `docs/dev/modules/`，CLAUDE.md 只留索引。

## 日志

### 2026-10-17 P1

- `accepts_vce`（`core/_vcov_spec.py`）接到 49 个只有 `cluster=` 的估计器；schema 里 `vce` 是独立属性（`x-statspai-vce-kinds`）。契约测试用探针替换被包函数，不需要数据（`tests/test_vce_grammar_contract.py`）。
- 自带 `vce` 词汇的估计器读 Stata 的 `cluster v`：fixest 家族（`stata_vce_to_fixest`）、rdrobust / rdbwselect（`_split_rd_vce`，含 `nncluster`）、`cluster_from_vce` 接到 9 个。
- RD：`rdbwselect` 及 6 个转调 `rdrobust` 的函数有了真正的 `vce=`（之前固定 nn）。
- `seed=` 别名 89 个；house style 加 `seed` 主题，10 个主题全部 0 遗留，棘轮冻结在 0（基线写全 0，空字典曾被当成"未冻结"）。
- `sp.vcov` / `sp.vcov_compare`：按 `@records_call` 记录的原调用重拟合（DataFrame 弱引用），先原样重放、点估计 1e-10 复现才换 `vce`。158 个估计器有调用记录。
- ⚠️ `regtable(vcov=)` 忽略权重、把 OLS 公式套到 logit（0.0078 对 0.276），改走 `sp.vcov`。
- ⚠️ `callaway_santanna(clustervars=常数)` 返回 6e-18 的 SE，改为 DataInsufficient。
- `regress` 的 provenance 补记 `weights=` 等。
- 剩下的长尾：约 70 个自带 `vce` 词汇的函数（`mswitch` 的 oim/opg 等）不读 `cluster v`；多数没有 `cluster=`，读不读无所谓。

### 2026-10-17 P2

- 新增 37 个配置维度映射（`_validation_scope_ldv.py` / `_te.py` / `_fe.py`）：logit / probit / poisson / nbreg / glm / tobit / qreg / heckman，aipw / ipw / tmle / ebalance / g_computation，feols / ppmlhdfe。`sp.validation_scope(result)` 由调用记录确定函数，逐维报告实际配置有没有参考证据，以及"只差一个维度"的近邻证据。
- 新参考：R `sandwich` 对似然模型的 HC0–HC3（`test_ml_sandwich_r_parity.py`）、Stata 补缺（`test_ldv_gaps_stata_parity.py`）、fixest 的 vcov × FE 网格（`test_feols_grid_fixest_parity.py`）、aipw 交叉拟合恒等式。
- 记为 T4 的约定差异：probit 的 HC0 / HC1 对 R（观测与期望信息阵）。
- 未做：`hdfe_ols` 没有带权重的参考、`xtabond` / `xtdpdsys` 没有映射；tmle 默认 Super Learner 那一格没有参考。

### 2026-10-17 P3

- CausalAgentBench v2（`tests/agent_bench/`）：50 个任务、14 个冻结数据文件（SHA-256 清单）、R 侧 94 个参考值与已知真值互验、6 个对照单元（StatsPAI / MCP / 纯 Python / R …）、沙箱会话、确定性评分（含捏造数字、虚构 API、越界工具三种幻觉）、预注册假设 H1–H5 的聚类 bootstrap。
- oracle 在 StatsPAI 单元 50/50 命中 R 金标准；纯 Python 与 R 单元的 oracle 各抽查通过。
- 真实模型调用被 `prereg.verify()` 挡住：没有 OSF DOI 的锁文件就不建 API 客户端。等 Bryce 决定：模型版本、effort、H4 的 token 口径（建议主口径计全部 token、次口径计未缓存 token）、重复次数与轮数上限（`BUDGET_RUNBOOK.md`）。
- 顺带修了 `sp.oaxaca` 缺三分解的问题（写参考解时发现）。

### 2026-10-17 P4.1

- `sp.predictions` / `avg_predictions` / `comparisons` / `avg_comparisons` / `slopes` / `avg_slopes` / `hypotheses`（`postestimation/marginaleffects.py`），默认值跟 R marginaleffects；`wts=True, df='model'` 时 `avg_slopes` 与 `sp.margins` 逐位一致。
- 对 R marginaleffects 0.32.0（黑盒）：估计值 1e-8 以内，标准误 2e-5（R 用数值雅可比）。黑盒对出的约定：`"sd"` 以均值为中心；平均后的比值是均值之比；0/1 变量在 `slopes` 里也是对比。R `glm` 的两处参考精度问题记在测试 docstring。

### 2026-10-17 P4.2

- 冻结层就是 `CORE_FUNCTIONS`（32 个，改为分组数据 `CORE_FUNCTION_GROUPS`）。签名快照 `tests/core_api_snapshot.json`，`tests/test_core_api_freeze.py` 与 pre-push hook `core-api-freeze` 拦截破坏性改动；有意的破坏要 `--ack` 并引用 MIGRATION.md 原文。
- 卡片与 schema 加 `api_contract`（frozen / deprecation-cycle / unstable / deprecated）。`sp.help()` 首屏先列核心层；`stable` 的说明从"签名锁定"改为如实的"先发 DeprecationWarning"。

### 2026-10-17 P4.3

- `.github/workflows/windows.yml`：每次改代码的推送在 windows-latest × 3.10 / 3.13 上跑约 700 条（上次全量 Windows 运行的全部失败项 + MCP stdio + 路径 + 核心估计量），2–3 分钟；`workflow_dispatch` 选 `full` 跑全量。
- 首次运行只剩 3 条失败，全部定位并修掉：release gate 的 Windows 路径、示例脚本向 cp1252 控制台打印非 ASCII、`sp.zinb` 在离散参数边界上的平台相关结果（改为按估计值判边界）。两条测试本身的 bug 也修了。详见 `docs/dev/2026-10-04-windows-test-gaps.md` 的更新。
- 未做：Windows 回到主 CI 矩阵要等一次干净的全量运行（约 5 小时）。

### 2026-10-17 P4.4

- `sp.*` 统一接受 polars / Arrow / `__arrow_c_stream__` / `__dataframe__`（`_frame_input.py`，`__init__` 安装）。pandas 参数以同一对象直通，所以该文件被两个 provenance tracer 排除；为此在包装后把 Track A 89 个、原始数据 12 个模块全部重跑核对。
- 换绑到所有已加载的 `statspai.*` 命名空间，保住 pickle 与 `sp.x is statspai.pkg.x`；parity 索引的身份比较改为看包装之下的原函数（惰性导出解析顺序无关）。
- 顺带：marginaleffects 系列在 parity 索引里登记为 `aligned`（4.1 那次提交漏跑了 `build_parity_index`）。

### 2026-10-17 P4.5

- CLAUDE.md §11 的 72 条领域约定逐字搬到 `docs/dev/modules/` 的 21 个文件，§11 只留索引表与 3 条跨领域约定；CLAUDE.md 157 KB → 63 KB。`tests/test_module_notes_index.py` 守索引与目录一一对应。
