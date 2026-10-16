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

- [ ] 2.1 盘点高频入口（前 50）的 scope 映射缺口。
- [ ] 2.2 用本机 R / Stata 补参考，登记进 `validation_scope`。
- [ ] 2.3 结果对象上直接报告"这个配置有没有参考证据"。

## P3 真实 agent 评测（预注册之前能做的全部）

- [ ] 3.1 读 `tests/agent_bench` 现状。
- [ ] 3.2 任务集、对照组（statsmodels / linearmodels）、确定性评分器。
- [ ] 3.3 OSF 预注册草稿（假设、提示集哈希、分析计划），Bryce 提交。

## P4 次一级

- [ ] 4.1 marginaleffects 式后估计：`predictions` / `comparisons` / `slopes` / `hypotheses`。
- [ ] 4.2 稳定性分层（核心层冻结 API，`sp.help` 默认展示核心层）。
- [ ] 4.3 Windows 进 CI 测试矩阵。
- [ ] 4.4 polars / Arrow 原生输入。
- [ ] 4.5 CLAUDE.md §11 迁到 `docs/dev/modules/`，CLAUDE.md 只留索引。

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
