# 2026-10-02 仓库审查：逐项状态

对应 `docs/dev/2026-10-02-repository-agent-parity-review.md`。每一行写明状态、落地位置和验收测试。状态只有三种：**已完成**（有测试守着）、**已核实无需改动**（审查时的缺口在基线之后已经补上）、**未做**（附理由和下一步）。

审查基线是 `d1025b29`（1.34.2）。本轮工作基于 1.35.0（`e4fdfddc`）。

## 已完成

| ID | 事项 | 落地位置 | 验收测试 |
| --- | --- | --- | --- |
| M1 | 风险字段最后裁剪，裁剪后留下 `risk_summary` 与 `risk_details_complete: false` | `src/statspai/agent/_output_budget.py` | `tests/test_mcp_output_budget_risk.py` |
| M1 | 超过 20 条的 runtime warning 不再无计数丢弃 | `mcp_server._run_tools_call` | 同上，`test_tools_call_reports_more_than_twenty_distinct_warnings` |
| M2 | `output_budget` 报告 `truncated` / `unavoidable_overflow`、`actual_bytes`、`scope` | `_output_budget.apply_budget` | 同上，含非 ASCII 文本与四档预算 |
| M2 | 文档写明预算只覆盖 `structuredContent` | `docs/guides/agent_api.md`、skill 的 `mcp-and-cli.md` | 无（文档） |
| M3 | 排队上限、orphan 上限、入站行大小上限、重复 request id 拒绝 | `mcp_server.serve_stdio` | `tests/test_mcp_hardening_stdio.py` 末四条 |
| M4 | 一条完整分析链跑在真实 stdio 子进程上 | `tests/test_mcp_stdio_chain.py` | 自身 |
| A2 | 同一入口切换 `robust`，`result_card.evidence.outputs.se` 随之变化 | 1.35.0 已接入（`result_card._evidence`） | `tests/test_mcp_stdio_chain.py`（`ivreg` 的 hc1 对 hc3） |
| A2 | 预算裁剪不得切掉 evidence 块 | `_output_budget._PROTECTED_PATH_PREFIXES` | `test_result_card_evidence_is_never_cut` |
| A4 | `replay_completeness`：`call_only` / `session_replayable` / `standalone` | `src/statspai/agent/_replay.py` | `tests/test_mcp_replay_completeness.py`（`standalone` 一条真的只凭字符串和文件重跑） |
| S1 | packaged skill 的 smoke fit 进入 pytest，不再只有旧归档有完整验证 | `tests/test_skill_package.py::test_bundled_gate_full_passes` | 自身 |
| S1 | 代码块里的每个 `sp.*(...)` 调用按真实签名绑定 | `src/statspai/agent/_skill/validate_api_claims.py::check_call_keywords` | `test_call_check_catches_a_wrong_call_in_a_reference`（五种注入） |
| S1 | "每个签名和属性都已检查"的措辞改为实际范围 | `SKILL.md` 与 9 个 reference 文件头 | 无（文档） |
| R1 | 入口 × 配置 × 输出的 evidence 清单，生成式，带漂移闸门 | `scripts/build_evidence_inventory.py` → `docs/evidence_inventory.{md,json}`；pre-push hook `evidence-inventory` | `tests/test_evidence_inventory.py` |
| R1 | 盘点已有 vcov 产物：Sun-Abraham 完整事件研究协方差已对 `fixest`，登记进 scope（新增 `share_variance` 维度） | `src/statspai/validation_scope.py` | `tests/test_validation_scope.py` 末条 |
| R3 | option fixture 进入统一清单：消费者测试、SHA-256、无人读取即失败 | 同上生成器 | `tests/test_evidence_inventory.py::test_option_fixtures_are_read_and_hashed` |
| R3 | 六个 option fixture 全部按双精度实跑重生成；消费者测试改为直接读文件；84 号的 7 个 Stata SE 首次被比较 | `tests/stata_parity/option_parity/*.do` 与 `results/`；`tests/reference_parity/test_option_fixture_bindings.py` | 同左，另有 `test_bjs_fe_covariates_parity.py::TestExactSolution` |
| R1 | `did_imputation`、`gardner_did` 建 scope 映射并登记已有的完整协方差证据 | `src/statspai/validation_scope.py` | `tests/test_validation_scope.py` 末三条 |
| R1 | `event_study`（TWFE）、`etwfe` 建 scope 映射；`etwfe` 默认调用通过逐位相等测试挂到模块 17 的证据上 | `src/statspai/validation_scope.py`；`tests/reference_parity/test_validation_entry_points.py` | `tests/test_validation_scope.py` 末三条 |
| G1 | roadmap 里已落地的 CI 项不再标 pending；`CLAUDE.md` 更新日期 | `plans/2026-09-28-agent-native-roadmap.md`、`CLAUDE.md` | 无（文档） |
| G2 | fast gate 覆盖整条 agent 链 | `.github/workflows/ci-cd.yml` | CI |

## 已核实无需改动

- **A3 的诊断状态区分。** `audit_result` 已经返回 `passed` / `failed` / `missing` / `not_applicable`，并给出 `summary` 与 `coverage`。全链路测试钉住了"未运行不算通过"。`not_run` 与 `unavailable` 两个状态、以及把 `next_steps` 统一成对象列表，仍然未做（见下）。
- **M3 的 sampling 死锁。** 审查已说明这是旧问题，本轮未动。

## 未做，以及为什么

| ID | 事项 | 理由 | 下一步 |
| --- | --- | --- | --- |
| M3 | 进程级 worker（超时可杀） | 需要处理不可序列化的结果对象和句柄提交时序，是一个独立设计 | 先按工具能力圈出可序列化的一批做原型 |
| M3 | 排队期限（queue deadline） | 有了队列上限后收益小 | 有真实长驻部署需求再做 |
| M3 | 资源压力实测（RSS、取消后写文件） | 审查本身只做了静态分析，本轮也没有做 | 与进程 worker 原型一起做 |
| M4 | 客户端能力矩阵（协议版本、分页、断线重启） | 只有一条链落地 | 逐个加到 `test_mcp_stdio_chain.py` |
| A1 | 前 30 个高频入口的语义 card 逐项审查 | 这是人工审查工作量，不是一次提交 | 先生成 30 项清单，标 curated / inherited / inferred |
| A1 | schema 每个枚举值至少一次有效调用 | 依赖上一项的清单 | 同上 |
| A3 | 统一诊断条目与 `next_steps` 对象化 | 改动 `to_dict(detail='agent')` 的形状，波及 schema 包和论文里的示例输出 | 论文改锚窗口之外做，走 MIGRATION |
| A3 | seed 字段统一（requested / effective / source） | 审查要求先确认 2026-09-28 遗留项的现状 | 先盘点再定 |
| A4 | 可导出的复现 bundle（新进程按 lineage 重建数据） | `replay_completeness` 只是标记，bundle 是新对外能力 | 论文周期内新增对外 API 需要强理由，暂缓 |
| S1 | 带 `**kwargs` 的 81 个调用的关键字检查 | 静态无法判定 | 可执行 snippet 清单，按 extras 分组 |
| S2 | skill 行为评测 | 需要预注册、预算和授权 | 先做确定性 workflow fixture |
| S3 | 短路径 playbook | 建议项 | — |
| R1 | `etwfe` 的协变量、`xvar`、加权、`agg_weights='unit'`、GLM 族；`event_study` 的其它窗口与交错面板 | 没有任何产物跑过这些配置（GLM 族有自己的 parity 文件，未接入映射） | 按清单里的空格逐项补参考，或确认不该补（交错面板上的 TWFE 事件研究本来就不该被认证） |
| R1 | reference_parity / orig_parity / external_parity 各轨的统一 manifest | 本轮只做了 option fixture 一轨 | 按同一生成器扩展 |
| R2 | 锁定环境与漂移环境分轨 | 改 `r-parity.yml` 的触发与 `renv::restore`，需要在 CI 上实测 | 单独一条线 |
| R4 | Stata 翻译 holdout 五层覆盖 | 需要冻结语料 | — |
| R5 | 推断可靠性仿真接入 evidence view | 新仿真要先预算 | — |
| G1 | machine-readable backlog 生成 roadmap | 本文件是手写的过渡形态 | 与 R1 的 builder 一起做 |

## 本轮顺带发现

- **macOS 上 fork 不安全。** 同一 pytest 进程里先跑过一次估计、再用 `subprocess.Popen` 默认方式启动 MCP 子进程，子进程会在 `exec` 之前段错误（返回码 -11）。原有的 `tests/test_mcp_stdio_subprocess.py` 也受影响，只是此前排在它前面的测试恰好没触发。两个子进程测试文件现在都走 `posix_spawn`（`close_fds=False`）。包内其他在估计之后起子进程的路径（R / Stata 后端）没有排查。
- **`sp.regress` 带 `**kwargs` 但会拒绝未知关键字。** 运行时是安全的；只是静态检查看不到，所以 skill 的调用检查对这类函数只能数位置参数。
- **option fixture 没有被测试读取。** `option_parity` 的 82–84 号 Stata fixture 没有任何测试打开，README 却写着 "consumed by"；测试里是硬编码数字。数字目前与 fixture 一致（90 个值全部对上），所以不是数值问题，是绑定缺失。已补。
- **六个 option fixture 是在 Stata 单精度下生成的。** 这些 do 文件不走 Track A 的 `_common.do`（那里强制 `set type double`），`import delimited` 把小数列存成 float，Stata 实际上是在第 8 位被舍入的数据上估计。此前记录的差距（`csdid` 相对 2e-5、`did_imputation` 2e-6）被解释成优化器和吸收容差，其实是这个。按双精度重生成后分别是 4.9e-13 和 6e-8。定位过程走的是 §5.1 决策树第 1 步：与稠密精确最小二乘解比较，StatsPAI 差 5e-16，Stata 差 9.7e-8。Track A 不受影响。
- **`did_imputation, unitcontrols(year)` 这一行达不到 1e-6。** StatsPAI 与精确解差 7.5e-12；Stata 差 1.4e-6，且它的 ATT 在不同 `tol()` 下非单调地漂移 6e-7。按决策树第 3 步记为参考精度披露（scope 里是 T4），不写成对齐；独立证据是 `TestExactSolution`。
- **`validation_scope` 的总体状态名不够用。** 上面这一行的估计量是 `disclosure`、SE 是 `reference`，总体状态落到 `stochastic_only`，而文档对这个词的定义是"没有任何主输出有 T1/T2 证据"。逐输出的状态是对的，总体标签有歧义。没有改分类法，记在这里。
- **`sp.etwfe` 的默认调用此前没有证据行。** 模块 17 和协方差测试跑的是 `panel=False`，默认是 `panel=True`。两者在面板上逐位相同（估计、SE、事件研究协方差），现在由 `test_etwfe_default_panel_call_is_bit_identical_to_module_17` 断言并据此挂接。
- **`etwfe` 的 headline SE 不是同字节对齐。** 对 `etwfe::emfx` 差 2.4e-6（参考侧用前向差分求 Jacobian），对 Stata `jwdid` 差 6e-4（K 约定）。在注册的 1e-3 预算内，scope 里记为 `disclosure`，总体状态 `estimate_only`。协方差测试的 R fixture 用 Richardson 外推后事件研究矩阵能到 1e-6；headline SE 若也用同样办法重生成参考，有机会升到 T2，本轮没做。
