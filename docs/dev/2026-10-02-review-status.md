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
| R3 | 82–84 号 fixture 与测试里的硬编码数字绑定；84 号的 7 个 Stata SE 首次被比较 | `tests/reference_parity/test_option_fixture_bindings.py` | 自身 |
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
| R1 | 其余入口的 vcov / joint test 证据 | 清单显示只有 `callaway_santanna`（1 格）、`sun_abraham`（2 格）有 vcov 参考，只有 `regress`（3 格）有 joint test 参考；`test_event_study_vcov_R_parity.py` 里 did2s / etwfe / TWFE 的矩阵已对上，但这些入口没有 scope 映射 | 给事件研究家族建 scope 映射，再登记 |
| R1 | reference_parity / orig_parity / external_parity 各轨的统一 manifest | 本轮只做了 option fixture 一轨 | 按同一生成器扩展 |
| R2 | 锁定环境与漂移环境分轨 | 改 `r-parity.yml` 的触发与 `renv::restore`，需要在 CI 上实测 | 单独一条线 |
| R4 | Stata 翻译 holdout 五层覆盖 | 需要冻结语料 | — |
| R5 | 推断可靠性仿真接入 evidence view | 新仿真要先预算 | — |
| G1 | machine-readable backlog 生成 roadmap | 本文件是手写的过渡形态 | 与 R1 的 builder 一起做 |

## 本轮顺带发现

- **macOS 上 fork 不安全。** 同一 pytest 进程里先跑过一次估计、再用 `subprocess.Popen` 默认方式启动 MCP 子进程，子进程会在 `exec` 之前段错误（返回码 -11）。原有的 `tests/test_mcp_stdio_subprocess.py` 也受影响，只是此前排在它前面的测试恰好没触发。两个子进程测试文件现在都走 `posix_spawn`（`close_fds=False`）。包内其他在估计之后起子进程的路径（R / Stata 后端）没有排查。
- **`sp.regress` 带 `**kwargs` 但会拒绝未知关键字。** 运行时是安全的；只是静态检查看不到，所以 skill 的调用检查对这类函数只能数位置参数。
- **option fixture 没有被测试读取。** `option_parity` 的 82–84 号 Stata fixture 没有任何测试打开，README 却写着 "consumed by"；测试里是硬编码数字。数字目前与 fixture 一致（90 个值全部对上），所以不是数值问题，是绑定缺失。已补。
- **`did_imputation` 选项测试的 ATT 相对误差是 2e-6 到 3.4e-6。** 该测试用绝对容差 1e-6 并注明是 lsqr 迭代解对 `reghdfe` 的差距；按 §5.1 的相对 1e-6 门槛它不是严格 T2。本轮没有改它的等级表述，只是记在这里。SE 的相对误差在 1.8e-7 以内。
