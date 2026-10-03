# 2026-10-02 仓库审查：逐项状态

由 `python scripts/build_review_status.py` 从 `docs/dev/review_backlog.json` 生成，不要手改本文件。

对应 `docs/dev/2026-10-02-repository-agent-parity-review.md`。审查基线是 d1025b29 (1.34.2)，工作基于 e4fdfddc (1.35.0)。

共 36 项：已完成 **29**，部分完成 **2**，未做 **5**。“已完成”的每一项都列出守着它的测试；“部分完成”和“未做”写明缺什么、为什么、下一步。

## 已完成

| ID | 事项 | 落地位置 | 验收测试 | 备注 |
| --- | --- | --- | --- | --- |
| M1 | 风险字段最后裁剪，裁剪后留下 `risk_summary` 与 `risk_details_complete: false`；超过 20 条的 runtime warning 报总数 | `src/statspai/agent/_output_budget.py` | `tests/test_mcp_output_budget_risk.py` |  |
| M2 | `output_budget` 报告 `truncated` / `unavoidable_overflow`、`actual_bytes`、`scope`；文档写明预算只覆盖 `structuredContent` | `src/statspai/agent/_output_budget.py`<br>`docs/guides/agent_api.md` | `tests/test_mcp_output_budget_risk.py` |  |
| M3 | 准入限制：排队上限、orphan 上限、请求行大小上限、重复 request id 拒绝、排队期限 | `src/statspai/agent/mcp_server.py` | `tests/test_mcp_hardening_stdio.py`<br>`tests/test_mcp_isolation.py` |  |
| M3 | 数据加载纳入超时与取消（此前 `data_path` 的读取在受监督的 runner 之外，阻塞读会永久占住 worker） | `src/statspai/agent/mcp_server.py` | `tests/test_mcp_isolation.py` |  |
| M3 | 已超时或已取消的调用不能再提交句柄 | `src/statspai/agent/_result_cache.py` | `tests/test_mcp_isolation.py` |  |
| M3 | 资源压力实测 | — | `tests/test_mcp_isolation.py` | 进程隔离模式下连续三次超时：orphan 线程 0、残留子进程 0、ping 延迟 2 秒内、下一次调用正常、服务器 RSS 不增长（一次实测：预热后 165 MiB，三次超时后反而低 40 MiB）。线程模式下同一个调用会留下 orphan，测试里作为对照。只在 macOS 上量过；没有做长时间（小时级）的压力运行。 |
| M4 | 一条完整分析链跑在真实 stdio 子进程上 | — | `tests/test_mcp_stdio_chain.py` |  |
| M4 | 客户端能力矩阵：三个协议版本、未知版本、只读 text、无 sampling、发 cursor、重启后旧句柄、跳过 initialize、未知方法 | — | `tests/test_mcp_client_matrix.py` | 只覆盖服务器声称支持的组合；没有 HTTP / OAuth。 |
| A1 | 前 30 个高频入口的 card 用真实调用核对：必填参数、返回类型、每个枚举值、替代方法是否存在、字段来源（curated / inherited / 其它） | `scripts/agent_card_audit.py`<br>`docs/dev/agent_card_audit.md` | `tests/test_agent_card_audit.py` | 查出并修复 6 个缺陷，见“顺带发现”。 |
| A1 | 前 30 个入口的 card 文字通读：家族陈述继承到不适用变体上的问题 | `src/statspai/_family_cards.py` | `tests/test_agent_card_audit.py` | 通读 30 个 card 的 `assumptions` 与 `failure_modes`，修正 14 个：二元 `logit` / `probit` / `cloglog` 不再列多项模型的 IIA 和有序模型的比例优势；`callaway_santanna` / `sun_abraham` / `did_imputation` / `etwfe` 不再被建议“改用 CS 或 SA”，也不再带 TWFE 的失败模式；`rdrobust` 补上 fuzzy、kink、聚类的条件，并去掉把连续性框架说成 local randomization 的一句；点处理的 `ipw` 不再套用纵向 g 方法的表述；`dml` 的“√n CATE”改为目标参数。这是我读的一遍，不是领域专家的评审。 |
| A2 | 配置级证据进入标准结果与 MCP 输出；预算不得裁掉 evidence 块；自动注册的长尾工具也带 `result_card` | `src/statspai/result_card.py`<br>`src/statspai/agent/auto_dispatch.py` | `tests/test_mcp_stdio_chain.py`<br>`tests/agent_eval/test_red_line_scenarios.py` |  |
| A3 | 诊断状态统一：`result_card.assumptions.checks` 给出 `passed` / `failed` / `not_run` / `not_applicable` | `src/statspai/result_card.py` | `tests/test_result_card_checks.py` |  |
| A3 | `next_steps()` 统一返回对象列表（`CrossValidationResult` 是最后一个返回字符串的） | `src/statspai/crossval/_result.py`<br>`MIGRATION.md` | `tests/test_result_card_checks.py` |  |
| A3 | 种子参数名统一被结果卡识别；`did_imputation` / `gardner_did` 记录 bootstrap 种子；全量盘点脚本 | `src/statspai/_result_contract.py`<br>`scripts/seed_inventory.py` | `tests/test_seed_contract.py` |  |
| A4 | `replay_completeness` 标记 | `src/statspai/agent/_replay.py` | `tests/test_mcp_replay_completeness.py` |  |
| A4 | 可导出的复现 bundle：`statspai://result/<id>/bundle`，含数据哈希、转换步骤、调用、期望数值和可独立运行的脚本 | `src/statspai/agent/_replay.py`<br>`src/statspai/agent/_resources.py` | `tests/test_mcp_replay_bundle.py` | 内联数据或远程数据没有脚本（只记录了哈希），bundle 会说明原因。 |
| S1 | packaged skill 的 smoke fit 进入 pytest；代码块里的每个 `sp.*(...)` 调用按真实签名绑定；带 `**kwargs` 的调用按 schema、路由问题或转发目标核对 | `src/statspai/agent/_skill/validate_api_claims.py` | `tests/test_skill_package.py` |  |
| S2 | 确定性的红线场景：弱 IV、少簇、检查未运行、未验证的 SE、森林不算同字节对齐、缺设计输入、句柄失效、风险列表被裁、Stata 选项未翻译 | — | `tests/agent_eval/test_red_line_scenarios.py` | 检验的是输出里有没有无歧义的信号，不是某个模型会不会照做。 |
| S3 | 短路径 playbook：一次估计、一次检查或一张表的五步流程，以及三个应当停下的地方 | `src/statspai/agent/_skill/references/quick-path.md` | `tests/test_skill_package.py` | 五个代码块在测试里端到端执行。 |
| R1 | 入口 × 配置 × 输出的 evidence 清单，生成式，带漂移闸门 | `scripts/build_evidence_inventory.py`<br>`docs/evidence_inventory.md` | `tests/test_evidence_inventory.py` |  |
| R1 | 登记已有但未入册的证据：`sun_abraham`、`did_imputation`、`gardner_did`、`event_study`、`etwfe`（含默认调用的逐位相等挂接）、非线性 `etwfe` | `src/statspai/validation_scope.py` | `tests/test_validation_scope.py`<br>`tests/reference_parity/test_validation_entry_points.py` |  |
| R1 | joint test 的参考证据：`regress`、IV、面板 FE、logit、poisson | `tests/reference_parity/_fixtures/_generate_joint_test_stata.do`<br>`src/statspai/validation_scope.py` | `tests/reference_parity/test_joint_wald_stata_parity.py` | 本机 Stata 18 MP 双精度实跑 33 个 `test` / `testparm`。`sp.test` 在 `regress`（五种方差）、`ivreg`（三种方差，对应 `ivregress, small`）、面板 FE、logit、poisson 上的统计量、两个自由度和 p 值都对到 1e-10 以内。证据清单里 joint test 有参考的格子从 3 个变成 16 个。DiD 家族的联合预趋势检验、RD、DML 仍然没有 joint test 参考。 |
| R2 | R parity CI 的范围写准确；每周一的上游漂移探针 | `.github/workflows/r-parity.yml`<br>`tests/r_parity/R_ENVIRONMENT.md` | `tests/test_r_parity_ci_scope.py` | 每周定时任务还没在 CI 上实际跑过。 |
| R2 | 机器可读的 run manifest：每个模块每一侧的状态、参考版本、平台、输入输出哈希、是否由 CI 重推导、缺 Stata 侧的理由、上次重推导的提交与日期 | `scripts/build_reproduction_manifest.py`<br>`docs/reproduction_manifest.json` | `tests/test_reproduction_manifest.py` |  |
| R3 | option fixture 双精度重生成、被测试直接读取、哈希入清单；跨六条证据轨道的统一清单 | `scripts/evidence_track_manifest.py`<br>`tests/stata_parity/option_parity/README.md` | `tests/reference_parity/test_option_fixture_bindings.py`<br>`tests/test_evidence_tracks.py` |  |
| R4 | Stata 翻译 holdout：39 条按文档语法写的命令，Stata 18 MP 实跑的金标准，五层评分 | `tests/stata_translation_holdout/build_holdout.py` | `tests/test_stata_translation_holdout.py` | 首次评分：33 条可执行命令里 29 条五层全过，4 条被明确拒绝，0 条静默错误。四个缺口现已全部关闭，33 条全部复现 Stata，6 条应拒绝的命令全部拒绝。其中 `[fweight=]` 和 `xtreg, re` 是并行的另一条线在没看过这份语料的情况下补的，holdout 对它们是真正的盲测，并且通过了。 |
| R5 | 把已有的覆盖率、压力设计、size / power 结果接进 evidence 视图，失败的重复计入分母，附 Monte Carlo SE | `scripts/build_evidence_inventory.py` | `tests/test_evidence_inventory.py` | 4 个设计的覆盖率离 0.95 超过 2 个 MC SE（`rdrobust` 0.934、`sdid` 0.928、DML IRM 0.968、DML PLR 0.883），表里照实标出。 |
| G1 | 机器可读的 backlog 生成状态文档 | `docs/dev/review_backlog.json`<br>`scripts/build_review_status.py` | `tests/test_review_backlog.py` |  |
| G2 | fast gate 覆盖整条 agent 链 | `.github/workflows/ci-cd.yml` | — |  |

## 部分完成

| ID | 事项 | 已有的 | 还缺什么 | 下一步 |
| --- | --- | --- | --- | --- |
| M3 | 进程级 worker：`STATSPAI_MCP_ISOLATION=process`，超时或取消时杀掉子进程 | `src/statspai/agent/_process_worker.py`<br>`tests/test_mcp_isolation.py` | 只覆盖不依赖服务器状态的调用（无 `result_id` / `data_id` / `as_handle`）。带句柄的调用仍走线程 runner，因为拟合结果没有序列化协议。每次隔离调用要付一次解释器冷启动。 | 给拟合结果定义可序列化的最小形态后，再把 `as_handle` 调用纳入 |
| A1 | 30 个之外的入口：家族卡片陈述按成员限定范围 | `src/statspai/_family_cards.py`<br>`tests/test_agent_card_audit.py`<br>`tests/test_family_cards.py` | 通读了全部 30 张家族卡片（271 个成员）的 `assumptions` 和 `failure_modes`。52 条假设（分布在 24 个家族里）本来就点名了适用对象（“Cox: …”、“Frailty models: …”、“Romano-Wolf …”），现在限定到对应成员，220 个成员的卡片因此变短。41 条家族失败模式里有 28 条同样只关乎个别成员，142 张卡片原先带着不属于自己的那条（`kaplan_meier` 被告知比例风险检验拒绝时怎么办，`lincom` 被告知 Hausman 统计量为负时怎么办）。限定后有 55 个方法没有任何失败模式，为它们各写了属于自己的（`MEMBER_FAILURE_MODES`，31 条）。没做的：真实调用核对（必填参数、枚举值、返回类型）仍只覆盖 30 个；两个专门方法（`assimilative_causal`、`evidence_without_injustice`）限定后没有任何假设，我不熟悉到能替它们写的程度，留空了；新写的 31 条失败模式是统计常识层面的陈述，没有逐条对着实现验证触发条件，所以 `exception` 一律写的是“无，仅提示”。 | 给审查脚本每次加 10 个函数的调用；请领域作者过一遍新写的 31 条失败模式 |

## 未做，以及为什么

| ID | 事项 | 理由 | 下一步 |
| --- | --- | --- | --- |
| A3 | 把各估计器的默认种子统一成一个值 | 故意不做。259 个带种子的函数里默认值有 `None` 125、`42` 73、`0` 56 等；改默认值会改变已发表的带种子数字。审查本身也说不应机械统一。 | 若要改，逐个估计器走 ⚠️ correctness / MIGRATION 流程 |
| S2 | 真实模型的行为评测（成功率、严重错误率、token、延迟） | 需要付费模型调用、预注册和你的授权；`tests/agent_bench` 的 900-trial 设计已经写明这三个前提。mock 结果只能验证 harness。 | 先批一个小规模 smoke 的预算和模型快照 |
| R1 | 清单里空着的格子补参考（`etwfe` 的协变量 / `xvar` / 加权 / `agg_weights='unit'`，`event_study` 的其它窗口，`rdrobust` / `dml` / `psm` 的大部分网格） | 每一格都要在 R 或 Stata 里实跑一份新参考并登记容差，是逐格的 parity 工作。交错面板上的 TWFE 事件研究不该补。 | 按使用频率挑格子；每补一格重跑 `build_evidence_inventory.py` |
| R2 | 在 CI 里按 `renv.lock` 复现；重依赖 R 模块与 Stata 的定期自动重推导 | 需要自托管 runner（354 个 R 包，含仅 GitHub 发布的）和 Stata 许可。一个没法在 CI 上实测的 workflow job 我没有加：写了不跑等于没有，写了跑挂会挡住别人。 | 有 runner 之后加一个只在手动触发时运行的 job |
| R5 | 新的仿真：少簇、不平衡面板、RD mass points、极端权重、学习器变化 | 需要预先定义设计和计算预算；审查也写明不应作为隐含任务执行。 | — |

## 已核实无需改动

- **M3 的 sampling 死锁。** 审查已说明这是旧问题，reader thread 与真实 subprocess 的 sampling 测试都在，本轮未动。
- **R4 的 R 迁移概念映射表。** `docs/guides/migration-from-r.md` 已经是一份按包分节的 R → StatsPAI 映射（215 行）；其中出现的 82 个 `sp.*` 名字全部能解析。没有对每一行的参数写法逐条执行验证。

## 做的过程中查出的问题

- **macOS 上 fork 不安全。** 同一 pytest 进程里先跑过一次估计、再用 `subprocess.Popen` 默认方式启动 MCP 子进程，子进程会在 `exec` 之前段错误（返回码 -11）。原有的 `tests/test_mcp_stdio_subprocess.py` 也受影响，只是此前排在它前面的测试恰好没触发。两个子进程测试文件现在都走 `posix_spawn`（`close_fds=False`）。包内其他在估计之后起子进程的路径（R / Stata 后端）没有排查。
- **`sp.regress` 带 `**kwargs` 但会拒绝未知关键字。** 运行时是安全的；只是静态检查看不到，所以 skill 的调用检查对这类函数只能数位置参数。
- **option fixture 没有被测试读取。** `option_parity` 的 82–84 号 Stata fixture 没有任何测试打开，README 却写着 "consumed by"；测试里是硬编码数字。数字目前与 fixture 一致（90 个值全部对上），所以不是数值问题，是绑定缺失。已补。
- **六个 option fixture 是在 Stata 单精度下生成的。** 这些 do 文件不走 Track A 的 `_common.do`（那里强制 `set type double`），`import delimited` 把小数列存成 float，Stata 实际上是在第 8 位被舍入的数据上估计。此前记录的差距（`csdid` 相对 2e-5、`did_imputation` 2e-6）被解释成优化器和吸收容差，其实是这个。按双精度重生成后分别是 4.9e-13 和 6e-8。定位过程走的是 §5.1 决策树第 1 步：与稠密精确最小二乘解比较，StatsPAI 差 5e-16，Stata 差 9.7e-8。Track A 不受影响。
- **`did_imputation, unitcontrols(year)` 这一行达不到 1e-6。** StatsPAI 与精确解差 7.5e-12；Stata 差 1.4e-6，且它的 ATT 在不同 `tol()` 下非单调地漂移 6e-7。按决策树第 3 步记为参考精度披露（scope 里是 T4），不写成对齐；独立证据是 `TestExactSolution`。
- **`validation_scope` 的总体状态名不够用。** 上面这一行的估计量是 `disclosure`、SE 是 `reference`，总体状态落到 `stochastic_only`，而文档对这个词的定义是"没有任何主输出有 T1/T2 证据"。逐输出的状态是对的，总体标签有歧义。没有改分类法，记在这里。
- **`sp.etwfe` 的默认调用此前没有证据行。** 模块 17 和协方差测试跑的是 `panel=False`，默认是 `panel=True`。两者在面板上逐位相同（估计、SE、事件研究协方差），现在由 `test_etwfe_default_panel_call_is_bit_identical_to_module_17` 断言并据此挂接。
- **`etwfe` 的 headline SE 不是同字节对齐。** 对 `etwfe::emfx` 差 2.4e-6（参考侧用前向差分求 Jacobian），对 Stata `jwdid` 差 6e-4（K 约定）。在注册的 1e-3 预算内，scope 里记为 `disclosure`，总体状态 `estimate_only`。协方差测试的 R fixture 用 Richardson 外推后事件研究矩阵能到 1e-6；headline SE 若也用同样办法重生成参考，有机会升到 T2，本轮没做。
- **`data_path` 的读取不在超时之内。** 数据在受监督的 runner 启动之前就加载了。一个永不返回的读（FIFO、卡住的网络挂载、慢 URL）会永久占住唯一的 worker，后面的调用全部排队。做进程隔离测试时发现，已修。
- **自动注册的长尾工具没有 `result_card`。** 只有手工整理的那批工具会附结果卡；通过 registry 走的函数（包括森林家族）返回里没有配置级证据，也没有诊断状态。已补。
- **`sp.match` 的 schema 写错了参数名。** schema 把 `treatment=` / `outcome=` 标成必填，函数实际只接受 `treat=` / `y=`，照着 schema 调用会直接失败。card 审查查出，已改。同批查出：`sp.panel` 的枚举里有不存在的 `method='cre'`；`sp.sun_abraham` 的 `control_group` 枚举写的是 `notyettreated`（实际是 `lastcohort`）；`sp.aipw` 的 `estimand` 枚举多了不支持的 `ATC`；`sp.feols` 的返回类型被解析成 `list`；`sp.match` 的返回类型写成 `MatchEstimator`。
- **`sp.match(method='llr')` 在默认带宽下会崩。** bootstrap 的某个重抽样里局部线性权重之和为零，匹配结果的记账代码直接除零，抛出 `ZeroDivisionError`，整个调用失败。已改为把该处理单位的匹配结果记为缺失。点估计不受影响。
- **六种种子参数名不被结果卡识别。** `boot_seed`、`bootstrap_seed`、`rng_seed`、`wild_seed`、`halton_seed`、`rng`。用这些名字的函数，结果卡对其随机输出是否可复现一个字都不说。已补，并有测试挡新拼法。
- **`areg` 翻译出来的常数项不是 Stata 的 `_cons`。** 斜率和 SE 与 `areg` 完全一致；`Intercept` 是第一组的水平，`_cons` 是平均吸收效应处的截距。翻译说明原来只提了 SE 的自由度，没提常数项。holdout 查出，已补说明。
- **holdout 第一次真正派上用场。** 我补 `noconstant` 的同时，并行的另一条线也补了它，还补了 `[fweight=]` 和 `xtreg, re`。rebase 之后 holdout 对后两者是盲测：`xtreg, re` 的系数、常规 SE、聚类 SE、以及用正态分布的置信区间都与 Stata 逐位一致；`[fweight=]` 由 runner 按行展开，N 和 SE 也对。评分规则里有一处误判（把“只有 runner 能翻译”算成了静默错误），已修正。
- **`sp.panel(vce=)` 静默忽略它不认识的值。** `vce='robust'`、`vce='cluster'`、甚至 `vce='nonsense'` 都悄悄返回常规 SE，所有面板方法都一样。核对 `xtreg, re vce(robust)` 的翻译时发现。已改为明确报错并提示正确写法（`robust='robust'` / `cluster=`），记了 ⚠️ correctness 和 MIGRATION。
- **`sp.panel` 默认约定下，聚类后的 joint test 与 Stata 不同。** `sp.panel(..., method='fe', cluster='id')` 之后 `sp.test` 的 F 比 `xtreg, fe vce(cluster id)` 大 1.7%，并且参考分布用的是 F(q, N−K) 而不是 F(q, G−1)。60 个簇时 p 值是 0.0017，Stata 是 0.0040。传 `ssc='stata'`（或 `'fixest'`）后与 Stata 逐位一致。这是默认约定的问题，不是 bug：单个系数的这个差异此前已在 scope 里记为没有参考。我在 scope 里把它记成 `disclosure`，没有动默认值。簇少的时候默认 p 值偏乐观，是否把默认改成 `ssc='stata'` 需要你定，改了会变动已有数字。
