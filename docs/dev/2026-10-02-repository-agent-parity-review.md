# StatsPAI 仓库审查：MCP、skills、agent-native 与 Stata/R 对齐

审查日期：2026-10-02（Asia/Shanghai）
基线：`d1025b2953ec45e5399d3f196b613176ba61767f`，本地包版本 `1.34.2`。

## 1. 总体判断与审查边界

StatsPAI 已经建立相当完整的 agent 使用基础：统一发现入口、schema、curated MCP profile、数据与结果句柄、结构化错误、可追溯的数据转换、结果协议、可安装 skill，以及区分 T1/T2/T3/T4/S 的数值证据体系。下一阶段的主要收益来自**把已有机制连接成可以持续验证的完整工作流，并使证据精确到本次调用的配置和输出**。

这是一份仓库架构与契约审查，不是对全部估计器的数学正确性认证。检查覆盖 `src/statspai/agent/`、registry/schema、结果协议审计、packaged skill 与旧 skill、validation scope、parity 测试与 CI、既有 roadmap；执行了定向测试与小型复现探针。没有运行全仓 pytest，没有重新生成全部 R/Stata golden，没有执行付费 LLM benchmark，没有核实远程 CI 的当前状态。本文的依赖和客户端建议是本仓库的设计建议，不是外部软件最新兼容性的断言。

开始时工作区已有 `examples/README.md` 和两个 notebook 的修改；本审查不归因、不修改这些内容。论文锚点和已发表 JOSS 归档保持不动。

结论分为三类：**确认缺口**有源代码或探针依据；**风险**表示从实现推导的可能故障，尚未做完整故障实验；**建议**表示值得新增的能力，不能据此宣称现有实现错误。P0 指容易误导统计解释或破坏服务的事项；P1 指直接影响真实 agent 的可靠性；P2 指效率、扩展与维护改进。

## 2. 本次实测基线

| 项目 | 结果 | 如何解释 |
| --- | --- | --- |
| 注册符号 | 1,274，87 个子模块 | registry 计数包含类；不能直接称为 1,274 个可执行工具 |
| 排除 `_kind == class` 的条目 | 954 | 仍不等于 MCP 安全工具数，也不等于独立估计器数；包含别名和辅助函数 |
| callable 有 `returns` | 797 / 954（83.5%） | 非空覆盖，不证明语义和真实返回值一致 |
| callable 有 `example` | 945 / 954（99.1%） | registry 的 example 字段，与 docstring Examples 覆盖是不同指标 |
| callable 有 `reference` | 407 / 954（42.7%） | 辅助函数不必有方法引用；不宜用全体函数作唯一考核分母 |
| callable 自身有 assumptions / failure_modes | 666 / 662 | 未把继承字段折算进去；不能与 card coverage 简单对比 |
| 配置级 evidence map | 14 个入口 | `validation_scope.SCOPE_FUNCTIONS`；与函数级 evidence 的覆盖不同 |
| agent card 审计 | 17 项计数 floor 全部通过 | 表明未退步，不表示已经达到足够的语义质量 |
| result protocol 审计 | 检查 302 个 result class，通过 | 有两个显式豁免；不是全部运行时返回对象逐一实例化验证 |
| 定向 pytest | 155 passed，1 deselected，2 warnings，14.66 秒 | warnings 为 panel 少簇提示；慢速 skill 属性验证本轮没有运行 |

执行命令（本仓库 venv）：

```bash
.venv/bin/python scripts/agent_card_coverage.py --check
.venv/bin/python scripts/result_protocol_audit.py --check
.venv/bin/python scripts/registry_stats.py --check
.venv/bin/python -m pytest \
  tests/test_mcp_protocol.py tests/test_mcp_stdio_subprocess.py \
  tests/test_mcp_tool_contract.py tests/test_registry_metadata_ratchet.py \
  tests/test_skill_package.py tests/test_skill_api_claims.py \
  tests/test_validation_scope.py -q
```

## 3. 优先级总表

| ID | 优先级 | 类型 | 改进事项 | 核心验收 |
| --- | --- | --- | --- | --- |
| M1 | P0 | 确认缺口 | 裁剪后的统计风险仍必须完整可见 | violations/warnings 不因预算变成看似干净的结果 |
| M2 | P1 | 确认缺口 | 输出预算从尽力裁剪改为明确契约 | 不可裁剪字段超限时显式报告；计量范围清晰 |
| M3 | P0/P1 | 风险 | 请求排队、超时后台线程与资源隔离 | 队列有上限；取消后不会无限新增后台计算 |
| A1 | P1 | 确认缺口 | 从字段存在率转向 agent metadata 语义准确率 | 核心方法的条件、failure mode、schema 经真实调用验证 |
| A2 | P1 | 确认缺口 | 本次配置证据进入标准结果与 MCP 输出 | 特定 SE 未验证时不能只显示函数级 T2 |
| A3 | P1 | 建议 | 统一诊断状态、后续步骤和复现规范 | 未运行诊断、失败诊断、不可用诊断明确区分 |
| S1 | P1 | 确认缺口 | packaged skill 的真实返回值测试进入持续闸门 | 不依赖旧归档 skill 来验证新 skill |
| S2 | P1 | 建议 | 用行为测试证明 skill 会正确完成和拒绝任务 | 成功率、严重统计错误、token/延迟可重复衡量 |
| R1 | P1 | 确认缺口 | parity 从函数/模块扩展到配置 × 输出 | 联合 vcov、CI、joint test、样本限制有独立证据 |
| R2 | P1 | 确认缺口/风险 | 外部参考重推导策略与 CI 宣称一致 | 清楚列出自动重跑、冻结、本地复现的边界 |
| R3 | P1 | 建议 | Stata 翻译按可执行语义和真实语料评估 | 不支持的选项显式拒绝，跨项目 holdout 测试 |
| G1 | P2 | 确认缺口 | 自动生成审计状态与路线图 | 已落地工作不再显示 pending；旧表不被当现状 |

## 4. MCP 审查

### M1. 输出裁剪可以删掉统计风险信息（P0，已复现）

依据：[输出预算实现](../../src/statspai/agent/_output_budget.py)，尤其 `PROTECTED_KEYS` 与 `apply_budget`；[MCP 调用返回路径](../../src/statspai/agent/mcp_server.py) 对 structured result 直接应用该预算。

当前保护 estimate/SE/CI/p-value、error 和句柄等顶层字段，但没有保护 `violations`、`runtime_warnings`、`degradations`。探针在 1,500 字节预算下把 100 条 violations 留成 1 条，把 100 条 runtime warnings 留成 11 条。裁剪记录是存在的，因此这不是无痕吞错；问题是 agent 需要另行正确解释 `truncated` 才能知道风险并不完整。

建议把数值表格和统计风险使用不同的裁剪策略。风险字段保留 `total/shown/omitted`、严重程度汇总、风险类别、以及读取完整详情的 URI；任何存在未展示风险的结果必须带 `risk_details_complete=false`。至少不可裁掉“有未完成诊断”“有识别违背”“有推断降级”这类摘要。若保护全部详细文本导致超限，应返回紧凑摘要加 handle，而不是无限扩大响应。

验收：对大量 violations、warnings、degradations 分别构造预算测试；最终解释必须仍提到每个风险类别和不完整状态。不能只验证 JSON 内有 `truncated`。

### M2. `max_output_bytes` 目前不是绝对上限（P1，已复现）

不可裁剪字段本身超限时，`apply_budget` 找不到候选就停止。探针 `{'estimate': list(range(1000))}`，预算 100，仍输出 3,904 字节，裁剪记录为空。这是对预算函数的边界复现，不表示常规标量 ATT 都会遇到它；向量结果和大 replay 等受保护值值得检查。

此外预算针对 structured object；同样 JSON 又出现在 text block，图像还单独 base64 编码。文档应明确这是 structured payload 的预算，不能当作整个 JSON-RPC 响应或 token 的预算。

建议新增 `budget_status`、`actual_bytes`、`budget_scope`；受保护结果过大时切换到摘要/handle，或明确 `unavoidable_overflow`。验收覆盖标量、向量、长 replay、中英文文本与图像，分别量 structured bytes 和完整响应 bytes。原始完整结果必须仍可读取。

### M3. 线程超时与排队缺少资源级约束（风险；部署为长期服务时 P0，本地使用 P1）

依据：`mcp_server.serve_stdio` 使用无显式容量的 `queue.Queue()` 和 `ThreadPoolExecutor.submit`；`_runner.py` 明确说明 Python 线程不能被杀死，超时只返回错误并请求协作退出。默认串行 worker 是合理的保护，但一次超时之后，未退出的后台数值线程可以与下一次估计同时运行。默认单 worker 不等于进程内永远只有一个估计在算。

这不是重新提出已经修复的 sampling 死锁。reader thread、stdout lock、真实 subprocess sampling 测试已经存在，本轮测试通过。

建议先加低成本的入站大小限制、排队数量上限、排队期限、orphan 阈值和 busy 错误，再评估进程 worker。重计算任务适合独立子进程：超时可以终止，warnings、matplotlib 与可变全局状态能隔离。句柄只在成功且请求仍有效时提交，防止取消后迟到的任务改变 cache 或导出文件。进程隔离需要处理不可序列化模型，宜按工具能力渐进开启。

验收：真实 subprocess 中运行无 checkpoint 慢任务，持续提交/取消，检查 RSS、后台活动数量、ping 延迟、EOF 退出时间、cache 状态；增加重复 request id、无效大报文和队列满载测试。本次只做静态风险分析，没有实测资源泄漏或取消后写文件。

### M4. 互操作性与真实链路应成为独立测试层（P1，建议）

已有协议和 subprocess 测试是强项。`tests/agent_eval/test_mcp_protocol_transcript.py` 使用进程内 `handle_request`，它证明 envelope 和分析链，但并不覆盖全部 stdio 生命周期。建议把至少一条完整链移到真实 subprocess：initialize → route → load → transform → fit → audit → follow-up → resource read → export。

增加可重复的客户端能力矩阵：不支持 sampling、只读 text、不消费 structuredContent、不同已支持协议版本、分页、断线重启、失效 handle。每次发布测试安装 wheel 后的 entry point，而不仅是在源码目录运行。这里只要求对仓库声称支持的组合建立测试，不建议未经需求就扩展 HTTP/OAuth 或远程多租户。

## 5. Agent-native 审查

### A1. Metadata ratchet 足以阻止倒退，尚不足以阻止误导（P1，确认缺口）

依据：`tests/test_registry_metadata_ratchet.py` 当前允许 placeholder 参数比例至 0.39、returns 覆盖至 0.80、option enum 覆盖至 0.66；card coverage 用 assumptions 或 inherits_from 判断。继承来源已有 provenance 标记，是正确做法，但“有继承”仍不代表当前变体的识别条件正确。

建议按 estimator/diagnostic/transform/export/utility/class/alias 分开考核；核心方法先要求参数说明、条件、返回 shape、枚举、失败与恢复路径完整。家族继承只作为基础，维护变体新增/撤销的条件。例如 fuzzy RD、sharp RD、cluster RD 不能共用一份未经细化的证据范围；动态 DML 不能由静态 DML 的描述自动获得时序识别保证。

新增语义测试：schema 中每个枚举值至少有有效调用或明确预条件；必填参数与真实签名相符；结果类与实际返回一致；card 的替代方法是否适用于同一 estimand。绝对计数 floor 保留，同时对核心方法建立“不允许新增空字段”的逐项闸门。不要为没有识别问题的导出函数强行填写因果 assumptions。

验收：先列出前 30 个高频入口逐项清单，每条条件标记 curated/inherited/inferred；不以全仓非空百分比代替这 30 项的准确性。

### A2. 将配置级证据接到标准响应（P1，确认覆盖缺口）

`sp.validation_scope` 已是很有价值的实现：按配置与输出区分 T1/T2/T3/S/B/T4，包含 vcov/joint_test，未知维度不默认算已验证。当前覆盖 14 个入口：callaway_santanna、causal_forest、dml、fast.feols、iv、ivreg、panel、psm、rddensity、rdrobust、regress、sdid、sun_abraham、synth。

风险在于发现阶段的函数级 evidence 与调用后的实际配置不是同一件事。现有 scope 示例已经展示 `iv(... robust='hc3')` 可为 `estimate_only`。agent 如果只读函数的 T2 标签，仍可能把该次 SE 当作对齐。无需再造 evidence 系统，应复用 scope，把调用后的摘要纳入 `to_dict(detail='agent')`、`result_card` 和 MCP envelope；其余方法明确 `scope_available=false`，不给出推断的覆盖结论。

验收：同一个入口切换受验证和未受验证的 VCE、权重、学习器、panel 约定，响应里的 `outputs.se.status` 必须变化。输出说明不得把 `estimate_only` 写成整体 aligned。新增 scope 按高频和推断风险排序，不追求给所有工具打统一数值等级。

### A3. 方法齐全还需要返回语义和诊断状态齐全（P1，建议）

结果协议审计通过并不意味着所有结果都有充足的 estimand、sample、inference、seed 记录；仓库明确选择缺失返回 null，这是比猜测更好的行为。两个显式豁免是 JAX-absent fallback stub 和 `HelpResult`，不应把它们包装成 302 个拟合结果全覆盖的宣传数字。

建议定义统一诊断条目：`status=passed/failed/not_run/not_applicable/unavailable`、`reason`、`scope`、`severity`、`artifact`。`violations=[]` 与 diagnostics 未运行必须能在标准摘要中区分。`next_steps` 统一为对象列表，带 function、arguments、所需 data/result、可执行前提，避免调用方兼容字符串与 dict 的混合类型。

随机性不宜为追求一致就机械统一为 42。优先统一 seed 字段和显式配置入口：requested_seed、effective_seed、seed_source、算法/后端版本；未设 seed 直接说明。默认 seed 若变更，按 correctness/reproducibility 变更流程处理。2026-09-28 roadmap 曾记录 seed 和 CrossValidationResult 的遗留项，本轮未逐个重现，实施前先确认当前状态。

### A4. `replay` 应区分调用说明与可独立复现脚本（P1，确认边界）

依据：`agent/_replay.py` 用 `data=data` 和 `result=result_<id>` 占位，数据源是注释；inline 只记录 hash，非 JSON 参数用 `<TypeName>`。因此 replay 有审计价值，但不一定能在全新 Python 进程直接执行。

建议保留当前 compact replay，同时增加可导出的 manifest：原始数据摘要/hash、转换链、有效参数、实际默认值、seed、包/后端版本、父结果依赖与产物。inline 数据与 session handle 应提供可选落盘的标准表格。另附 `replay_completeness=call_only/session_replayable/standalone`，不要仅凭一行字符串承诺跨会话复现。

验收：完成 transform → fit → follow-up 后关闭 server，在新进程仅凭导出 bundle 重跑，按方法性质比较结果或已定义随机等价，不只检查 replay 包含函数名。

## 6. Skills 审查

### S1. 当前 skill 已拆分，但完整验证仍偏向旧归档（P1，确认缺口）

当前维护源在 `src/statspai/agent/_skill/`，旧 `StatsPAI_full_data_analysis_skill/README.md` 已明确 Superseded，保留归档是合理的；不应再建议删除归档或从头拆分 monolith。

具体缺口：`tests/test_skill_api_claims.py` 加载旧目录 validator；完整属性测试标记 slow。`tests/test_skill_package.py` 对 packaged validator 跑的是 `--quick`。packaged validator 已能扫描 references，并有现代方法的属性 smoke，但这些完整检查没有由上述 packaged test 直接执行。本次 1 deselected 正是旧 skill 的慢速属性验证。

建议以 packaged validator 为唯一维护源，旧 validator 只保留历史兼容目的。增加明确的 packaged full test，按 extras 分为 base、fixest、plotting、neural 等能力组；核心无重依赖部分进入 fast gate，其余进入确定配置的 scheduled/release job。文档中的“每个签名、属性已检查”应反映实际测试范围：目前签名断言来自手工 `_SIGNATURE_CLAIMS`，不是自动验证全部 Markdown 调用的每个实参和值。

验收：在新 references 故意加一个存在但类型/参数错误的调用，测试应失败；在新返回值示例添加不存在属性，完整 packaged test 应失败。用 AST 或可执行 snippet 清单覆盖真实示例，不只 regex 查 `sp.<name>` 存在。

### S2. Skill 要用行为结果验收（P1，建议）

`tests/agent_bench/README.md` 清楚说明 900-trial 生产实验尚需预注册、预算和项目负责人授权，mock 验证的是 harness 而非真实模型能力。不要据 mock 表宣称 agent 成功率。此次不需要运行付费实验，后续可先建立低成本的 deterministic workflow fixtures 与小规模真实模型 smoke，两者分别报告。

建议任务覆盖：只导出表格、缺少 treatment 信息、staggered DiD + covariates、弱 IV、少簇、缺 extra、缓存驱逐、未验证 VCE、无法翻译的 Stata 命令。评价“正确停止/正确拒绝”与成功执行同样重要。先定义严重错误红线：把静态 TWFE 当 staggered ATT、弱 IV 只报常规 t 检验、未运行 diagnostic 说通过、T3/S 冒充 T2、遗漏 degradation。

固定模型快照、prompt、skill hash、schema hash、seed 与 scorer 版本；报告 task success、严重错误率、人工修复次数、工具轮次、token、p50/p95 延迟。失败样例作为 regression case，避免只发布一个平均分。

### S3. 缩短默认操作路径，按任务加载 playbook（P2，建议）

短 SKILL.md 已实现，应保持。下一步不是继续向入口塞方法名，而是把小任务与论文 pipeline 更清楚地区分；为 route → describe → minimal fit → inspect → export 准备一条常用路径，再按需要加载 econ/epi/ML/decomposition references。不同客户端的安装能力应显式描述，优先提供可移植 Markdown 入口和工具发现说明，不先承诺所有客户端原生支持某种 skill 格式。

## 7. Stata/R 对齐审查

### R1. 函数或模块 T2 不能自动覆盖全部输出与选项（P1，确认缺口）

强项已经存在：同字节输入、预注册容差、SE 逐行预算、native/third-party provenance、fixture hash lock、随机种子复制、参考分歧披露。无需再次搭建一套 parity framework。

主要改进是证据颗粒度。`docs/parity_object_coverage.md` 明确为 1.22.0 的手工时点审计，仅部分行在 1.27.0 更新，无 builder/drift gate。它自己承认可能低估当前覆盖，不能拿表里的未 pinned 条目直接认定当前方法仍未验证。

建议由当前 evidence ledger 生成“入口 × configuration × output”视图，分别列 estimate、SE、CI、vcov、joint test、nobs/sample mask、权重、聚合、预测、后估计量。优先 joint vcov，因为点估计和对角 SE 对上不能保证同时带、联合检验与 HonestDiD 的输入正确。不要依据旧表补重复测试，应先盘点目前已有的 vcov/joint_test artifacts。

验收：每个 claimed T2 输出能定位到真正执行该入口的测试、参考版本、输入 hash、容差与比较统计量；只有 headline 的模块不能给全部后估计量背书。函数级 evidence 保留作索引，运行时用 A2 的 scope 做约束。

### R2. 外部参考复现应有明确刷新策略（P1，确认边界与风险）

`.github/workflows/r-parity.yml` 重跑 17 个快安装 R 模块的 golden，对重依赖模块保持冻结；触发 paths 仅 `tests/r_parity/**` 和该 workflow，自身不包含 `src/statspai/**`。这不是所有源代码改动都重跑 R，也不是全套 R/Stata 每次 push 重推导。`parity-guards.yml` 的 Python reference suite 与 provenance/fixture 闸门是另一层，不能混为一谈。

该 R job 使用 `any::` 包声明，workflow 中没有直接执行 `renv::restore`。因此它更接近当前可安装参考版本的漂移探针，不能仅凭仓库有 renv.lock 就断言此 CI 在锁定环境里运行。

建议双轨：锁定环境复现保护既有证据；跟踪较新参考版本的 scheduled job 暴露上游变动。重依赖 R 和 licensed Stata 可采用定期/发版前人工或自托管重跑，并提交机器可读 run manifest。报告 last_reproduced_at、reference_version、platform、input/output/source hash、run_status、skip_reason。日期久不直接判错，但必须显式可见。

验收：文档准确列出自动重跑模块/触发条件；上游版本漂移与 StatsPAI 数值回归分别归因；失败保留 diff artifact。改变源代码时，受影响证据由路径 trace 决定是否重跑，不为文档改动重新跑所有外部软件。

### R3. Option fixtures 要纳入统一证据库存（P1，建议）

`tests/stata_parity/option_parity/README.md` 有意把四类选项 fixture 放在 Track A 非递归 glob 之外，避免伪造 headline 模块，这是合理设计。但它们不受那份 Track A inventory 的枚举，并不应成为没有统一 manifest 的第二套资产。

建议给 option/reference_parity/orig_parity 等各轨道统一登记 artifact 类型、生成器、消费者测试、参考环境、输入 hash、结果 hash、复现命令和版本变化策略，再分别应用适合该类型的验收规则。先核实已有锁定覆盖，不能由“不在 Track A”推导“完全无验证”。数值容差坚持共享预算和机制解释，绝不靠放宽容差消除未解释误差。

### R4. Stata 翻译应报告迁移语义，而非只有命令命中（P1，建议）

现有 `_stata_options.py`、lexer、script parser、`untranslated_options`、拒绝执行策略与 grammar invariants 已解决许多危险路径。`CLAUDE.md` 记录 12 篇语料修复前 12.9% 正确翻译、28% 静默错误；这是历史基线，不是本轮测出的当前命中率。

下一步优先跨项目 holdout：宏/续行、factor variables 和交互项、if/in、权重类别、cluster/SSC、xtset、样本标记、estimation sample、缺失值与 postestimation。报告命令可识别、参数可翻译、可执行、样本相同、estimate/SE/CI 对齐五层覆盖。用户命令按多项目频率与统计重要性选，不为一篇论文堆新 API。

验收：未知选项必须留下明确证据且 execution fail closed；生成 python_code 与 arguments 一致；支持的命令跑三侧样本与数值比较，不支持的命令用明确拒绝作为正确结果。R 迁移可先做概念映射表与配置说明，不必立即开发通用 R 语言 parser。

### R5. 从数值 parity 补到推断可靠性（P1，建议）

T2 回答“是否复现某参考”，不能单独回答识别正确、覆盖率正确或小样本可靠。仓库已有 coverage_monte_carlo、机制实验和森林种子研究，应与 evidence view 连通。对少簇、不平衡 panel、弱 IV、overlap 差、RD mass points、权重极端、学习器变化分别报告 coverage、size、bias 和失败率；把随机筛查 S 与种子等价 T3 保持分离。

验收：每个核心方法至少展示参考一致性和适用范围两条证据；失败/不收敛的模拟计入分母；不能删掉坏 replication 后只报告成功样本覆盖率。新大规模仿真先预算和预定义设计，不作为本次文档审查的隐含执行任务。

## 8. CI、文档与维护闭环

### G1. 已完成工作与 pending 文档有漂移（P2，确认缺口）

`plans/2026-09-28-agent-native-roadmap.md` 仍有 W1 CI patch “needs a user push” 状态；但当前 `ci-cd.yml` 的 fast gate 已包含四个 MCP/agent 套件，wheel smoke 已检查 MCP module help 和 schema snapshot。说明 roadmap 不能再作为未完成工作的唯一依据。CLAUDE.md 的“最后更新 2026-07-15”与后续 10 月内容也不一致。

建议用一个 machine-readable backlog 记录每项目标、实施入口、验收测试、状态和最近验证 commit，由它生成 roadmap 摘要；保留历史 worklog，但突出历史而非待办。审计报告、skill 入口、llms.txt、schema、capability table 共享统计定义，避免 registered symbols、callables、safe tools、estimators 混算。

### G2. 快速闸门应保住整条 agent 链（P1，建议）

不要用巨大全仓 suite 替代关键链路。快速闸门至少包含发现/schema、packaged skill 核心 claims、数据 transform/handle、真实 stdio 分析链、失效 handle、错误 envelope、预算下风险完整性、配置级 evidence。定期完整任务再覆盖 extras、Monte Carlo、外部软件重推导和较长 examples。

现有 fast job 包含 MCP 测试，但只列了部分套件；本报告所跑的 registry metadata、skill、validation scope 套件不全在该 fast 清单。它们可能由 full job 或其他本地闸门执行，实施前应做 test-to-job inventory，而不是宣称完全没有 CI。

## 9. 建议实施顺序（可拆为独立工作项）

### 第一阶段：先消除结果解释风险，约 1 周

1. M1 风险摘要不可裁剪；M2 超预算状态与摘要/handle 回退。
2. 把现有 validation_scope 接入 agent result/card/MCP，先保持 14 个入口，明确 unsupported。
3. 添加 packaged skill full 的核心方法验证，修正验证承诺的措辞。
4. 一条完整 stdio 链纳入 fast gate，增加 not_run 与 truncated 风险的解释断言。

交付：四组独立变更，每组有小型 failure case；不改估计器数值，不改论文表格和锚点。

### 第二阶段：资源与证据完整性，约 2–3 周

1. 请求大小/队列/orphan 上限和 busy/timeout 状态；按风险选择进程 worker 原型。
2. 前 30 个高频入口的语义 card 审查；由实际调用补 schema/return shape 断言。
3. 生成配置 × 输出 evidence inventory；补最缺的 vcov/joint test/sample parity。
4. 锁定参考环境与漂移环境分轨，明确 Stata 发版复现程序。

交付：资源压力报告、机器可读 evidence manifest、可生成的覆盖表。工期是规划估计，进程序列化或许可证环境会改变时间。

### 第三阶段：真实 agent 与迁移质量，约 4–8 周

1. 按既有预算/授权机制开展小规模真实模型 smoke；正式 900-trial 实验遵守已有预注册条件。
2. 冻结 Stata holdout corpus，逐层衡量翻译与数值覆盖。
3. 复现 bundle 在新进程恢复；统一 seed/diagnostic/next_steps 语义。
4. 以严重错误率与已验证配置覆盖率决定继续扩展哪个家族。

## 10. 长期指标与停止扩张条件

建议每次发版生成下列指标，不把所有东西压成一个“覆盖率”：

| 维度 | 指标 | 建议初始目标 |
| --- | --- | --- |
| MCP 正确性 | 完整 stdio 链、取消/缓存/预算失败案例 | 所有关键 fixture 通过 |
| 风险披露 | 严重风险类别在裁剪后遗漏率 | 0 |
| Discovery | 前 30 个入口的语义 card 审查通过率 | 100%，与全仓字段率分报 |
| Evidence | estimate/se/vcov/joint_test 各自覆盖配置数 | 先生成基线，再设置 ratchet |
| Skill | packaged 实际调用/属性验证覆盖 | 核心路径全覆盖，extras 单独列 |
| Agent 行为 | 严重统计错误、正确拒绝、任务成功 | 严重错误案例进入 blocking regression |
| 复现 | standalone bundle 新进程重跑成功率 | 核心 deterministic 链 100% |
| 翻译 | holdout 可执行率、样本一致率、静默丢参率 | 静默丢参 0，其余先量基线 |
| 参考环境 | 自动/人工重推导覆盖与最后成功时间 | 每条 evidence 有状态和来源 |

若新增估计器缺少真实调用、结果风险披露、配置级证据或基本边界测试，先补这些再扩充入口。StatsPAI 最值得巩固的优势是：agent 能知道该调用什么、哪些输出有证据、哪里失败，以及怎样继续分析。后续改进应围绕这四个可验证问题推进。
