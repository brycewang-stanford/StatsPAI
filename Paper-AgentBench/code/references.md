# CausalAgentBench — 参考文献与现有 Benchmark 调研

> 目标：为「构建一个评测 Agent 做因果推断（causal inference）能力的 benchmark」做前期调研。
> 本文整理了截至 2026-05 与「因果推理 / 因果发现 / 处理效应估计 / 因果分析 Agent / 计量经济学 Agent / 数据科学与科学发现 Agent」相关的 benchmark、系统与综述。
>
> 阅读建议：**B 类（带数据的因果/统计推断 benchmark）**、**C 类（处理效应估计）** 与 **E/F 类（因果/计量 Agent 系统）** 与你的目标最接近，是核心对标对象；A、D 类是上游能力评测；G 类是更广义的 agent 评测范式可借鉴；H 类是论文/代码索引。
> 文末「对 CausalAgentBench 的启示」+「设计草案」总结了现有工作的空白与可落地的方案骨架。

---

## 执行摘要（TL;DR）

1. **命名空间已拥挤、需差异化**：`CausalBench` 至少有 3 个完全不同的工作（LLM 因果学习 / 多域因果推理 / GSK 单细胞因果发现），另有 `CauSciBench`、`InterveneBench`、`CausalTQA`、`CounterBench` 等。建 benchmark 前要先明确 scope，避免撞名撞定位。
2. **真正的蓝海是「带数据的端到端因果估计」**，不是纯文本因果推理。文本/符号类（Corr2Cause / CLadder / CRASS）已较饱和；带数据、要写代码做真实估计的（QRData / CauSciBench / InterveneBench / BLADE / Causal Agent-CausalTQA）才稀缺，且 SOTA 表现普遍很差（真实数据 MRE 50%+、方法选择准确率 ~49%、StatQA 仅 64.8%）→ 提升空间大、值得做。
3. **识别策略（identification）是计量/因果的灵魂，却少有 benchmark 系统评测**：现实难点不在「跑回归」，而在「能否识别？该用 DiD / IV / RDD / 事件研究 / 合成控制中的哪个？平行趋势/排他性是否成立？」。已有证据显示 LLM **过度依赖 OLS**、在方法间选择困难——这是明确的切入点。
4. **处理效应估计（CATE/ATE）有成熟的半合成评测传统**（IHDP / Twins / Jobs / ACIC，指标 PEHE/ATE error）可直接嫁接到 agent 场景，且有「真值已知」的天然优势用于自动判分。
5. **复现型 benchmark 给了现成的判分工程**（CORE-Bench / ReplicatorBench /「Read the Paper, Write the Code」：系数符号一致率、落入原 95% CI 比例）。
6. **差异化定位建议**：以**计量经济学家工作流**为蓝本、覆盖 **识别 → 估计 → 稳健性 → 解释** 全链路、带真实数据 + 可自动判分、强调 **agent 工具调用与自我校正**，把目前散落在 QRData / CauSciBench / InterveneBench / Econometrics AI Agent 的能力维度**整合并标准化**。

---

## 0. 速查对比表（最相关的核心 benchmark）

| Benchmark | 年份/会议 | 任务形态 | 是否带数据 | 规模 | 评测重点 | 链接 |
| --- | --- | --- | --- | --- | --- | --- |
| **Corr2Cause** | ICLR 2024 | 纯文本因果推断（相关→因果） | 否（符号/文本） | 200K+ | 形式因果推断、鲁棒性 | [arXiv 2306.05836](https://openreview.net/forum?id=vqIH0ObdqL) |
| **CLadder** | NeurIPS 2023 | 因果三阶梯 QA（关联/干预/反事实） | 否（符号→自然语言） | 10K | Pearl 因果引擎规则遵循 | [arXiv 2312.04350](https://arxiv.org/abs/2312.04350) |
| **QRData** | ACL 2024 Findings | 带数据的统计 & 因果定量推断 | **是**（数据表） | 411 题 | 数据驱动的统计/因果推理 | [arXiv 2402.17644](https://arxiv.org/abs/2402.17644) |
| **BLADE** | EMNLP 2024 Findings | 数据驱动科学的分析决策 | **是**（数据集） | 12 RQ（专家真值） | 变量/变换/统计模型选择 | [arXiv 2408.09667](https://arxiv.org/abs/2408.09667) |
| **StatQA** | NeurIPS 2024 D&B | 统计方法选择 & 假设检验 | **是**（数据表） | 11,623 | 列识别/方法选择/适用性 | [arXiv 2406.07815](https://arxiv.org/abs/2406.07815) |
| **Causal Agent / CausalTQA** | 2024 | 表格因果 QA（变量/边/图/效应 4 级） | **是**（表格） | ~1.4K | Agent 解决分层因果问题 | [arXiv 2408.06849](https://arxiv.org/abs/2408.06849) |
| **CauSciBench** | 2025 (preprint) | 端到端因果分析全流程 | **是**（真实+合成+教材） | — | formulation→变量→方法→实现→解释 | [OpenReview EO8mTLqDuT](https://openreview.net/forum?id=EO8mTLqDuT) |
| **InterveneBench** | 2026 | 政策干预推理 & 因果研究设计 | **是**（真实社科研究） | 744 研究 | 无给定因果图下的识别假设/study design | [arXiv 2603.15542](https://arxiv.org/abs/2603.15542) |
| **Econometrics AI Agent (MetricsAI)** | 2025 | 计量经济学专家级任务复现 | **是** | 课程+论文两套 | 计量任务规划/代码/复现 | [arXiv 2506.00856](https://arxiv.org/abs/2506.00856) |
| **Mining Causality（找 IV）** | 2024 | LLM 搜索工具变量 | **是**（经济场景） | 3 经典案例 | IV 发现/控制变量/DiD | [arXiv 2409.14202](https://arxiv.org/abs/2409.14202) |
| **CATE 经典（IHDP/Twins/Jobs/ACIC）** | 2017– | 处理效应估计（半合成） | **是**（真值已知） | 多套 | PEHE / ATE error | [arXiv 2107.13346](https://arxiv.org/abs/2107.13346) |
| **CausalBench (LLM)** | 2024 | 因果学习能力（相关/骨架/因果） | **是**（结构化，2–109 节点） | — | LLM 从数据学因果结构 | [arXiv 2404.06349](https://arxiv.org/abs/2404.06349) |
| **CausalBench (single-cell, GSK)** | ICLR 2023 | 基因调控网络推断 | **是**（单细胞扰动） | 200K+ 干预点 | 因果发现（生物） | [arXiv 2210.17283](https://arxiv.org/abs/2210.17283) |
| **DiscoveryBench** | ICLR 2025 | 数据驱动科学发现（含因果假设） | **是** | 264 真实 + 903 合成 | 多步假设搜索与验证 | [arXiv 2407.01725](https://arxiv.org/abs/2407.01725) |

---

## A. 因果推理 Benchmark（纯文本 / 符号 QA，无需操作数据）

评测 LLM 的「因果推理能力本身」，通常不提供原始数据表，而是给自然语言/符号化的因果问题。是 Agent 因果能力的上游基础。

### A1. 因果推断（相关→因果、Pearl 三阶梯）

- **Corr2Cause — Can LLMs Infer Causation from Correlation?** (Jin et al., ICLR 2024)
  - 首个测试 LLM「纯因果推断」的 benchmark：给定一组相关性/条件独立性陈述，判断变量间因果关系。200K+ 样本。
  - 关键发现：17 个 LLM 接近随机；微调模型在变量改名/改写扰动下崩溃 → 多为表面模式匹配。
  - [OpenReview](https://openreview.net/forum?id=vqIH0ObdqL) · 后续改进：[PC-SubQ prompting (2412.13952)](https://arxiv.org/abs/2412.13952)

- **CLadder — Assessing Causal Reasoning in Language Models** (Jin et al., NeurIPS 2023)
  - 10K 题，覆盖 Pearl 因果三阶梯（关联 / 干预 / 反事实）；符号问题经因果推断 oracle 引擎求解后翻译为自然语言。
  - 提出 CausalCoT 提示策略。任务对 LLM 高度困难。
  - [arXiv 2312.04350](https://arxiv.org/abs/2312.04350) · [GitHub](https://github.com/causalNLP/cladder) · [HF dataset](https://huggingface.co/datasets/causalNLP/cladder)

- **CausalBench (LLM Causal Reasoning)** — 注意有多个同名工作：
  - (a) *Comprehensive Benchmark for Causal Learning Capability of LLMs* (Zhou et al., 2024)：整合结构化数据、背景知识与 2–109 节点的真值图，三类任务——识别相关性、因果骨架、因果方向。[arXiv 2404.06349](https://arxiv.org/abs/2404.06349) · [GitHub](https://github.com/Rainy-ZhouYu/CausalBN-Bench) · [HF dataset](https://huggingface.co/datasets/CCLV/CausalBench)
  - (b) *Multi-Domain Benchmark for Evaluating Causal Reasoning* (Zhang et al., SIGHAN 2024)：从顶级经济/金融期刊抽取的因果关系构建，40,379 评测项，覆盖健康/环境/技术/法律/文化；四视角（因→果、果→因、含干预的两者）。最佳模型仅 57.6%。[ACL Anthology](https://aclanthology.org/2024.sighan-1.17/)

- **A Critical Review of Causal Reasoning Benchmarks for LLMs** (2024) — 对现有因果推理 benchmark 的批判性综述，指出多数 benchmark 可被表面捷径攻破，设计 benchmark 必读。[arXiv 2407.08029](https://arxiv.org/html/2407.08029v1)

### A2. 反事实推理（Counterfactual Reasoning）

- **CRASS — Counterfactual Reasoning Assessment** (Frohberg & Binder, 2021/LREC 2022) — 用「问题化的反事实条件句」测 LLM 能否推理事实的假设替代；早期、被广泛对比的反事实 benchmark。[arXiv 2112.11941](https://arxiv.org/abs/2112.11941)

- **e-CARE — Explainable CAusal REasoning** (Du et al., ACL 2022) — ~21K，因果问答 + 需给出**解释**，强调可解释因果推理（叙事型）。

- **CounterBench — A Benchmark for Counterfactuals Reasoning in LLMs** (2025) — 专门评测反事实推理，区分 Exposure/Covariate/Mediator/Outcome 四类变量在事实与反事实情景中的角色。[arXiv 2502.11008](https://arxiv.org/abs/2502.11008)

- **METER — Multi-Level Contextual Causal Reasoning** (2026) — 分层次评测上下文因果推理。[arXiv 2604.11502](https://arxiv.org/html/2604.11502)

- **CausalT5K** (2026) — 诊断「拒答/怀疑/谄媚/侦测-纠错/阶梯坍缩(rung collapse)」等可信因果推理失败模式。[arXiv 2602.08939](https://arxiv.org/html/2602.08939)

---

## B. 带数据的因果 / 统计推断 Benchmark（★ 与本项目最接近）

这类要求模型/Agent **真正读取数据、写代码、做统计估计**，而非只做文本推理——正是「Agent 做 causal inference」的核心。

- **QRData — Are LLMs Capable of Data-based Statistical and Causal Reasoning?** (Liu et al., ACL 2024 Findings) ★
  - 首个「带数据的高级定量推理」benchmark：411 题，配真实数据表（来自教材、在线课程、学术论文）。
  - 评测 NL 推理 / Program-of-Thought / ReAct / code interpreter。最强 GPT-4 仅 58%。
  - [arXiv 2402.17644](https://arxiv.org/abs/2402.17644) · [GitHub](https://github.com/xxxiaol/QRData) · [主页](https://xxxiaol.github.io/QRData/)

- **BLADE — Benchmarking Language Model Agents for Data-Driven Science** (Gu et al., EMNLP 2024 Findings) ★
  - 给定研究问题+数据集，agent 需产出完整分析：相关的**概念变量**、**数据变换函数**、**统计建模函数**；真值来自多位专家数据科学家的独立分析。
  - 核心发现：LLM 有世界知识但常停留在「基础分析」；能与数据交互的 agent 决策多样性更好但仍非最优——直接对应「分析决策质量」评测。
  - [arXiv 2408.09667](https://arxiv.org/abs/2408.09667) · [主页](https://blade-bench.github.io/) · [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.815/)

- **StatQA — Are Large Language Models Good Statisticians?** (Zhu et al., NeurIPS 2024 D&B) ★
  - 11,623 题，聚焦**统计方法选择 + 适用性判断**（含假设检验）；评测列识别、方法选择、适用性。
  - GPT-4o 最佳仅 64.83% → 「选对方法」本身就很难。与「因果方法选择」高度类比，判分范式可借鉴。
  - [arXiv 2406.07815](https://arxiv.org/abs/2406.07815) · [GitHub](https://github.com/HKUSTDial/StatQA) · [主页](https://statqa.github.io/)

- **Causal Agent based on LLM（含 CausalTQA benchmark）** (2024) ★
  - 提出因果 Agent（工具+记忆+推理）并建立 **CausalTQA**：表格因果 QA，4 个层级——变量级、边级、因果图级、因果效应级，约 1.4K 题。
  - 该 agent 在四级问题上准确率均 >80%。直接相关于「分层评测 agent 因果能力」。
  - [arXiv 2408.06849](https://arxiv.org/abs/2408.06849)

- **CauSciBench — Assessing LLM Causal Reasoning for Scientific Research** (2025) ★★
  - 首个覆盖**完整因果分析流水线**的 benchmark：问题 formulation → 变量选择 → 方法选择 → 统计模型实现 → 结果解释。
  - 数据来自已发表研究 + 合成场景 + 教材。o3 在真实数据上平均相对误差(MRE) 53.0%（合成 6.2% / 教材 30.6%）——真实场景仍远未解决。
  - 配套系统 **CAIS（Causal AI Scientist）** 在此评测。评测方法覆盖 RDD/IV/OLS/DiD/匹配/PSM/GLM，并发现 **LLM 过度依赖 OLS**、在方法间选择困难。[OpenReview EO8mTLqDuT](https://openreview.net/forum?id=EO8mTLqDuT) · [PDF](https://zhijing-jin.com/files/papers/2025_CauSciBench.pdf)

- **InterveneBench — Intervention Reasoning & Causal Study Design** (Shi et al., 2026) ★★
  - 首个端到端「真实政策干预下的因果研究设计」benchmark：744 篇同行评审社科研究；要求在**无给定因果图/结构方程**下推理政策干预与识别假设、选择识别策略与估计方法。
  - 最强模型 GPT-5.1 综合分仅 0.578、方法选择准确率 49.3%。提出 STRIDES 多智能体框架。
  - [arXiv 2603.15542](https://arxiv.org/abs/2603.15542)

- **CausalReasoningBenchmark — A Real-World Benchmark** (2026) — 面向真实世界的因果推理评测（搜索可见，建议核对最终版）。[arXiv 2602.20571](https://arxiv.org/pdf/2602.20571)

---

## C. 处理效应估计（CATE / ATE）Benchmark（★ 计量核心，真值已知）

机器学习/计量界评测「估个体/平均处理效应」的成熟传统。最大优势是**真值已知**（半合成或 RCT），天然适合做自动判分；可直接嫁接到 agent 评测。

- **经典半合成数据集**（CATE/ATE 评测事实标准）：
  - **IHDP**（Infant Health & Development Program）：747 样本，真实协变量 + 合成结果（A 线性 / B 非线性异质），标准 100 次重复。
  - **Twins**：1989–1991 美国双胞胎出生，估「体重对死亡率」的效应。
  - **Jobs**（NSW + PSID）：源自 RCT，估职业培训对就业的效应。
  - **ACIC 2016/2018**：数据分析挑战赛，行政健康记录协变量 + 半合成处理/结果，非线性机制。
  - 常用指标：**PEHE**（异质效应估计精度）、**ATE error**。
  - 综合入口：[Treatment Effect Estimation Benchmarks (IEEE DataPort)](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)

- **Really Doing Great at Estimating CATE? — A Critical Look at ML Benchmarking** (Curth et al., NeurIPS 2021 D&B) ★ — **批判 CATE benchmark 的必读**：半合成数据生成机制与基线算法的耦合会让评测产生误导。设计本项目判分协议前应读。[arXiv 2107.13346](https://arxiv.org/abs/2107.13346)

- **CausalPFN — Amortized Causal Effect Estimation via In-Context Learning** (2025) — 用「先验拟合网络/上下文学习」做摊销式因果效应估计，代表「foundation-model 式因果估计」方向，可作为 baseline。[arXiv 2506.07918](https://arxiv.org/pdf/2506.07918)

---

## D. 因果发现（Causal Discovery）Benchmark

侧重「从数据中恢复因果结构/图」，是因果流水线的上游环节。

- **CausalBench — Network Inference from Single-cell Perturbation Data** (GSK, ICLR 2023)
  - 大规模因果发现 benchmark：RPE1/K562 两个细胞系、20 万+ CRISPR 干预数据点；提出生物学意义的评测指标。配有公开挑战赛。
  - [arXiv 2210.17283](https://arxiv.org/abs/2210.17283) · [GitHub](https://github.com/causalbench/causalbench) · [Nature Comms Bio 2025](https://www.nature.com/articles/s42003-025-07764-y)

- **Auto-Bench — Automated Benchmark for Scientific Discovery in LLMs** (2025) — 含化学、社交网络等设定，模拟真实科学问题做自动因果发现。[arXiv 2502.15224](https://arxiv.org/abs/2502.15224)

- **Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts** (2026) — 成对因果方向判别评测。[arXiv 2601.15479](https://arxiv.org/pdf/2601.15479)

- **Large Language Models for Causal Discovery: Current Landscape and Future Directions** (综述) — LLM 用于因果发现的总览。[arXiv 2402.11068](https://arxiv.org/html/2402.11068v2)

---

## E. 因果分析 Agent / 系统（对标「被评测对象」与流水线设计）

这些是**系统/框架**而非 benchmark，但定义了「Agent 做因果」的能力边界与流水线，是设计评测任务时的重要参照（也可作为 baseline）。

- **Causal-Copilot — An Autonomous Causal Analysis Agent** (Wang et al., 2025) ★
  - LLM 驱动的端到端因果分析 agent：自动完成因果发现、因果推断、算法选择、超参优化、结果解释；集成 20+ SOTA 因果方法，支持表格与时序数据；HF Space 可在线试用。
  - [arXiv 2504.13263](https://arxiv.org/abs/2504.13263) · [GitHub](https://github.com/Lancelot39/Causal-Copilot) · [主页](https://www.charonwangg.com/project/copilot/)

- **Causal AI Scientist (CAIS)** (2025) — 给定自然语言问题+数据描述，用决策树式方法选择，执行并带自我校正反馈闭环，全自动因果分析；在 CauSciBench 上评测。[OpenReview EDWTHMVOCj](https://openreview.net/forum?id=EDWTHMVOCj)

- **CATE-B** — 开源 co-pilot，agentic 框架引导端到端处理效应(treatment effect)估计：因果发现+LLM 定边构建 SCM、识别稳健调整集、按因果结构选回归方法。

- **Confounder & Subgroup agents** — *LLM-based Agents for Automated Confounder Discovery and Subgroup Analysis in Causal Inference* (2025)。[arXiv 2508.07221](https://arxiv.org/pdf/2508.07221)

- **IV Co-Scientist** — 多智能体 LLM 框架，为给定 treatment-outcome 对**提出/批判/精修工具变量**，展示从大型观测库中发现有效 IV 的潜力 (2026)。[arXiv 2602.07943](https://arxiv.org/html/2602.07943v1)

- **Facilitating Adoption of Causal Inference Methods via LLM-Empowered Co-Pilot** (技术报告, 2025)。[arXiv 2508.10581](https://arxiv.org/abs/2508.10581)

- **Beyond Correlation: Towards Causal LLM Agents in Biomedicine** (2025)。[arXiv 2505.16982](https://arxiv.org/pdf/2505.16982)

- 其他被点名的因果 agent（多用合成场景、偏因果发现）：**LLM4Causal、CausalAgent、Causal-PFN、MAC**。

- **StatsPAI** — agent-native 因果推断 & 计量工具链（本机已接入的 MCP server：detect_design → preflight/recommend → fit → audit_result → 设计专属敏感性分析 → 验证引用），可作为 agent 工具/baseline 参考。

---

## F. 计量经济学（Econometrics）Agent 与 Benchmark

与「econometric agent」诉求直接对应。计量更强调识别策略（DiD/IV/RDD/事件研究）、稳健性与可复现性。

- **Can AI Master Econometrics? — Econometrics AI Agent (MetricsAI)** (Chen et al., 2025) ★
  - 基于 MetaGPT 的计量专用 agent：任务规划、代码生成执行、基于报错的反思、多轮迭代精修；两套数据集（课程作业 + 已发表论文）。
  - 结果：领域专用 agent 显著优于通用 LLM 与通用 agent；课程复现 >66%、论文任务 >40%。
  - [arXiv 2506.00856](https://arxiv.org/abs/2506.00856)

- **Mining Causality: AI-Assisted Search for Instrumental Variables** (Sukjin Han, 2024) ★
  - 经济学家视角（致谢 Guido Imbens）：用 LLM 通过叙事+反事实推理**搜索新的工具变量**，模拟经济主体的内生决策；多步 + 角色扮演提示。
  - 应用于三个经典案例：教育回报、供需、同伴效应；并扩展到回归/DiD 的控制变量发现。是「LLM 辅助识别」的代表作。
  - [arXiv 2409.14202](https://arxiv.org/abs/2409.14202) · [作者 PDF](https://sukjinhan.github.io/mining_causality.pdf)

- **EconEvals — Benchmarks & Litmus Tests for Economic Decision-Making by LLM Agents** (2025) — 采购、调度、定价等经济问题，测在环境中在线学习的能力。[arXiv 2503.18825](https://arxiv.org/pdf/2503.18825)

- **EconAgentBench — Economic Benchmarks for LLM Agents in Unknown Environments** — agent 在未知经济环境中行动/学习/博弈。[OpenReview bxZUPQbvp0](https://openreview.net/forum?id=bxZUPQbvp0)

- **EconWebArena — Autonomous Agents on Economic Tasks in Realistic Web Environments** (2025) — 强调对权威数据源的忠实性。[arXiv 2506.08136](https://arxiv.org/pdf/2506.08136)

---

## G. 更广义的数据科学 / 科学发现 / 复现 Agent Benchmark（评测范式可借鉴）

虽非纯因果，但其「给数据+任务→agent 写代码→执行→打分」的评测工程与你高度同构，是 harness/打分设计的参考。

- **DSBench — How Far Are Data Science Agents from Becoming Experts?** (2024) — 466 数据分析 + 74 建模任务（源自 ModelOff/Kaggle），长上下文、多模态、多表。[arXiv 2409.07703](https://arxiv.org/abs/2409.07703) · [主页](https://liqiangjing.github.io/dsbench.github.io/)

- **DiscoveryBench — Towards Data-Driven Discovery with LLMs** (ICLR 2025) ★ — 264 真实（6 领域，从论文反推发现流程）+ 903 合成任务；多步假设搜索与验证；最佳系统仅 25%。很多任务本质是因果/关系发现。[arXiv 2407.01725](https://arxiv.org/abs/2407.01725)

- **DataSciBench — An LLM Agent Benchmark for Data Science** (2025) — 数据科学全流程 agent 评测。[arXiv 2502.13897](https://arxiv.org/html/2502.13897v1)

- **InfiAgent-DABench — Evaluating Agents on Data Analysis Tasks** (2024) — 首个数据分析 agent benchmark，评测 34 个 LLM。[arXiv 2401.05507](https://arxiv.org/pdf/2401.05507)

- **IDA-Bench — Interactive Guided Data Analysis** (2025) — 多轮交互、源自 Kaggle notebook 的顺序自然语言指令。[arXiv 2505.18223](https://arxiv.org/pdf/2505.18223)

- **ScienceAgentBench** (OSU-NLP, 2024) — 数据驱动科学发现的语言 agent 严格评测。[主页](https://osu-nlp-group.github.io/ScienceAgentBench/)

- **CORE-Bench — Computational Reproducibility Agent Benchmark** (Princeton, 2024) ★ — 270 任务 / 90 篇论文（CS/社科/医学），三个难度，复现已发表结果；最难级最佳 agent 仅 21%。复现型评测的范式参考。[arXiv 2409.11363](https://arxiv.org/abs/2409.11363)

- **ReplicatorBench — Benchmarking LLM Agents for Replicability in Social & Behavioral Sciences** (2026)。[arXiv 2602.11354](https://arxiv.org/html/2602.11354v1)

- **Read the Paper, Write the Code — Agentic Reproduction of Social-Science Results** (2026) — 前沿 agent 重写社科论文分析代码：系数符号一致 >85%、落入原 95% CI >70%（48 篇可复现论文）。[arXiv 2604.21965](https://arxiv.org/html/2604.21965v1)

- **LongDA — Benchmarking LLM Agents for Long-Document Data Analysis** (2026)。[arXiv 2601.02598](https://arxiv.org/pdf/2601.02598)

- **统计能力专项**：**StatEval**（统计综合评测，[arXiv 2510.09517](https://arxiv.org/html/2510.09517v1)）、**StatLLM**（统计分析数据集，[arXiv 2502.17657](https://arxiv.org/html/2502.17657v1)）。

- **Evaluation and Benchmarking of LLM Agents: A Survey** (2025) — agent 评测方法学综述，搭 harness 前的方法论参考。[arXiv 2507.21504](https://arxiv.org/html/2507.21504v1)

---

## H. Awesome 列表 / 论文索引 / 综述

- **anpwu/Awesome-Causal-LLM** — LLM × 因果 论文合集（配套综述）。[GitHub](https://github.com/anpwu/Awesome-Causal-LLM)
- **chendl02/Awesome-LLM-causal-reasoning** (NAACL 2025) — LLM 因果推理 论文/代码/数据集合集。[GitHub](https://github.com/chendl02/Awesome-LLM-causal-reasoning)
- **zhijing-jin/CausalNLP_Papers** — 因果 × NLP 阅读清单（CLadder/Corr2Cause 作者维护）。[GitHub](https://github.com/zhijing-jin/CausalNLP_Papers)
- **wan19990901/Causal-LLM-Paper** — 知识图谱 + 因果 + LLM 论文合集。[GitHub](https://github.com/wan19990901/Causal-LLM-Paper)
- 综述：
  - *Causal Inference with Large Language Model: A Survey* (2024)。[arXiv 2409.09822](https://arxiv.org/pdf/2409.09822)
  - *Causality for Large Language Models* (2024)。[arXiv 2410.15319](https://arxiv.org/html/2410.15319v1)
  - *Improving Causal Reasoning in LLMs: A Survey* (2024)。[arXiv 2410.16676](https://arxiv.org/html/2410.16676v1)

---

## 对 CausalAgentBench 的启示（Gap Analysis）

综合上面工作，现有 benchmark 的空白与可切入点：

1. **「纯文本因果推理」已较饱和，「带数据的端到端因果估计」仍是蓝海。**
   Corr2Cause / CLadder / CRASS / CausalBench(A) 把文本/符号因果推理做透了；但真正稀缺的是「给真实数据集 + 研究问题 → agent 自主完成识别策略选择 + 估计 + 稳健性检验 + 解释」。QRData、CauSciBench、InterveneBench、BLADE、Causal Agent(CausalTQA) 是少数方向，且模型表现都很差（真实场景 MRE 50%+、方法选择 ~49%、StatQA 64.8%）→ 有充分提升空间，值得做。

2. **识别策略（identification）是计量/因果的灵魂，但少有 benchmark 系统评测。**
   现实研究中难点不在「跑回归」，而在「这个问题能不能识别？该用 DiD / IV / RDD / 事件研究 / 合成控制中的哪个？平行趋势/排他性是否成立？」InterveneBench 已切入「无因果图下的 study design」，CauSciBench 已观察到 **LLM 过度依赖 OLS**；建议把**识别策略选择 + 假设检验 + 稳健性自检**作为一等公民。

3. **评测应分层、可自动判分。**
   参考 CausalTQA 的 4 级（变量/边/图/效应）、CauSciBench 的 5 阶段（formulation→变量→方法→实现→解释）、StatQA 的「方法选择 + 适用性」。建议「分层 + 端到端」两套并行。

4. **借鉴复现型 + CATE 半合成的判分工程。**
   CORE-Bench / ReplicatorBench /「Read the Paper, Write the Code」给出「论文→代码→执行→自动比对系数」流水线（符号一致率、落入原 CI 比例）；CATE 半合成数据（IHDP/Twins/ACIC，PEHE/ATE error）给出「真值已知」的数值判分。两者结合可覆盖「真实复现」与「可控真值」两端。

5. **数据来源与防污染。**
   多数工作从已发表论文/教材构建——存在训练集泄漏风险（Corr2Cause 已证明扰动后崩溃；Curth 等指出半合成评测易误导）。建议：合成可控数据（已知真值 SCM）+ 真实数据 + 扰动测试（改名/改写/改分布）三结合，并保留 held-out 私有测试集。

6. **工具使用与 agent 能力是评测对象的一部分。**
   对标 Causal-Copilot / CAIS / MetricsAI / StatsPAI——agent 会调用 DoWhy/EconML/statsmodels。Benchmark 应允许（并评测）工具调用、多轮自我校正、报错反思，而不仅是单轮问答。

7. **差异化定位建议。**
   现有命名空间已有 CausalBench（≥3 个不同工作）、CauSciBench、InterveneBench、CausalTQA、CounterBench。CausalAgentBench 的差异点可定位为：**「以计量经济学家的工作流为蓝本、覆盖识别→估计→稳健性→解释全链路、带真实数据与可自动判分、强调 agent 工具使用与自我校正」**的统一评测，把目前散落在 QRData / CauSciBench / InterveneBench / Econometrics AI Agent 的能力维度整合并标准化。

---

## 设计草案（下一步骨架）

> 以下为基于上述调研的初步方案骨架，供讨论与迭代，不是最终设计。

### 1. 任务分层 Schema（融合 CausalTQA 4 级 + CauSciBench 5 阶段）

| 层级 | 能力 | 输入 | 期望产出 | 判分方式 |
| --- | --- | --- | --- | --- |
| L0 概念 | 因果概念/术语理解 | 文本题 | 选择/简答 | 精确匹配 |
| L1 变量角色 | 识别 treatment/outcome/confounder/mediator/IV | 数据描述 + 问题 | 变量分类 | 分类 F1 |
| L2 识别策略 | 判断可识别性 + 选 DiD/IV/RDD/PSM/合成控制 | 数据 + 问题 | 方法 + 假设清单 | 方法准确率 + 假设对错 |
| L3 估计执行 | 写代码跑出因果效应估计 | 数据 + 选定方法 | 点估计 + CI + 代码 | 数值（落入真值 CI / 相对误差）+ 代码可执行 |
| L4 稳健性与解释 | 稳健性检验 + 结论解释 + 局限 | L3 结果 | 稳健性表 + 文字结论 | 检验覆盖度 + LLM/规则评审 |

- **分层评测**用于诊断能力短板；**端到端评测**（L1→L4 串联，agent 自主走完）用于真实能力评估。

### 2. 判分协议（混合，尽量自动化）

- **数值判分**：半合成数据真值已知 → 用 PEHE / ATE error / 「点估计是否落入真值 CI」；真实复现题 → 系数符号一致率 + 落入原文 95% CI 比例（借 CORE-Bench / Read-the-Paper-Write-the-Code）。
- **结构判分**：方法选择（分类准确率）、识别假设判断（对/错/缺失）、变量角色（F1）。
- **执行判分**：代码可执行性、是否产出要求的产物（点估计/CI/图）。
- **解释判分**：稳健性检验覆盖度（规则核对清单）+ 结论合理性（LLM-as-judge + 人工抽检校准）。

### 3. 数据来源（三类 + 防污染）

- **合成/半合成（真值已知）**：自建 SCM 生成 + 嫁接 IHDP/Twins/ACIC，控制混杂/异质性/违背假设的程度。
- **真实复现**：可复现的计量/社科论文（参考 ReplicatorBench、48 篇社科复现集），比对已发表系数。
- **真实开放问题**：无标准答案、用专家 rubric 或 StatsPAI/审计工具核对（参考 BLADE 的专家真值思路）。
- **防污染**：变量改名/改写/改分布的扰动副本（验证非记忆）；保留 held-out 私有测试集；记录数据时间戳避免训练泄漏。

### 4. Agent 接口与 baseline

- 提供工具层（pandas / statsmodels / DoWhy / EconML / linearmodels；可接 StatsPAI MCP），评测 agent 的工具调用 + 多轮自我校正。
- Baseline：直接 LLM（CoT）、Program-of-Thought、ReAct/code-interpreter、Causal-Copilot、CAIS、MetricsAI。

---

*整理时间：2026-05-29 ｜ 来源：arXiv / OpenReview / ACL Anthology / NeurIPS D&B / GitHub / Nature / IEEE DataPort 等公开检索。部分 2026 年新预印本（InterveneBench、ReplicatorBench、CausalReasoningBenchmark、METER、CausalT5K 等）建议在正式引用前再核对最终版元数据。*
