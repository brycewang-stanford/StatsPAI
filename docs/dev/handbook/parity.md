# Parity 规则全文

> 从 CLAUDE.md §5.1 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；改动 Track A / 原始数据 parity、容差、golden 或 provenance trace 之前先读完本文件。新的细则直接写在这里。

### 5.1 Parity 规则（JSS 论文的核心断言，改动前必读）

规则全文在 `tests/r_parity/compare.py` 顶部 docstring、[`docs/dev/r_parity_tolerances.md`](docs/dev/r_parity_tolerances.md)、[`docs/guides/stability.md`](docs/guides/stability.md)；这里只列硬约束。

- **默认门槛**：同一份 CSV 字节、同一估计量，StatsPAI 与 R / Stata 参考的相对误差 **≤ 1e-6**（点估计与 SE 各自计）。这是 T2「同字节严格 parity」的定义，87 个 Track A 模块里绝大多数实际落在 1e-15 到 1e-9。
- **四个证据等级**：T1 已知真值回收（解析 DGP，无外部参考）；T2 严格跨语言 parity；T3 随机估计器的**种子复制等价**（固定数据、两侧各跑多个种子，比较种子分布；等价界按抽样 SE 事先写明，用 TOST 判定——单次各跑一次、拿抽样 SE 当 Monte Carlo 误差做分母，不算 T3；见 `tests/reference_parity/test_grf_seed_mc_equivalence.py`）；T4 有记录的约定差异或参考实现之间自身分歧（如 Basque SCM，R `Synth` 与 Stata `synth` 自己就不一致）。**只有 T2 可以写成「对齐 R / Stata」**，T3 / T4 必须按原等级表述，不得记为 parity 胜利。**达不到 T3 标准的随机比较**（单种子或少数几个种子、MC 容差、一个网格步长——如 CS bootstrap、fect/interflex CV、GRF 家族的已知真值筛查、HonestDiD C-LF）一律记为 **S（stochastic screen）**：只报告、不评级，不得写成 T3（2026-09 JSS v2 审稿指出论文把这几类都叫 T3）。
- **放宽只有两种合法理由**：两边计算的是有文档的不同量（自由度除数、ssc 小样本簇修正、解析 SE 对影响函数 SE），或方法论上不可能确定性一致。「算法相同但对不上」不是理由，是 bug。
- **放宽必须登记**：容差写入 `tests/r_parity/compare.py::TOLERANCES`（预注册预算，按模块一条）；≥ 5e-2 的条目在 `docs/dev/r_parity_tolerances.md` 打 A（机制明确）/ B（经验）/ C（未能解释）等级并写明机制；Stata 侧超预算的模块登记到 `compare.py::STATA_HEADLINE_GAP_EXCEPTIONS`；没有 Stata 参考的模块在 `STATA_SKIP_REASON` 写实测过的原因。**禁止**为了让某个测试通过而单独放宽一个模块。
- **Stata 与 R 共用一个预算**，不为 Stata 另设更松的门槛。
- **Track A 的 Python 侧必须跑原生实现**：`backend="honestdid"` / `backend="r"` 这类委托参考实现本身的调用，拿来和 R 比等于 R 对 R，**不得**作为 parity 行（10/21 号模块 1.31 前就是这样，JSS 审稿抓出）。官方 Python 端口（如 `bwselect="cct"` 的 rdrobust 包）只能作不参与 join 的旁证行；第三方 Python 库（linearmodels / statsmodels / pyfixest）服务的模块登记在 `compare.py::IMPLEMENTATION_PROVENANCE`，附录表打 † 标记。分类由 `scripts/trace_parity_provenance.py` 实跑调用追踪核验（`tests/test_parity_implementation_provenance.py`）。trace 绑定入口脚本、**估计路径上每个被执行的 StatsPAI 源文件**与已提交结果文件的 SHA-256：**改了任何 Track A 模块估计路径上的源码（不只是模块 .py）都要重跑受影响模块的 trace**（`python scripts/trace_parity_provenance.py NN ...`，全量约 15 分钟）；名单外的包记为 `unclassified`，要人工审查后加进名单。**ledger 有两个**：Track A（`tests/r_parity`）和原始数据（`tests/orig_parity`，重录要加 `--ledger orig`）；改了 `regression/ols.py`、`regression/iv.py`、`matching/`、`synth/scm.py`、`inference/ipw.py` 这类两边都执行的文件，两个都要重录。**重录之前先实跑模块、把结果与已提交的 `*_py.json` 做 diff**：trace 只绑定哈希，不检查已提交的结果还能不能从当前代码复现（末位抖动在 1e-9 复现容差内就 `git checkout` 还原；真变了就是冻结产物变了，按下文走重生成 + JSS 记录）。pre-push hook `parity-traces`（2026-10-05 起）检查两个 ledger 是否描述当前树——在此之前没有任何推送前检查，原始数据 ledger 在两条并行线下过期而无人察觉，LaLonde 的 `psm_att` 一行已经复现不出来。
- **每一行 SE 都被闸门盯着**：`tests/test_parity_harness_contract.py::test_every_r_se_row_is_inside_budget` 对每个 PASS 模块的**所有** R 侧 SE 行按注册预算门控；Stata 侧超预算的 SE 行必须在 `compare.py::STATA_SE_GAP_NOTES` 登记机制（且要能从我们自己的量重建出对方的数字）。GitHub-only 的 Stata 参考（如 `fect_stata`）在 do 文件里 `net install` 到 `tests/stata_parity/_ado_fect/`（已 gitignore），不要装进用户的 PLUS。
- **每一行超过 1e-6 的差距都要写明原因（2026-10-27 起）**：T2 模块里任何一行（点估计或 SE，R 或 Stata 侧）只要相对差 > 1e-6 且绝对差 > 1e-8，就必须在 `compare.py::ROW_GAP_NOTES` 登记为 `convention` / `reference_numerics` / `stochastic_reference` 之一；`convention` 必须给出 `pinned_by` 行，在对方约定下跑 StatsPAI 并对到 1e-6 以内（`tests/test_parity_harness_contract.py` 的三条测试：全部登记、无过期条目、钉住行在机器精度）。理由：模块预算只说"能漂多远"，不说"为什么不在机器精度"；Oaxaca 的 Stata SE 差 1.2%，在一个为 R 自助法写的 5e-2 预算下躲了几个月，76 号的 1.96 截断值被当成积分器噪声。**加新模块时**，跑完 `compare.py` 先看哪些行超 1e-6，逐行找机制，能钉就加约定行，不要直接放宽预算。
- **同字节 ≠ 同一组 double**：`%.16g` 不能无损还原 double。`_common.dump_csv` 会把 float 列换成正确舍入的解析值（Stata `import delimited` 读到的正是这些值），所以 Python 侧必须在 `dump_csv` 之后拟合同一个 `df`。R 的 `read.csv`（以及 `readr`）在 10–36% 的值上末位不同，这是 py–R 一致性约 1e-16 的下限，不是 bug。Stata 侧 `import delimited` 之前要 `set type double`（`_common.do` 里有；自己写的 do 文件别漏，orig 03 号曾因此在 float 上跑 `synth`）。
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
