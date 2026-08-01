# DiD 两个月工作计划（2026-07-31 起）

> 目标：把 `sp.did` 家族从"方法齐全但数据形态受限"推进到"post-2021 主流 DiD 的完整可用实现"。
> 每个工作包（WP）的验收标准都是**对齐参考实现的数值证据**，不是"代码写完了"。

## 0. 现状基线

- **已 R 对齐（certified）**：`callaway_santanna`、`sun_abraham`、`did_imputation`、`honest_did`
- **有证据（validated）**：`gardner_did`、`etwfe`、`aggte`、`continuous_did`
- **无数值证据（api_stable）**：`harvest_did`、`design_robust_event_study`、`cohort_anchored_event_study`、`did_misclassified`、`lp_did`、~~`ddd_heterogeneous`~~、`did_timevarying_covariates`、~~`stacked_did`~~、~~`cic`~~、`overlap_weighted_did`
  （划掉的三个已通过 Track A 74/75/77 拿到跨语言证据，registry 里升为 `certified`）

参考环境（已就绪）：R 4.5.2 + `did` 2.3.0 / `etwfe` 0.6.2 / `fixest` 0.14.0 / `did2s` 1.2.1 /
`didimputation` 0.5.1 / `HonestDiD` 0.2.8 / `staggered` 1.2.2 / `fect` 2.4.1 / `DRDID` 1.2.3。
`pretrends` 0.1.0（GitHub，非 CRAN）、`triplediff` 0.2.4、`DIDmultiplegtDYN` 2.3.4
（需 `options(rgl.useNULL=TRUE)` + r-universe `polars` + `rlang >= 1.2.0`）现均已装好。

### 参考值生成的两个坑（已踩过，务必记住）

1. **`data.table::fread` 返回整数列**。`did::att_gt` 内部把 never-treated 重编码为 `Inf`，
   整数列会静默截断成 `NA`，never-treated 组消失，12 个 ATT(g,t) 单元变成 6 个，
   汇总值全错。**生成参考值一律从 R 包自带数据读，或先 `as.numeric()`**。
2. **`didimputation` 用 `:=` 原地改数据**。同一个 R 脚本里先跑它再跑 `att_gt`，
   后者拿到的是被污染的数据。**每个估计量用独立的 fresh 副本**。

---

## WP-1 · CS 重复截面（RCS）— 第 1–2 周

**问题**：`callaway_santanna(panel=False)` 只支持 `estimator='reg'`，强制 `control_group='nevertreated'`，
且拒绝 `bstrap` / `clustervars`。CPS/ACS/DHS 这类数据完全跑不了。

**为什么是结构性改动**：现有面板路径在 `callaway_santanna.py:722` 把数据 pivot 成宽表做组内差分。
RCS 没有个体配对，需要一条**长表路径**。

**方案**
1. ~~验证 `sp.drdid` 能否复用~~ → **不能**。从 `did:::compute.att_gt` 源码确认，
   RCS 分派是 `dr`→`DRDID::drdid_rc`、`ipw`→`std_ipw_did_rc`、`reg`→`reg_did_rc`；
   而 `sp.drdid(method='imp')` 精确等于 `DRDID::drdid_rc1`（3.0016328005），
   **不是** `drdid_rc`（3.0026780231）。直接复用会静默产生非对齐结果。
2. ✅ **[已完成]** 新增 `did/_rcs.py`，从 R 源码逐行移植三个估计量**含影响函数**。
3. 新增 `_prepare_rcs()`：保持长表，按 (g, t, base) 切子样本 = {期 ∈ {base, t}} × {队列 g ∪ 控制组}。
4. `_estimate_single_att_rcs()` 调用 `_rcs.py`，把影响函数映射到全样本（注意 `n_total/n_rel` 缩放，
   与面板路径 `callaway_santanna.py:858` 同理），接入现有 `aggte` / 乘子 bootstrap（这一层不用改）。
5. 放开三个 guard，`control_group='notyettreated'` 与 `bstrap` / `clustervars` 全部支持。

### Phase 1 已交付（`did/_rcs.py`）

对齐 R `DRDID` 1.2.3，**ATT 和解析 SE 均 ≤1e-11**：

| 估计量 | ATT | SE |
|---|---:|---:|
| `drdid_rc` | 3.0026780231 | 0.0924898005 |
| `std_ipw_did_rc` | 3.0392849753 | 0.1325097857 |
| `reg_did_rc` | 2.9901454617 | 0.1098485192 |

守卫：四个 treatment×period 单元任一为空 → `DataInsufficient`（§7 响亮失败）。
测试 `tests/reference_parity/test_drdid_rc_parity.py`（12 项）。

**Phase 2（下一步）**：长表 (g,t) 循环 + 影响函数全样本映射 + 放开 guard + 6 组组合对齐。

**参考值（已生成，`did::mpdta`，`xformla=~lpop`）**

| est_method | control_group | simple ATT | SE |
|---|---|---:|---:|
| dr | nevertreated | −0.0417517721 | 0.0460680525 |
| dr | notyettreated | −0.0413516293 | 0.0474594680 |
| ipw | nevertreated | −0.0417770822 | 0.1672310175 |
| ipw | notyettreated | −0.0413894098 | 0.1710682403 |
| reg | nevertreated | −0.0419686124 | 0.1497787162 |
| reg | notyettreated | −0.0413747698 | 0.1502463356 |

**验收**：6 组组合点估计对 R ≤1e-6；SE ≤2%；`tests/reference_parity/test_cs_rcs_parity.py`。
`allow_unbalanced_panel` 语义随之落地（R 的不平衡面板就是走 RCS）。

**风险**：RCS 的 DR 影响函数比面板版复杂，SE 对齐可能比点估计慢。
若 SE 卡住，先交付点估计 + bootstrap SE，把解析 SE 记为已知缺口。

---

## WP-2 · 原生 HonestDiD FLCI — 第 3 周

**问题**：`backend='native'` 算的是 `θ̂ ± 最坏偏差 ± z·SE`，不是 Rambachan-Roth 的部分识别置信集。
在 `smoothness` 下**每个 M 都比 R 窄**（M=0.02 时 0.086 vs 0.097），**高估稳健性**。
这是目前唯一一处默认行为会主动误导人。

**方案**
1. 实现 Δ^SD 下的 FLCI（Rambachan-Roth §3）：给定 `betahat` / `sigma` / `numPre` / `numPost`，
   解固定长度置信区间的凸优化。用 `scipy.optimize`，不引入新依赖。
2. 实现 ARP 条件/混合检验（Δ^RM 用 C-LF）。
3. 补齐 Δ 菜单：Δ^SDRM、Δ^SDPB、单调性 Δ^MB、符号约束 Δ^SDB。
4. `parallel_trends_robustness` 的 `_FAMILY_TO_METHOD` 映射已预留位置，直接扩。

**验收**：native 对 R `HonestDiD` 0.2.8 的区间端点 ≤1e-3（绝对）；
`test_honest_did_backend_parity.py` 中"native 比 R 窄"的断言**反转为相等性断言**。

**风险**：这是整个计划里唯一的研究级实现。若 FLCI 优化不稳定，
降级方案是把 native 默认切成"拒绝执行并要求 `backend='r'`"，而不是继续给近似值。

---

## WP-3 · Roth & Sant'Anna (2023) 高效交错估计量 — ✅ 已完成

**问题**：处理时机真随机（政策抽签、分批上线、RCT rollout）时，设计型推断才是正确的，目前完全没有。

**已交付**：`did/_staggered_rollout.py` + `sp.staggered_rollout`，对齐 R `staggered` 1.2.2，
6 组 `estimand × efficient` 的**点估计和保守（Neyman）SE 均 ≤1e-8**：

| estimand | efficient | plug-in |
|---|---:|---:|
| simple | −0.0470539142 (se .0116138788) | −0.0397636256 (se .0118272142) |
| cohort | −0.0298479506 (se .0125571289) | −0.0304622281 (se .0125590491) |
| calendar | −0.0579882830 (se .0144374235) | −0.0442670835 (se .0157172229) |

⚠️ **更正**：本文档早先版本记的 `simple` = −0.3704347696 是**错的**，来自 never-treated
编码陷阱——R 的 `staggered` 要求 `g = Inf`，喂 `g = 0` 会把未处理组读成"样本前已处理的队列"，
静默给出 −0.3704 而不报错。`sp.staggered_rollout` 接受 `0`/`NaN`/`inf` 并统一归一化，
该陷阱从公开 API 不可达（`test_never_treated_coding_is_normalised` 盯住）。

估计量结构：按队列折叠成均值路径，`A_theta` 放在结果期、`A_0` 放在队列的 `g−1` 预处理期
（随机时机下期望为零，故是无偏控制），`beta* = Xvar⁻¹ X_theta_cov`。

**遗留**：`estimand='eventtime'` 以及 `staggered_cs` / `staggered_sa` 包装未做
（后两者只是把权重换成 CS / SA 的约定，不是新识别）。
测试 `tests/reference_parity/test_staggered_rollout_parity.py`（18 项）。

---

## WP-4 · `fect` 反事实估计量诊断 — ✅ 已完成（诊断层）

**问题**：`interactive_fe` 目前**零诊断**，整个文件只有一个函数。
Liu-Wang-Xu (2024, AJPS) 之后，TSCS 平行趋势检验的标准做法是 F 检验 + 等价性检验（TOST）+ 掩码窗口安慰剂。

**方案**
1. `sp.fect_diagnostics(result)`：no-pretrend F 检验、**等价性检验（TOST）**、掩码窗口安慰剂、
   carryover / exiting-treatment 检验。
2. 接入 `interactive_fe` 和 `matrix_completion` 两个已有估计量。
3. `sp.audit_result` 里把它列为交错设计的必查项。

**参考**：R `fect` 2.4.1。**验收**：F 统计量与等价性检验 p 值对 R ≤1e-4。

---

## WP-5 · CGS 2024 连续处理 — 第 6 周

**问题**：`continuous_did` 自己的 docstring 就写着 `method='cgs'` 是 non-parity、只有 OR、bootstrap SE；
默认的 `att_gt` 只是剂量分箱启发式。

**方案**：实现 Callaway-Goodman-Bacon-Sant'Anna (2024) 的 ATT(d|g,t) / ACRT(d|g,t)，
强平行趋势假设，影响函数方差。把分箱启发式降级为 `method='binned'` 并标注。

**验收**：对论文公开数字或 R 实现（若发布）对齐；否则走解析/仿真验证并在 docstring 明确标注证据等级。

---

## WP-6 · 非吸收性处理 + anticipation 统一 — 第 7 周

**问题**：处理反转只有 `did_multiplegt_dyn` 支持；`anticipation` 在 dispatcher 里只对 `method='cs'` 放行，
但 BJS 和 ETWFE 上游都支持。

**方案**
1. `sp.did` dispatcher：`anticipation` 放行到 `bjs` / `etwfe`。
2. 新增 `sp.detect_design` 的**吸收性检测**；对非吸收性数据，
   在 CS / BJS / ETWFE / stacked 上**主动报错并推荐** `did_multiplegt_dyn` / `lp_did`（§7 响亮失败）。
3. `sp.recommend` 把"是否吸收""是否有预期"纳入决策树。

**验收**：非吸收性面板喂给 CS 必须 raise 而不是给出静默错误答案；`auto_did` 路由测试。

---

## WP-7 · 验证债清理 — 贯穿，第 7 周收口

1. ~~补 parity：`stacked_did`、`lp_did`、`cic`、`ddd_heterogeneous`~~ 除 `lp_did` 外已完成：
   `cic` → Track A 74（`qte::CiC`，4.4e-15）；`stacked_did` → 75（手写 `fixest` stack，1.3e-13，
   两种控制组约定都钉住）；`ddd_heterogeneous` → 77（**`triplediff::ddd` 0.2.4 已上 CRAN**，
   六个 ATT(g,t) cell 1e-14，聚合权重差异开成 `weight_by=`）。
   `lp_did` 仍无参考实现（需付费论文原文），保持 `validation_status='api_stable'`。
   额外拿下 78：`did_multiplegt_dyn` → `DIDmultiplegtDYN` 2.3.4，5e-15，并**修出一个真 bug**
   （placebo 定义错、静默少用一个 cohort）。
2. ~~装 R `pretrends`，补 `pretrends_power` parity。~~ 已完成：Track A 76_pretrends（iterative tier，R 自身 MC 噪声 ~5e-4），并修正 `pretrends_power` 默认检验（joint Wald → Roth 包的逐系数检验）。
3. 修 `gardner_did` 的 parity index 记录（现在还写着"无跨包参考、`sides:['py']`"，
   实际已对齐 `did2s` 到 1e-8）——要往 `tests/r_parity/compare.py` 加 runner。
4. 清理 `待核验` 标记：`lp_did`、`did_timevarying_covariates`、`ddd_heterogeneous`、`did_multiplegt_dyn`。
   核验不了的就按 §10 降级为"（citation needed）"或只留 bib key。
   进度：`lp_did` / `did_timevarying_covariates` / `ddd_heterogeneous` 已处理（`ddd_heterogeneous`
   registry 里的 Strezhnev 占位换成核验过的 `ortiz2025better`）；`did_multiplegt_dyn` 的
   控制组窗口、逐 horizon 权重、placebo 定义三项已被 parity 结掉，剩下 switch-off 处理和
   解析 IF 方差两项——这两项是它还留在 `experimental` 的原因。

5. **新发现**：parity index 漏报了三个 QTE 估计器。`panel_qtet` / `qdid` / `qte` 都有冻结的
   `qte` 1.3.1 参考值（`panel_qtet` 19 个分位数 6.8e-12），但 index 里写着
   `analytical-only` / `sides:['py']`，因为 promotion 表没登记。已补。

---

## WP-8 · Tier-4 现代化 — 第 8 周

1. **溢出 DiD**：现在是 Delgado-Florax (2015)（TWFE + 空间滞后），继承全部负权重问题。
   实现 Butts (2023) 的异质性稳健版（溢出环 + 干净控制组），保留旧版但标注。
2. **分布式 DiD**：补 Callaway & Li (2019) 的 copula QTT（现在 `qdid` 是 Athey-Imbens 2006）。
3. **Roth-Sant'Anna (2023 ECMA) 函数形式检验**：平行趋势能否同时在 level 和 log 成立。
   有了 WP-0 的非线性 ETWFE 之后这个很便宜，天然配套。

---

## WP-9 · 非线性 ETWFE 后续（穿插做）

`etwfe_emfx(type='calendar')` 对 GLM 报错；缺 negbin / fractional 族、`xvar` 调节、非线性 + 重复截面
（依赖 WP-1）。

---

## 需要用户拍板的默认约定（不自行更改）

| 函数 | StatsPAI 默认 | R 默认 | `mpdta` 差距 |
|---|---|---|---|
| `sun_abraham` | `aggregation='event_time'` | 处理观测数加权 | −0.0772 vs −0.0400（**2 倍**） |
| `callaway_santanna` | `base_period='universal'` | `'varying'` | 仅影响 pre-period 安慰剂 |

两者都已有测试盯住，但改默认属于破坏性变更，需显式授权 + MIGRATION 登记。

---

## 工作环境（并发隔离）

两个 Claude 窗口曾同时在主工作树的 `main` 上作业，导致 8 个文件反复冲突：
`registry.py` / `__init__.py` / `CHANGELOG.md` / `MIGRATION.md` 是手工追加点，
`schemas/*` / `_parity_index.json` / `docs/parity.md` / `docs/stats.md` 是**生成产物**
——后者纯粹因为两边都重新生成才冲突。每次提交都要手工做「备份共享文件 → 还原到
HEAD → 重生成派生产物 → 提交 → 还原」五步，做了四轮，且推送闸门按*已提交*状态检查
而工作区混着两边改动，两个视角每次都打架。

现在 DiD 线在独立 worktree 作业：

```bash
cd .claude/worktrees/did-wp-continued        # 分支 worktree-did-wp-continued
```

工作树只含本线改动，生成产物按构造就正确，不再需要手工拆分。

### ⚠️ 必须带 PYTHONPATH

仓库是 editable 安装（`pip install -e .`），`statspai` 被钉死在**主工作树**：

```
Editable project location: /Users/brycewang/Documents/GitHub/StatsPAI
```

所以在 worktree 里裸跑 `import statspai` 仍会加载主树代码——测试会跑在别人的改动上，
隔离形同虚设。每条命令显式指定：

```bash
PYTHONPATH="$(pwd)/src" python3 -m pytest ...
PYTHONPATH="$(pwd)/src" python3 scripts/dump_schemas.py
```

自检：worktree 内 `len(sp.list_functions())` 应等于 `scripts/registry_stats.py --check`
报的数字；不等就说明 `PYTHONPATH` 没生效。

**不要**为此改 `pyproject.toml` 加 pytest `pythonpath`——主树的 editable 安装对另一条
线是正确的，改了会污染共享配置。

## 全局约束

- **JOSS**：以上全部**不需要发 GitHub Release**，不触发 Zenodo，审稿安全。
  改数值的一律走 CHANGELOG ⚠️ correctness + MIGRATION。
- **并发**：另有窗口在改 `matching/`（`cbps.py` / `ebalance.py` / `match.py`）。
  提交时**只 stage 自己的文件**，绝不 `git add -A`。
- **既有失败**：`test_jss_manuscript_artifacts`、`test_jss_release_manifest`（2）、
  `test_jss_formal_compliance` 在 `HEAD~1` 即失败，与 DiD 无关，不在本计划范围。


---

## 完成状态（2026-08 收口）

WP-5 / WP-7 / WP-8 已落地，均带跨语言证据或明确写明的证据边界。

| 项 | 结果 | 证据 |
| --- | --- | --- |
| WP-5 CGS 连续处理 | `sp.cgs_continuous_did`（ATT(d) / ACRT(d)，B-spline + 影响函数） | Track A 80，对 `contdid` 0.1.1，曲线与两个整体量 1e-12 |
| WP-7 `stacked_did` | Track A 75 | 手写 `fixest` stack，1.3e-13，两种控制组约定 |
| WP-7 `pretrends_power` | Track A 76 + ⚠️ 默认检验修正 | 对 `pretrends` 0.1.0，在其自身 MC 噪声内 |
| WP-7 `ddd_heterogeneous` | Track A 77 + 协变量 / 解析 SE / not-yet-treated | 对 `triplediff` 0.2.4，cells 与解析 SE 1e-12（dr/ipw/reg） |
| WP-7 `did_multiplegt_dyn` | Track A 78 + ⚠️ placebo 定义修正 + 解析 SE | 对 `DIDmultiplegtDYN` 2.3.4，5e-15（含 switcher 计数） |
| WP-8 函数形式检验 | `sp.functional_form_test`，Track A 79 | 对 `didFF` 0.1.0，1.2e-15，接受与拒绝两种设计都钉住 |
| WP-8 溢出 DiD | `sp.spillover_did` | **无参考实现**，仅设计恢复证据 + 对 `spatial_did` 有偏的对照 |

### 上游缺陷（已记录，未继承）

两处刻意与参考实现不一致，都有可复现的证据而非断言：

1. `triplediff` 0.2.4 的 not-yet-treated 路径把各控制组队列的影响函数写进面板长度向量时用了长度不符的布尔索引（R 每次调用都会警告）。逐控制组估计与我们完全一致；只有组合环节不同。比较范围覆盖全面板的 cell 仍然逐位相符。
2. `contdid` 0.1.1 在拟合用的 dose 范围上估计样条，却把上报曲线放在以 dose 网格端点为边界的另一组基上求值，因此上报曲线是拟合响应的一个缩放版本，与它自己返回的 overall ACRT 对不上。`curve_basis="reference"` 可复现其输出。

### 仍未闭合

- `lp_did`：论文需付费获取，无参考实现，保持 `api_stable`。
- `did_multiplegt`（dCDH 2020）：R 包 2.1.0 的 `mode="old"` 连自带示例都返回 NaN，属上游问题。
- `did_multiplegt_dyn`：switch-off 事件、论文自有方差公式两项 `[待核验]` 仍在，也是它仍标 `experimental` 的原因。
- `did_timevarying_covariates`：归属文献无法核验，按 §10 保持「（citation needed）」。
