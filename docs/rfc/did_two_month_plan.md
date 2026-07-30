# DiD 两个月工作计划（2026-07-31 起）

> 目标：把 `sp.did` 家族从"方法齐全但数据形态受限"推进到"post-2021 主流 DiD 的完整可用实现"。
> 每个工作包（WP）的验收标准都是**对齐参考实现的数值证据**，不是"代码写完了"。

## 0. 现状基线

- **已 R 对齐（certified）**：`callaway_santanna`、`sun_abraham`、`did_imputation`、`honest_did`
- **有证据（validated）**：`gardner_did`、`etwfe`、`aggte`、`continuous_did`
- **无数值证据（api_stable）**：`harvest_did`、`design_robust_event_study`、`cohort_anchored_event_study`、`did_misclassified`、`lp_did`、`ddd_heterogeneous`、`did_timevarying_covariates`、`stacked_did`、`cic`、`overlap_weighted_did`

参考环境（已就绪）：R 4.5.2 + `did` 2.3.0 / `etwfe` 0.6.2 / `fixest` 0.14.0 / `did2s` 1.2.1 /
`didimputation` 0.5.1 / `HonestDiD` 0.2.8 / `staggered` 1.2.2 / `fect` 2.4.1 / `DRDID` 1.2.3。
`pretrends` **未安装**（WP-7 需先补）。

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

## WP-3 · Roth & Sant'Anna (2023) 高效交错估计量 — 第 4 周

**问题**：处理时机真随机（政策抽签、分批上线、RCT rollout）时，设计型推断才是正确的，目前完全没有。

**方案**：新增 `sp.staggered_rollout(df, i, t, g, y, estimand=)`，
实现 Roth-Sant'Anna 的高效 GMM 估计量 + 设计型方差；同时提供 `estimand='simple'/'cohort'/'calendar'/'eventstudy'`
以及他们的 `staggered_cs` / `staggered_sa` 包装。

**参考**：R `staggered` 1.2.2。`did::mpdta` 上 `estimand='simple'` = −0.3704347696（SE 0.1256399086）——
注意这与 CS 的 −0.0400 **不可直接比较**，估计量不同（随机时机 vs 平行趋势）。

**验收**：4 种 estimand 对 R ≤1e-6；registry 注册 + `sp.recommend` 在检测到随机 rollout 时能推荐它。

---

## WP-4 · `fect` 反事实估计量诊断 — 第 5 周

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

1. 补 parity：`stacked_did`、`lp_did`、`cic`、`ddd_heterogeneous`（`cic` 对 Stata `cic`，其余对手写 R）。
2. 装 R `pretrends`，补 `pretrends_power` parity。
3. 修 `gardner_did` 的 parity index 记录（现在还写着"无跨包参考、`sides:['py']`"，
   实际已对齐 `did2s` 到 1e-8）——要往 `tests/r_parity/compare.py` 加 runner。
4. 清理 `待核验` 标记：`lp_did`、`did_timevarying_covariates`、`ddd_heterogeneous`、`did_multiplegt_dyn`。
   核验不了的就按 §10 降级为"（citation needed）"或只留 bib key。

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

## 全局约束

- **JOSS**：以上全部**不需要发 GitHub Release**，不触发 Zenodo，审稿安全。
  改数值的一律走 CHANGELOG ⚠️ correctness + MIGRATION。
- **并发**：另有窗口在改 `matching/`（`cbps.py` / `ebalance.py` / `match.py`）。
  提交时**只 stage 自己的文件**，绝不 `git add -A`。
- **既有失败**：`test_jss_manuscript_artifacts`、`test_jss_release_manifest`（2）、
  `test_jss_formal_compliance` 在 `HEAD~1` 即失败，与 DiD 无关，不在本计划范围。
