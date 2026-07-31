# DML / ML-causal 板块 parity 补强计划（8 周）

> 状态追踪文档。每个 Phase 完成后就地更新 `状态` 字段，不另开文件。
> 起始日期：2026-07-31。目标完成：2026-09-25。

## 0. 问题陈述

2026-07-31 的板块审计给 DML / ML-causal 的评语是
**"Adequate, thin anchoring"**。核对 `src/statspai/_parity_index.json`
后，这个评语成立，且"薄"的位置非常集中：

| 子块 | 当前 status | 锚点 | 问题 |
| --- | --- | --- | --- |
| `dml`（PLR） | `bit-exact` | `DoubleML::DoubleMLPLR` | 无。三方 py/R/Stata，rel 0.0 |
| `dml`（IRM/PLIV/IIVM） | 未单列 | `doubleml-for-py`（可选 import） | 无 R 侧；未装即 skip；index 上不可见 |
| `causal_forest` | `aligned` | `grf::causal_forest` | 点估计 0.19% 已达 T3；SE 容差 `rel_se<=0.50` |
| `tmle` | `bit-exact` | base R `stats::glm` 手搭 | 锚的是自建参考，不是 `tmle` 包 |
| `metalearner` | `external-replication` | CausalML Book DGP 真值 | **零跨包数值 pin** |
| `policy_tree` | `analytical-only` | oracle 恢复 + 会计恒等式 | **零跨包 pin**；`policytree` 未对 |
| `dml_panel` | 不在 index | — | 有单测，index 空白 |
| `model_averaging_dml` | 不在 index | — | 有单测，index 空白 |
| `dml_sensitivity` | 不在 index | — | DoubleML 有官方对应实现，未对 |
| `super_learner` / `hal_tmle` / `ltmle` / `ope` | 不在 index | — | index 空白 |

## 1. 中心设计原则：共享 nuisance

跨包 pin ML 估计器的经典失败模式是：两边各自训练 nuisance 模型，
RNG / learner / fold 都不同，于是只能挂一个 25%–500% 的容差带，
然后把它叫做"parity"。**那不是验证。**

勘查确认三个参考实现都开放了外部 nuisance 注入口：

| 参考包 | 注入点 | 效果 |
| --- | --- | --- |
| `policytree::policy_tree(X, Gamma, depth)` | `Gamma` 是 n×d 奖励矩阵 | 树搜索与 AIPW 打分完全解耦 |
| `tmle::tmle(Y, A, W, Q=, g1W=)` | `Q` 是 n×2 矩阵，`g1W` 是倾向得分 | 只剩 1 维 fluctuation，可 bit-exact |
| `DoubleML(..., draw_sample_splitting=FALSE)` + `set_sample_splitting(fold_id)` | 显式 fold | 08_dml 已在用 |

**因此本计划的每一个新 pin 都走同一条路**：Python 侧算 nuisance →
写进 CSV → R 侧读同一份字节 → 两边跑各自的估计器 → 比较。
残差就只剩估计器本身的实现差异，容差可以压到 1e-8 量级。
这同时把"这个数值差是 learner 噪声还是实现 bug"这个问题永久消除。

## 2. Phase 分解

### Phase 1（W1–2）— `policy_tree` × `policytree`：模块 70

**状态：** ✅ 完成（2026-07-31）。`policy_tree` `analytical-only` → **`bit-exact`**，
rel 9.6e-16 vs `policytree` 1.2.4，两侧 1200 行策略向量逐行一致。

勘查已发现实质缺陷（先于 pin 存在）：

1. `policy_tree.py::_grow_tree` 的 docstring 声称
   *"For depth-1 (stump) and depth-2 trees, an exact solution is found
   via exhaustive search over all possible splits"*，但实现是**贪心**：
   根分裂按"两个子节点都当叶子"打分，然后递归。depth=2 时这不是最优解。
2. 候选分裂点被**分位数下采样到每特征 ≤50 个**
   （`n_candidates = min(50, len(vals) - 1)`），
   所以连贪心解都不是全网格上的贪心解。
3. `policytree` 对给定 depth 做精确穷举（Sverdrup et al. 2020），
   两者必然不一致。

**实测影响**：模块 70 fixture 上，旧贪心搜索的 policy value 比最优解低
**0.70%**，1200 行里 **78 行**给出不同的治疗建议。

**工作内容**

- [x] 实现精确 depth-1 / depth-2 树搜索（`policy_learning/_exact_tree.py`），
      全分裂网格，对齐 `policytree` 的 `<=` 分裂语义与 `min.node.size` 约定
- [x] depth ≥ 3 保留贪心（穷举组合不可行），在 docstring 和
      `result["search_mode"]` 里显式标注哪一档是精确的；`search="auto"`
      超预算回退贪心时发 `UserWarning`，不静默降级
- [x] 新增 `scores=` 注入口（共享 nuisance 架构的前提，同时对用户有用）
- [x] 新增 Track A 模块 70：共享 Γ scores，比较 policy value / treated
      fraction / 根分裂变量与阈值（depth 1 和 2）
- [x] `tests/reference_parity/test_policy_tree_r_parity.py` 逐行断言两侧
      policy 向量完全一致（标量相等不足以证明是同一棵树）
- [x] `tests/test_policy_tree_exact_search.py` 对独立暴力枚举做随机化验证
      （含并列值），1800 组零不匹配
- [x] CHANGELOG ⚠️ correctness fix + MIGRATION 条目
- [x] `renv.lock` / `R_ENVIRONMENT.md` / README 模块表 / tier lock 同步
      （顺手修掉 `_gen_renv_lock.R` 漏列 `spatialreg` 导致重生会丢包的既存问题）

**验收：** ✅ `policy_tree` 从 `analytical-only` → `bit-exact`（rel 9.6e-16）。


---

### Phase 2（W2–3）— `causal_forest` SE 收紧：模块 13 强化

**状态：** ✅ 完成（2026-07-31）。算子级 pin 建立后暴露并修掉一个 ATT 约定错误；
`rel_se` 从 0.50 收到 0.25，ATT 的 SE 相对差从 14.6% 塌到 **0.087%**。

现状：点估计 `rel_est` 0.19%（容差 0.005）已是 T3 合并 MC 误差通过；
弱点只在 `rel_se <= 0.50`（实测 7.7%，历史最坏 14.6%）。

50% 这个数不是懒惰，是**两边森林 RNG 不同导致 AIPW SE 天然不可比**。
硬压容差是造假，正确做法是换判据。

**工作内容**

- [x] 算子级 pin：新增 frozen fixture `_generate_grf_scores.R`，携带 grf
      自己的 `tau.hat` / `Y.hat` / `W.hat` + `grf::get_scores()` + ATE/ATT。
      把估计器拆成**森林**（跨实现不可 pin）与 **AIPW 算子**（闭式，可精确 pin）
      两个因子，只对后者做精确断言
- [x] 算子抽成 `aipw_scores` / `grf_att_atc` 两个可测函数，测试走真实代码路径
      而不是在测试里重写公式
- [x] **发现并修复 ATT/ATC 约定错误**（详见下）
- [x] 新增干净 overlap 覆盖率守卫（`@pytest.mark.slow`），ATE 与 ATT
      覆盖率均 **94.3%**（名义 95%，B=300 试跑）。**注**：手稿级的
      `results_b1000/*.json` 行已撤回 —— 该 JSON 受
      `test_jss_release_manifest.py` 合约约束，每个数字必须同步进 7 处叙述
      文件（含 `README.md` / `README_CN.md` / JSS 手稿）。那属于论文范围，
      且当时有并发会话正在编辑同一批 README。CI 强制的守卫才是持久证据
- [x] `rel_se` 0.50 → **0.25**（实测最坏 7.7%，3.2× 余量），
      `rel_est` 0.005 → 0.01（ATT 改用 grf 估计量后残差全是森林 MC，
      0.005 只剩 1.07× 余量），`docs/dev/r_parity_tolerances.md` 三处同步
- [x] index 记录加 factor-level note，`sp.parity_status('causal_forest')`
      现在能说清哪部分是精确的、哪部分是蒙特卡洛

**⚠️ 顺带发现的正确性问题**：`average_treatment_effect(target_sample='treated'
/'control')` 的 docstring 宣称 "GRF-style"，但 ATT/ATC 实际是"单个 Robins
score 除以 `p̂₁`"，而 grf 是"目标臂 CATE 插值均值 + Hájek 归一化 DR 修正"，
方差是两个分量之和。**在森林输出完全相同的前提下**，旧算法的 ATT 点估计与 grf
差 9.3e-5，但 SE **大 12%**。这个偏差端到端看不见 —— 它正好藏在 `rel_se<=0.50`
那条宽带里。这印证了 Phase 2 的设计判断：宽容差不是"暂时不够好"，
而是**主动掩盖了一个真实缺陷**。

**验收：** ✅ SE 证据从"50% 相对带"换成"算子精确 pin（1e-15）+ 覆盖率校准（94.3%）"。

---

### Phase 3（W3–4）— DML 家族 R 侧扩展：模块 71

**状态：** ✅ 完成（2026-07-31）。PLIV **6.5e-16**（机器 floor），
IRM / IIVM **1.1e-10**；顺带修掉一个 SE 自由度不一致。

`08_dml` 只覆盖 PLR，index notes 自陈 *"Grade is variant-specific"*。
IRM / PLIV / IIVM 目前只有 Python 侧 `doubleml` pin，且是 optional
import（`pytest.importorskip`），CI 未装即静默跳过。

**工作内容**

- [x] **前置条件**：`fold_indices` 原先只有 PLR 支持，其余三个模型直接抛异常
      （好在不是静默忽略）。抽出 `_DoubleMLBase._make_splits`，四个模型统一走
      共享折；供折时估计值与 `random_state` 无关（这正是 bit-exact 的前提）
- [x] 供折绕过 `StratifiedKFold`，因此新增校验：训练折若单类则抛
      `DataInsufficient` 并指名是哪一折，而不是让分类器拟出退化倾向得分
- [x] 新增模块 71，三个模型类全覆盖，learner 用两边闭式对应物
      （`regr.lm`/`LinearRegression`、`classif.log_reg`/`LogisticRegression(penalty=None)`）
- [x] 约定对齐**显式记录**：`trimming_threshold` DoubleML 默认 1e-12 vs
      StatsPAI 1e-2，两侧统一设 1e-12；DGP 让倾向得分留在 (0.2, 0.8)，
      所以该参数在本模块里是惰性的 —— 目的是消除混淆项，不是躲在它后面
- [x] index 的 variant-specific 提示改为**枚举全部已认证调用**
      （顺带让 `sp.parity_status('decompose')` 也从只显示一个变成两个都显示）

**⚠️ 顺带发现的正确性问题**：IRM/IIVM 的无权重分支用 `ddof=1` 归一化影响函数
方差，而**同一个 dispatcher 的另外四条路径**（PLR、PLIV、IRM/IIVM 的加权分支、
以及 `irm.py` 内部的 `normalize_ipw`/ATTE 分支）全用 `n`。DoubleML 也用 `n`。
实测比值 1.00025009389849 对 `sqrt(2000/1999) = 1.00025009378908` —— 吻合到 1e-11，
确认是自由度约定而非别的。n=2000 时影响 0.025%，n=50 时约 1%。

**验收：** ✅ `sp.parity_status('dml')` 现在列出全部四个已认证变体。

---

### Phase 4（W4–5）— `dml_sensitivity` × DoubleML `sensitivity_analysis`

**状态：** ✅ 完成（2026-07-31）。`dml_sensitivity` 从"不在 index" →
**`bit-exact`**，`bias_bound` 2.5e-15、`RV` 9.2e-8。

**计划偏离（如实记录）**：原计划做成 Track A 模块 72（R 侧参考）。
勘查后发现 **DoubleML R 1.0.2 根本没有敏感性分析** —— `DoubleMLPLR`
及其基类的公开方法只有 `initialize / print / fit / bootstrap /
split_samples / set_sample_splitting / tune / summary / confint /
learner_names / params_names / set_ml_nuisance_params / p_adjust /
get_params / clone`。该功能只存在于 doubleml-for-py。
因此改为 Python↔Python 的 `external_parity` pin，不再占用 Track A 模块号。

**工作内容**

- [x] 核对 estimand 定义。**关键发现**：我先按教科书推的
      `sqrt(cf_y/(1−cf_y))·sqrt(cf_d/(1−cf_d))` 是错的；读 DoubleML 源码
      （`DoubleMLFramework._calc_sensitivity_analysis`）确认它用的是
      `|rho|·sqrt(cf_y·cf_d/(1−cf_d))` —— 与 StatsPAI **完全一致**。
      这一步说明了为什么必须读参考实现源码而不是靠推导
- [x] 定位到唯一差异在 `max_bias = sqrt(sigma2·nu2)` 的 `sigma2`
- [x] 新增 `tests/external_parity/test_dml_sensitivity_parity.py`（6 测试），
      共享折 + 先断言两边 PLR 拟合完全相同，再比较敏感性量
- [x] `RVa` 的约定差**显式断言**（<5e-3，实测 1.4e-3）而不是塞进宽容差
- [x] `dml_sensitivity` 通过 `_FROZEN_PROMOTIONS` 进 index，
      等级 `bit-exact` + 准确的 reference 字符串

**⚠️ 发现的正确性问题**：模块头声称实现 CCNSS (2022) 的 DML-OVB 界，其
PLR 缩放因子应为 `S = sqrt(σ²ν²)`，`σ² = E[(Y−ℓ−θ(D−m))²]`（**结构残差**）。
代码用的是 `sd(Y−ℓ)`，把 `θ(D−m)` 留在了分子里。由于
`sd(Y−ℓ)² = σ² + θ²·sd(D−m)²`，`S` 被系统性放大 →
**偏差界被高估、稳健值被低估**，即分析把估计值说得比实际**更脆弱**。
实测：bias bound 0.0671（应为 0.0529，高 27%），`RV_1` 0.454（应为 0.533）。
IRM 路径不受影响（它存的 `y_resid` 已是 `ψ−θ̂`，再减会重复扣除）。

**验收：** ✅ `dml_sensitivity` `bit-exact`。

---

### Phase 5（W5–6）— `metalearner` 跨包 pin

**状态：** ⬜ 未开始

S/T/X-learner 目前零跨包锚。`econml` 0.16.0 已在系统 python。

**工作内容**

- [ ] `.venv` 安装 `econml`
- [ ] T-learner / S-learner 在给定 base learner 时是**确定性**的
      → 目标 bit-exact
- [ ] X-learner 的加权约定（用倾向得分 g 还是 1-g 加权两个 CATE 估计）
      各家不同 → 逐条核对，差异记录为 convention 而不是失败
- [ ] DR-learner 对 `econml.dr.DRLearner`
- [ ] 进 external_parity（Python↔Python）而非 r_parity

**验收：** `metalearner` 从 `external-replication` → 至少一个变体 `bit-exact`。

---

### Phase 6（W6–7）— `tmle` 锚点升级到 `tmle` 包：模块 73

**状态：** ⬜ 未开始

现锚是 base R `stats::glm` 手搭 TMLE（frozen fixture，数值精确但锚点弱）。
`tmle::tmle` 2.1.1 已装，且接受外部 `Q` / `g1W`。

**工作内容**

- [ ] 新增模块 73：Python 侧导出 `Q(0,W)` / `Q(1,W)` / `g1W` →
      R 侧 `tmle::tmle(Y, A, W, Q=Q, g1W=g1W)`
- [ ] 比较 `psi` / `var.psi` / `epsilon`（fluctuation 参数）/ CI
- [ ] 保留现有 frozen-glm fixture 作为第二重证据，不删

**验收：** `tmle` 的 reference 字段从 "base R stats::glm TMLE" →
`tmle::tmle`（官方实现）。

---

### Phase 7（W7）— index 补全

**状态：** ⬜ 未开始

**工作内容**

- [ ] `dml_panel` / `model_averaging_dml` / `super_learner` /
      `hal_tmle` / `ltmle` / `ope` / `auto_cate` / `cate_eval`
      进 `_parity_index.json`
- [ ] 没有跨包锚的诚实标 `analytical-only`；索引空白 ≠ 未测试，
      但外部读者看不出区别，这个亏必须补上

**验收：** `sp.parity_status()` 对本板块所有对外函数都返回非 `unverified`
或明确的 `unverified` + 理由。

---

### Phase 8（W8）— 收尾

**状态：** ⬜ 未开始

- [ ] `docs/parity.md` 重生
- [ ] `scripts/tier_a_fixture_lock.py --write`
- [ ] `tests/r_parity/README.md` 模块表 + R 依赖清单更新
- [ ] `renv.lock` 补 `policytree` / `tmle` / `SuperLearner`
- [ ] CHANGELOG / MIGRATION
- [ ] 全量 `pytest -q` + `pytest tests/reference_parity/ -q`

## 2.5 中期小结（Phase 1–4 完成，2026-07-31）

四个 Phase 各自都**先建 pin、再由 pin 暴露缺陷**，这个顺序不是巧合：

| Phase | pin 建立后暴露的缺陷 | 影响方向 |
| --- | --- | --- |
| 1 | `policy_tree` depth-2 是贪心而非文档承诺的精确搜索 | 学到的策略次优，1200 行中 78 行给错建议 |
| 2 | `causal_forest` ATT/ATC 未用 grf 的估计量 | ATT 的 SE 偏大 12% |
| 3 | IRM/IIVM 的 SE 用 `n−1`，与同包另外四条路径不一致 | SE 偏大 `sqrt(n/(n−1))` |
| 4 | `dml_sensitivity` 的 `S` 漏减 `θ(D−m)` | 偏差界高估 27%，稳健值低估 |

**Phase 2 的那一条最说明问题**：`rel_se <= 0.50` 这条宽容差不是"暂时不够好"，
它**主动掩盖了一个 12% 的系统性偏差**。宽容差的代价不是精度，是让缺陷隐形。

**方法论上值得记下的一条**：Phase 4 里我先按论文推导出混淆强度因子应为
`sqrt(cf_y/(1−cf_y))·sqrt(cf_d/(1−cf_d))`，结果是**错的** —— 读 DoubleML 源码
才发现它用 `sqrt(cf_y·cf_d/(1−cf_d))`，而 StatsPAI 本来就是对的。
**对齐参考实现必须读参考实现的源码，不能靠从论文重新推导。**

## 3. 不做的事（显式排除）

- **不碰 JOSS 论文** `paper.md`。本计划只动 `Paper-JSS/`
  的 Appendix B 表格（由 `compare.py` 自动重生），与 JOSS 审稿无关。
- **不发 GitHub Release**（会触发 Zenodo 归档，审稿期禁止）。
- **不为了压容差而改 DGP**。把 overlap 调好是合法的（模块 13 已这么做过
  并写明了理由）；把噪声调小到刚好过线不是。
- **不新增重依赖到核心 `dependencies`**。`econml` / `doubleml` /
  `policytree` / `tmle` 全部是测试期可选，缺失即 skip。

## 4. 风险

| 风险 | 缓解 |
| --- | --- |
| `policy_tree` 精确化改变现有用户数值 | ⚠️ correctness fix 走 CHANGELOG + MIGRATION，旧行为不保留（贪心结果本就与文档不符） |
| R 侧 learner 无法与 sklearn 完全一致 | 共享 nuisance 架构绕开这个问题；无法绕开的场景退回解析可解的 learner（OLS / 无罚逻辑回归） |
| 新模块拖慢 CI | Track A 模块不进默认 pytest 路径，只在 `external_parity_runtime` 标记下跑 |
| tier lock 漂移 | 每个 Phase 结束跑 `scripts/tier_a_fixture_lock.py --write` + 合约测试 |
