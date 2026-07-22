# One-Week Plan — 《Rebel on the Canal》复现缺口

> 来源：`top5-paper-reading/AER2022-Rebel-on-the-Canal/StatsPAI 需要改进的df.md`
> 目标：把该文列出的 7 项缺口在一周内落地。
> 状态：**计划已按实测结果修订** —— 见 §0。

---

## 0. 动手前的实测发现（修订原清单）

原 md 文件的若干判断与 v1.20.0 实际行为不符。以下全部经 Stata 18 MP 交叉验证，不是代码阅读推断。

### 0.1 🔴 新增 P0-0：varying-slope FE 被**静默丢弃**（原清单未发现）

```
sp.feols("y ~ d | county + pref[year]")  ->  d = 0.33723149929448304
sp.feols("y ~ d | county")               ->  d = 0.33723149929448304   # 完全相同
Stata: reghdfe y d, absorb(county i.pref#c.year)  ->  d = 0.066701865575718
```

`sp.feols` 委托 pyfixest 0.50.1；后者接受 `pref[year]` 语法但**不吸收斜率**，返回值与不写该项**逐位相同**，且**不发任何 warning**。系数差 5 倍。

这违反 CLAUDE.md §7「失败要响亮」与 §12「不要悄悄改数值」，优先级高于原清单任何一项。**Day 1 上午先修**（至少改成响亮失败），再谈功能扩展。

正确通路今天已存在：`i(pref, year)` 放在 RHS，对 Stata 差 8e-9。

### 0.2 🟡 P0-2 范围比原文小得多 —— 交互/FE 组合**已经对齐 Stata**

原文称 `feols` "只接受裸列名，不支持 `a:b` / `a*b` / `fe1^fe2`"。实测（600 obs 面板，对 `reghdfe` 比对）：

| 规格 | StatsPAI | Stata `reghdfe` | 结论 |
|---|---|---|---|
| `y ~ d + x \| county + year` | 0.35262441 | 0.35262441 | ✅ 一致（~1e-8） |
| `y ~ d + x:year \| county` | 0.036653193 | 0.036653188 | ✅ 一致（~5e-9） |
| `y ~ d \| county + prov^year` | 0.048011144 | 0.048011141 | ✅ 一致（~3e-9） |
| `y ~ d \| county + i.pref#c.year` | **SyntaxError** | 0.066701866 | ❌ 真缺口 |
| `y ~ d \| county + pref[year]` | **0.337231（静默错）** | 0.066701866 | 🔴 见 §0.1 |

残差量级 ~1e-8 是交替投影的收敛容差，非算法差异（`i(pref, year)` 走 RHS 估计时对 Stata 差 8e-9，同量级）。**不是逐位相同**，parity 测试容差应设 1e-6 而非 1e-12。

原文的判断大概率基于 `sp.hdfe_ols`（`panel/feols.py:163-195` 的自研解析器，确实只认裸列名），而非 `sp.feols`（走 pyfixest）。**两条通路解析能力不一致**，这本身是要修的问题。

P0-2 实际范围收窄为三件事：
- (a) Stata 语法 `i.f#c.x` → pyfixest `i(f, x)` 的翻译层；
- (b) StatsPAI 自研 HDFE kernel 增加 varying-slope 投影（`panel/hdfe.py` 目前全无 slope 概念）；
- (c) `panel/feols.py` 解析器补齐到与 `sp.feols` 同级。

### 0.3 🔴 新增 P1-3 前置：`event_study` 不暴露 vcov，honest-DiD 在**静默用对角近似**

`did/event_study.py:389-416` 算出完整 `vcov` 后只 `return np.sqrt(np.diag(vcov))`。`model_info` 里没有 `vcv_pre`。于是 `pretrends.py:267` `_pre_vcv` 落到 `return np.diag(se_pre**2)` 分支 —— **假设前期系数互相独立**。

事件研究的前期系数几乎必然相关（共用参照期、共用 FE）。对角近似会**系统性错报** Rambachan-Roth breakdown $\bar M^*$ 和 Roth power。P1-3 的 pipeline 建在这上面就是建在沙上，**必须先修**。

### 0.4 🟡 P2-1 无法按原文执行 —— 复现包**没有数据**

`AER2022-Rebel-on-the-Canal/` 只有 `master.do` + `Program/setup.do`。`Data/Final/rebellion.dta`、`Program/Analysis/*.do`、`spatial_HAC` ado **全部缺失**（该仓库 README 自己说明了这点）。原文 §6.2「拿到 openICPSR 复现包 → 跑 Table 3 列 1-5」在当前工作区**不可执行**。

替代方案（本地 Stata 18 MP 可用，已验证）：
- **oracle 对齐**：合成 GIS 面板 → Stata `reghdfe` / `acreg` → StatsPAI，容差 1e-8/1e-6。`acreg` 已安装，正是原文 §1.4 点名的 Conley 时空 oracle。
- **published-number 对齐**：论文正文数字（β=0.0380、SE=0.0166、breakdown $\bar M^*\approx1.9$）进 `tests/external_parity/`。
- openICPSR 数据到手后，parity 框架可直接复用 —— 框架本身才是长期资产。

已安装：`reghdfe` `ftools` `acreg` `synth`。缺：`cic` `ols_spatial_HAC` `qrprocess`（Melly 站点，Day 4 尝试装；装不上则 CIC 走解析/自洽测试）。

### 0.5 🟡 Conley 现状：三套距离约定并存 + 一处静默错位风险

| 通路 | 实现 | meat | 时间维 | 面板去重 |
|---|---|---|---|---|
| `sp.conley` | `inference/conley.py:75` | **稀疏 KD-tree** ✅ | ❌ | ❌ |
| `feols(vce="conley")` | `inference/jackknife.py:896` | **稠密 n×n** 🔴 | ❌ | ❌ |
| `hdfe_ols(vce="conley")` | 同上 | 稠密 n×n 🔴 | ❌ | ❌ |
| `spatial_did` | `spatial/did.py:816` | 稀疏 | ⚠️ 分期块对角（无跨期） | ✅ |

- 地球半径三个值并存：`conley.py` 用 6371.0，`jackknife.py:glm_conley_vcov` 用 6371.01，`ols_conley_vcov` 用平面 111 km/度。**应统一**。
- `jackknife.py:880-885` 只校验坐标长度 + NaN。若 pyfixest 因 NA/singleton 丢行，坐标会**静默错位**到错误观测上。这是潜在正确性 bug，Day 1 一并修。

### 0.6 修订后的优先级

| 级别 | 项目 | 依据 |
|---|---|---|
| **P0-0** | varying-slope 静默丢弃 → 响亮失败 | §0.1，静默错误 |
| **P0-0b** | Conley 坐标静默错位 → 响亮失败 | §0.5，静默错误 |
| **P0-1** | Conley 时空 HAC（时间核 + 面板去重 + 稀疏化三条通路） | 原 P0-1 |
| **P0-2** | varying-slope 吸收 + `i.f#c.x` 翻译 + 两解析器统一 | 原 P0-2（收窄） |
| **P1-0** | `event_study` 暴露 vcov | §0.3，honest-DiD 前置 |
| **P1-1/2/3** | CIC 两步法 / 事件研究分箱 / PT-robustness pipeline | 原 P1 |
| **P2-1** | parity 框架（Stata oracle + published numbers） | 原 P2-1（改道） |
| **P2-2** | GIS 薄封装 + 教程 | 原 P2-2 |
| **P2-3** | §8 agent-native（don't-use-when / 可恢复错误） | 原 §8 |

---

## 1. 日程

### Day 1 — 止血 + Conley 时空核

**上午（止血，两处静默错误）**
1. `fixest/wrapper.py`：检测 FE 部分的 `name[slope]` / `name[[slope]]`，pyfixest 无法吸收 → 抛 `ValueError`，消息给出可行替代（`i(f, x)` 或 `sp.hdfe_ols`）。
2. `inference/jackknife.py:880`：坐标对齐从「长度检查」升级为 index 对齐；无法对齐则抛错，不静默。
3. 两条各配一个回归测试（断言**抛错**，而非断言数值）。

**下午（P0-1 核心）**
4. `inference/conley.py` 新增 `time=` / `lag_cutoff=` / `time_kernel=`：
   - 空间核 × 时间核（uniform / Bartlett）张量积；
   - 坐标按 unit 去重建 KD-tree（575 点而非 140k 行）；
   - meat 按 (unit, time) 邻对分块累加，`O(邻对数)` 而非 `O(n²)`。
5. 统一地球半径为 6371.0 km，把 `ols_conley_vcov` 的平面近似标为 `convention="planar"` 显式选项（保留 acreg 兼容），默认球面。

**验收**：`acreg y x, spatial lat lon dist(500) lag(L) id() time() hac bartlett` 对齐 1e-6；140k 行内存 < 4 GB。

### Day 2 — P0-2 varying slopes

1. `panel/hdfe.py`：`Absorber` 支持 slope 项。交替投影里，slope 组的投影是组内对 `[1, x]` 的回归残差化（而非减均值）。加权路径同步。
2. `panel/_hdfe_kernels.py`：新增 `sweep_slope` / `sweep_slope_weighted`（numba + numpy 双实现）。Rust 侧暂不动（`hdfe_rust.py` 当前是 branch-only，主干本就回退）。
3. `panel/feols.py` 解析器：支持 `a:b`、`a*b`、`fe1^fe2`、`i.f#c.x`、`f[x]`，与 `sp.feols` 对齐。
4. `fixest/wrapper.py`：`i.f#c.x` → `i(f, x)` 翻译层；FE 侧 `f[x]` 改为路由到 StatsPAI 自研 HDFE（现在它能吸收了），不再报错。

**验收**：`sp.feols("y ~ d | county + i.pref#c.year")` 与 `reghdfe absorb(county i.pref#c.year)` 对齐 1e-8。

### Day 3 — event_study vcov + 分箱

1. `did/event_study.py`：`_cluster_se` 返回完整 vcov；写入 `model_info["vcv_pre"]` 与 `model_info["vcov"]`。**⚠️ correctness fix** —— 会改变现有 honest-DiD / pretrends 数值输出，进 CHANGELOG + MIGRATION。
2. `did/pretrends.py`：`_pre_vcv` 落到对角分支时发 `warnings.warn`，不再静默。
3. `event_study(bin_width=..., ref_period=("<=", -50))`：分箱 + 区间参照。`ref_period` 保持接受 `int`（向后兼容），新增 tuple/list 形式。
4. `fast/event_study.py` 同步（保持两实现语义一致）。

### Day 4 — CIC 两步法 + PT-robustness pipeline

1. `did/cic.py`：`covariates=` + `first_stage="feols"`；内部 feols 残差化，`keep_mask` 自动对齐；bootstrap **重抽两步**（当前只重抽第二步 → 低估 SE）。
2. `sp.parallel_trends_robustness(result, m_grid=..., families=["SD","RM"])`：串起 pretrends_power → honest_did(SD/RM) → breakdown $\bar M^*$，返回一个带 `.summary()` / `.plot()` / `.to_latex()` 的结果对象。
3. 尝试装 Stata `cic`（Melly 站点）；成功则做 parity，失败则记录为 analytic-only 并在 REFERENCES.md 说明。

### Day 5 — parity 框架

1. `tests/reference_parity/_fixtures/_generate_conley_stata.py` + `.do`：合成 GIS 面板 → `acreg` / `reghdfe` → JSON fixture。
2. `tests/reference_parity/test_conley_acreg_parity.py`、`test_hdfe_slopes_parity.py`。
3. `tests/external_parity/test_rebel_canal_published.py`：论文 β=0.0380 / SE=0.0166 / $\bar M^*\approx1.9$ 作为 published-number 锚点。
4. 可复用的 `StataOracle` helper（走 stata-code MCP 或 `subprocess`），后续每篇 AER 复现共用。

### Day 6 — GIS 薄封装 + agent-native

1. `src/statspai/spatial/utils.py`（lazy import geopandas，< 200 行）：
   `line_length_in_polygon` / `share_within_buffer` / `distance_to_feature`。
2. `pyproject.toml` 补 `spatial` extra（design doc 里规划过但从未落地）。
3. `docs/guides/gis_panel_construction.md`：shapefile → 分析面板全流程。
4. §8 agent-native：
   - registry 描述模板加 `⚠️ Don't use when …` + 内存成本（先覆盖 conley / callaway_santanna / feols 等易误用项）；
   - 错误消息模板「expected X, got Y; try Z」；
   - 清掉 `agent/workflow_tools.py:977-1010` 那段永远抛错的死代码（`betas=/sigma=` 签名早已不存在）。

### Day 7 — 收口

1. `pytest -q` 全绿；`pytest tests/reference_parity/ -q` 必过。
2. `black` / `flake8` / quality gate。
3. CHANGELOG（`⚠️ Correctness` 段：varying-slope 静默丢弃、event_study vcov、CIC bootstrap、Conley 坐标错位）+ MIGRATION。
4. `python scripts/registry_stats.py --check` 无漂移；docs 同步。

---

## 2. 风险

| 风险 | 缓解 |
|---|---|
| Day 3 的 vcov 修复会改变已发布的 honest-DiD 数值 | 明确标 **⚠️ correctness fix**；MIGRATION 给出新旧对照；不静默 |
| varying-slope 投影收敛慢（交替投影 + 斜率） | 沿用现有 Irons-Tuck 加速；必要时退 LSMR Krylov 通路 |
| Stata `cic` 装不上 | 退回解析/自洽测试，并在 REFERENCES.md 标注为 analytic-only |
| 无论文数据 → parity 说服力打折 | 框架先行；openICPSR 到手后直接复用。不假称「已复现 AER 论文」 |
| JOSS 审稿（issue #10604） | 全部改动不发 GitHub Release，不触发 Zenodo 归档；PyPI 发版与 commit 不影响审稿 |

## 3. 不做

按原文 §9：不新造 DML/RL/RDD 估计器。所有工作集中在**接口体验**与**论文级工作流**，外加实测暴露出的**静默正确性缺陷**。
