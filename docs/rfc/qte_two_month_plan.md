# QTE / 分布处理效应 两个月工作计划（2026-07-31 起）

> 目标：把 `sp.qte` 家族从"6 文件 / 2k LOC、零参考对齐、薄且无锚"推进到
> **与 R `qte` 1.3.1 / `quantreg` 6.1 数值对齐、估计量标签正确、带解析 SE 与一致置信带**的完整实现。
> 每个工作包（WP）的验收标准是**对齐参考实现的数值证据**或**已知真值的解析恢复**，不是"代码写完了"。

---

## 0. 现状基线（2026-07-31 实测，非推测）

### 0.1 模块清单

| 文件 | LOC | 对外函数 | 参考对齐 |
|---|---:|---|---|
| `qte/qte.py` | 647 | `qte`, `qdid` | 无 |
| `qte/distributional.py` | 498 | `distributional_te` | 无 |
| `qte/dist_iv.py` | 302 | `dist_iv`, `kan_dlate` | 无 |
| `qte/beyond_average.py` | 281 | `beyond_average_late` | 无 |
| `qte/hd_panel.py` | 248 | `qte_hd_panel` | 无 |
| `qte/__init__.py` | 41 | — | — |
| **合计** | **2,017** | **7** | **0** |

现有 4 个 `tests/reference_parity/test_*qte*` / `test_dist_iv_*` / `test_distributional_te_*` /
`test_beyond_average_late_*` 文件**全部是自设计的 location-shift 解析测试**，不是跨软件对齐。
Location-shift（常数效应）是 QTE 的**退化情形**——真值在所有 τ 上是一条水平线，
任何"能算出一个数"的估计量都能通过。这正是"thin + unanchored"的来源：
**测试挑了唯一一个 QTE 与 ATE 无法区分的 DGP。**

### 0.2 已实测确认的缺陷（每条都有复现脚本，不是代码审读推测）

#### 🔴 D1 — `dist_iv` / `kan_dlate` 估计量错误（渐近偏误，非噪声）

实现的是"分位数的 Wald 比"：
`LATE_q = [Q(τ|Z=1) − Q(τ|Z=0)] / [E(D|Z=1) − E(D|Z=0)]`（`dist_iv.py:152-172`）。

**分位数不是线性算子，Wald 比对分位数不成立。** 实测（n=200,000，
always-taker 30% / complier 50% / never-taker 20%，complier 真值 QTE(τ) ≡ 2.0）：

| τ | 真值 | `sp.dist_iv` | `sp.beyond_average_late` |
|---:|---:|---:|---:|
| 0.25 | 2.0 | **3.971** | 2.003 |
| 0.50 | 2.0 | **4.028** | 2.014 |
| 0.75 | 2.0 | **0.000** | 2.023 |

偏误恰为 `1/Δp = 1/0.5 = 2×`——这是把只对均值成立的 Wald 缩放硬套到分位数上的直接后果。
τ=0.75 返回 0.0 是分位数网格穿过 always-taker 质量点的产物。

**讽刺的是同一模块的 `beyond_average_late` 用 Abadie (2002) κ 加权 CDF 做对了。**
`dist_iv` 只需委托给它。

附带问题：`covariates=` 参数被接收后**静默丢弃**（`dist_iv.py:141` 算出 `X` 后再未使用）——
违反 §7"失败要响亮"。`kan_dlate` 是 `dist_iv` 的纯 alias，docstring 却称
"functional equivalence is preserved"——不实陈述，且继承全部偏误。

#### 🔴 D2 — `distributional_te` 的 KS 检验 p 值无效（零功效）

`distributional.py:494`：`ks_pvalue = mean(boot_ks >= ks_stat)`。
bootstrap KS 统计量**未中心化**，其分布随机大于观测值，所以 p 值恒偏大。

实测（H0 为真：完全无处理效应，40 个种子，n=400，n_boot=200）：

```
H0 下 KS p 值: min=0.565  median=0.808  max=0.995
5% 名义水平下的拒绝率: 0.000   （应为 ~0.05）
```

**p 值从未低于 0.565**——这个检验在任何常规水平下都不可能拒绝，
它不是 p 值。正确做法是用 `sup|DTE_boot − DTE_hat|` 重新中心化。

#### 🔴 D3 — `qte_hd_panel` 无法识别非位置移动的分布效应

`hd_panel.py:170-176` 先对 Y 和 D 做 unit/time 双向去均值，再对去均值数据跑分位数回归。
**分位数回归对组内变换不是不变的**（Koenker 2004; Canay 2011）——去均值改变了条件分位函数。

实测（scale-shift DGP：`Y_it = u_i + (1 + d_it)·e_it`, `e~N(0,1)`，
真值 QTE(τ) = Φ⁻¹(τ) 是一个**扇形**，T=4，n=2,000）：

| τ | 真值 | `qte_hd_panel` |
|---:|---:|---:|
| 0.10 | −1.282 | **−0.707** |
| 0.25 | −0.674 | **−0.391** |
| 0.50 | 0.000 | **−0.148** |
| 0.75 | 0.674 | **0.299** |
| 0.90 | 1.282 | **1.108** |

扇形被压平约 45%。**即：这个估计量在它唯一存在理由（探测分布异质性）上系统性失败。**
在常数效应 DGP 上它能恢复 1.2（已实测），所以现有测试全绿——测试选错了 DGP。

#### 🔴 D4 — `qte_hd_panel` 伪造标准误

`hd_panel.py:196` 和 `hd_panel.py:207`：当 `statsmodels.QuantReg` 抛异常或
statsmodels 不可用时，`se_list.append(0.1)` / `se_arr = np.full(len(quantiles), 0.1)`。
**硬编码 0.1 作为标准误**，然后据此算出 CI 并正常返回。
这是 §12"不要吞异常返回 None/NaN"的加强版违规——它不是返回 NaN，而是**编造推断**。

同文件其它问题：
- LASSO 只对 **Y ~ X** 做选择（`hd_panel.py:155`），忽略 D 方程 → 不是
  Belloni-Chernozhukov-Hansen 双重选择，失去 immunization 性质；
  docstring 却声称 "double/debiased quantile regression"。
- `seed` 参数从未使用（`hd_panel.py:149` 赋给 `_`）。
- LASSO 在未标准化的 X 上跑，`lasso_alpha` 尺度依赖。
- `except Exception:` 裸吞（`hd_panel.py:161`）。

#### 🟠 D5 — `sp.qte(method="quantile_regression")` 估计量标签错误

实现的是 Y 对 D+X 的**条件**分位数回归中 D 的系数（`qte.py:496-504`）。
Firpo (2007) 是**无条件** QTE——倾向得分加权、只对 D 回归。二者是不同的估计量：
条件 QTE 在没有秩不变性假设时没有因果解释。docstring、registry、`method` 标签
（`"QTE via Quantile Regression (Firpo, 2007)"`）全部需要改。

#### 🟠 D6 — `sp.qte(method="distribution")` 实际是 QTT 不是 QTE

`qte.py:583-593`：处理组分位数**未加权**，控制组按 `p/(1−p)` 加权 →
反事实是 `F_{Y(0)|D=1}`，得到的是 **QTT**（对处理组的分位数效应），不是 QTE。
Firpo (2007) 的 QTE 需要处理组按 `1/p`、控制组按 `1/(1−p)` 加权。

#### 🟠 D7 — `qdid` 的 Athey-Imbens 引用错误

`qte.py:255-258` 实现的是
`QTE(τ) = Q₁₁(τ) − Q₁₀(τ) − [Q₀₁(τ) − Q₀₀(τ)]`——这是 **QDiD**（分位数上的朴素 DiD）。
Athey & Imbens (2006) 提出的是 **Changes-in-Changes**，其反事实是
`F_{Y(0)|11}(y) = F₀₁(F₀₀⁻¹(F₁₀(y)))`，且论文**明确批评** QDiD。
R `qte` 包把二者分成 `QDiD()` 和 `CiC()` 两个函数，正是因为它们不同。
当前 method 字符串写 `"Quantile DID (Athey & Imbens, 2006)"` → 归错作者。

同时仓库里有**三条 CiC 代码路径**：`sp.cic`（did 模块）、
`distributional_te(method="cic")`、以及 `qdid` 被误标为 CiC。违反 §3.4 dispatcher 原则。

#### 🟠 D8 — `kan_dlate` 引用自相矛盾

模块 docstring（`dist_iv.py:2-3`）："KAN-Powered D-IV-LATE (**Kennedy** 2025, arXiv 2506.12765)"；
函数 docstring（`dist_iv.py:266`）："KAN-Powered D-IV-LATE (**Shaw** 2025, arXiv 2506.12765)"。
同一 arXiv ID 两个作者 → §10 零幻觉红线。落地前必须 Crossref/arXiv 双渠道核验，
核验不通过则删除该函数或改为明确的"未实现"错误。

#### 🟡 D9-D15 — 结构性缺口

- **D9** 全家族只有 bootstrap SE，无解析影响函数 SE。Firpo (2007) 有闭式影响函数；
  bootstrap 在 `n_boot=500 × 5 分位 × 500 IRLS 迭代`下也慢。
- **D10** 无一致置信带（uniform confidence band）、无分位数交叉修正
  （rearrangement, Chernozhukov-Fernández-Val-Galichon 2010）。
  分布估计量不给一致推断，等于只能做逐点断言，而 QTE 的核心命题
  （"效应在整个分布上为零/恒定/单调"）全部是**函数型**假设。
- **D11** 结果对象不满足 §3.3 契约：`DistIVResult` / `HDPanelQTEResult` /
  `BeyondAverageResult` 无 `.plot()` / `.to_latex()` / `.to_word()` / `.to_excel()` /
  `.cite()` / `_repr_html_`；`QTEResult` / `DTEResult` 无 `.to_frame()`。
- **D12** 无 dispatcher。7 个函数 + `cic` + `ivqreg` + `rifreg` 各自为政，
  违反 §3.4（对照 `sp.synth(method=...)`）。
- **D13** `distributional.py:478-482` `except Exception: boot[b] = nan` 裸吞，
  未走 `record_degradation` → 违反 §7。
- **D14** `distributional_te` 的 `_fit_cond_cdf_ctrl`（`distributional.py:232-246`）
  用 **LinearRegression** 拟合 `P(Y≤y|X)`——线性概率模型估 CDF，clip 到 [0,1]，
  不保证单调。模块 docstring 引的 Chernozhukov-Fernández-Val-Melly (2013)
  正是 **distribution regression**（logit 链接），实现却退化成 LPM。
  `_quantile_from_cdf`（`distributional.py:219-229`）把分位数离散到 `n_grid=100` 的网格点上。
- **D15** `qte.py` 自己实现 IRLS 分位数回归求解器 `_qreg_coef`（`qte.py:378-403`），
  而仓库已有 `sp.qreg` 且 `hd_panel.py` 用 statsmodels。违反 §4"不要在多个文件重复实现"。
  （实测该 IRLS **确实收敛**到 statsmodels 解，5 个 τ 上差异 < 1e-6，
  所以这是重复+性能问题，**不是**正确性问题——不夸大。）

### 0.3 参考环境（已就绪，实测）

```
R 4.5.2 (2025-10-31)
quantreg 6.1          # rq, rq.fit.br (Barrodale-Roberts), summary.rq
qte      1.3.1        # ci.qte, ci.qtet, QDiD, CiC, MDiD, ddid2, panel.qtet, spatt, bounds
did      2.3.0
```

`qte` 包自带数据集：`lalonde.exp`, `lalonde.psid`, `lalonde.exp.panel`, `lalonde.psid.panel`
——**这是本计划全部横截面/面板锚点的来源**（包内数据，无需外部下载，可复现）。

缺失、需按需安装：`Counterfactual`（CFM 2013 反事实分布）、`IVQR`、`qrcm`。

### 0.4 已核验可用的 bib key（`paper.bib` 现存）

`firpo2007efficient`, `firpo2009unconditional`, `athey2006identification`,
`chernozhukov2005model`, `chernozhukov2006instrumental`, `chernozhukov2008instrumental`,
`koenker1978regression`, `koenker2005quantile`, `machado2005counterfactual`,
`melly2005decomposition`, `melly2006estimation`, `belloni2016post`, `chernozhukov2013inference`

**需新增并按 §10 双渠道核验**（Crossref + arXiv/期刊官网，落地时现场核验，不凭记忆）：
Abadie (2002 JASA)、Callaway & Li (2019 QE)、Chernozhukov-Fernández-Val-Melly (2013 Ecta)、
Chernozhukov-Fernández-Val-Galichon (2010 Biometrika)、Frölich & Melly (2013 JBES)、
Canay (2011 Econometrics J.)、Koenker (2004 JMVA)、Athey-Imbens 的 CiC 离散界。
`kan_dlate` 的 arXiv 2506.12765 作者归属必须核验后才能保留该函数。

---

## 1. 优先级排序原则

排序依据**用户被误导的严重程度**，不是实现难度：

1. **返回错数的函数**（D1, D2, D3, D4）——用户拿到的数字是错的且无警告 → 最高优先。
2. **标签错的函数**（D5, D6, D7, D8）——数字可能对，但被归给错误的估计量/作者 → 次高。
3. **缺失的能力**（Callaway-Li 面板 QTT、一致推断、rearrangement）→ 第三。
4. **契约与工程**（dispatcher、result 方法、docs）→ 最后，但必须做完。

**所有 D1-D4 的修复都是 ⚠️ correctness fix**，必须进 CHANGELOG + MIGRATION（§12）。

---

## WP-0 · R 参考夹具基础设施 — 第 1 周

**产出**
- `tests/reference_parity/_fixtures/_generate_qte_R.R`（持久化生成器，可重跑）
- `tests/reference_parity/_fixtures/qte_R.json`（提交入库的参考值）
- `tests/reference_parity/_fixtures/qte_lalonde.csv` / `qte_lalonde_panel.csv`
  （从 R `qte` 包导出，锁定行序与列名）

**覆盖的 R 调用**（全部在 `lalonde.exp` / `lalonde.psid` 上，`τ ∈ {.05,.1,…,.95}`）

| 类别 | R 调用 |
|---|---|
| 无条件 QTE | `qte::ci.qte(re78 ~ treat, data=lalonde.exp, probs=…, se=TRUE, iters=…)` |
| 无条件 QTT | `qte::ci.qtet(re78 ~ treat, …)` |
| 带协变量 QTE/QTT | `ci.qte(…, xformla = ~ age + education + black + hispanic + married + nodegree)` |
| QDiD | `qte::QDiD(re ~ treat, t=1978, tmin1=1975, idname="id", tname="year", data=lalonde.psid.panel, panel=TRUE)` |
| CiC | `qte::CiC(…)` |
| MDiD | `qte::MDiD(…)` |
| 面板 QTT | `qte::panel.qtet(…)` |
| 条件分位数回归 | `quantreg::rq(re78 ~ treat + age + …, tau=…)` + `summary(…, se="boot")` |

**两个已知的夹具坑（写进生成器注释）**
1. `qte` 包的 `se=TRUE` 走 bootstrap，**必须固定 `set.seed()` 且记录 `iters`**，
   否则 SE 不可复现。点估计与种子无关，先锚点估计。
2. `lalonde.psid.panel` 的 `year` 是 1975/1978 两期；`QDiD`/`CiC` 的
   `t` / `tmin1` 参数语义是"后期/前期"，写反会静默给出符号相反的结果。

**验收**：`qte_R.json` 存在、包含上表全部条目、`_generate_qte_R.R` 重跑后点估计逐位一致。
本 WP 不改任何 Python 源码——它只是把"参照物"钉死。

---

## WP-1 · ⚠️ `dist_iv` / `kan_dlate` 正确性修复 — 第 1–2 周

**问题**：D1（2× 渐近偏误）+ D14（协变量静默丢弃）+ D8（引用矛盾）。

**方案**
1. `dist_iv` 的无协变量路径**改为委托** `beyond_average.py` 的 Abadie κ 加权 CDF
   （该实现已实测正确）。把 `_complier_cdfs` / `_invert_cdf` 提升到新的
   `qte/_core.py`，两处共用——同时消化 §4"共享基元放 `_core.py`"。
2. 有协变量时实现 Frölich & Melly (2013) 的无条件 IV-QTE：
   用倾向得分 `P(Z=1|X)` 构造 Abadie 权重，再对加权样本求分位数。
   协变量不再被丢弃。
3. `kan_dlate`：arXiv 2506.12765 作者归属**必须先核验**。
   - 核验通过 → 保留函数，修正 docstring 为唯一正确署名，
     并明确写"当前回退到 `dist_iv` 的核估计路径，KAN 桥函数未实现"（诚实降级，不说"等价"）。
   - 核验失败 / 论文不存在 → 删除 `kan_dlate`，走 §4 弃用流程
     （`DeprecationWarning` + MIGRATION 登记 + 一个小版本缓冲）。
4. 加解析影响函数 SE（Abadie κ 权重的 IF 是闭式的），bootstrap 降级为可选。

**验收**
- 0.2 节 D1 那张表（n=200k，always/complier/never = 3/5/2）：
  三个 τ 上 `|est − 2.0| < 0.02`；这是**回归测试**，直接进
  `tests/reference_parity/test_dist_iv_parity.py`（替换现有 location-shift 测试）。
- 对齐 R：`qte::bounds` 或手写 Abadie κ 的 R 实现，点估计 ≤1e-8。
- `covariates=` 传入后结果**必须**改变（现在不变）——加断言。
- CHANGELOG `⚠️ Correctness` + MIGRATION 条目，写明旧版偏误方向与量级。

**风险**：`dist_iv` 旧数值被用户引用过。MIGRATION 必须给出
"旧值 ≈ 新值 / Δp"的换算关系，让用户能自查。

---

## WP-2 · Firpo (2007) 真正的无条件 QTE/QTT + 解析 SE — 第 2–3 周

**问题**：D5（条件 QR 冒充 Firpo）+ D6（QTT 冒充 QTE）+ D9（无解析 SE）+ D15（重复求解器）。

**方案**
1. 新增 `qte/_firpo.py`，实现 Firpo (2007) 无条件 QTE 与 QTT：
   - QTE：处理组权重 `D/p̂(X)`，控制组权重 `(1−D)/(1−p̂(X))`；
   - QTT：处理组权重 `D`，控制组权重 `(1−D)·p̂/(1−p̂)`；
   - 加权分位数由加权 check-function 最小化得到（一维，用精确的加权分位数而非网格）。
2. **解析 SE**：Firpo (2007) Theorem 3 的影响函数
   `ψ_τ(y,d,x) = [w·(τ − 1{y ≤ q_τ})] / f_{Y}(q_τ) + 倾向得分估计的修正项`，
   密度 `f` 用核估计（复用 `rd/_core.py` 的 kernel 基元，§11）。
3. `sp.qte` 的 `method=` 扩展为
   `{"firpo_qte", "firpo_qtt", "conditional_qr", "distribution"}`：
   - `"conditional_qr"` = 现有 QR 路径，**method 标签改为
     "Conditional QTE via Quantile Regression (Koenker & Bassett, 1978)"**，
     移除 Firpo 归属；
   - `"distribution"` 保留但标签改为 QTT，并在 docstring 明确估计量；
   - 旧值 `"quantile_regression"` → `DeprecationWarning` 指向 `"conditional_qr"`。
4. `_qreg_coef` 删除，`"conditional_qr"` 改调 `sp.qreg`（消化 D15）。

**参考值**：WP-0 的 `ci.qte` / `ci.qtet`（含 `xformla` 版本）。

**验收**
- `firpo_qte` / `firpo_qtt` 对 `qte::ci.qte` / `ci.qtet` 点估计 ≤1e-6（19 个 τ 全过）。
- 带协变量版本 ≤1e-6。
- 解析 SE 对 R bootstrap SE ≤5%（bootstrap SE 本身有 MC 误差，5% 是合理带宽，
  在测试里就地注明理由）。
- `conditional_qr` 对 `quantreg::rq` 系数 ≤1e-8。
- **非退化 DGP 回归测试**：location-scale DGP（真值是扇形，不是水平线），
  确认估计量能恢复扇形——直接封死"测试选退化 DGP"这个洞。

---

## WP-3 · QDiD / CiC / MDiD 拆分与正名 — 第 3–4 周

**问题**：D7（Athey-Imbens 归错）+ 三条 CiC 代码路径。

**方案**
1. `qdid` 的 method 标签改为 `"Quantile DiD (QDiD)"`，
   docstring 引用改为 Koenker-Bassett 的分位数 + DiD 组合，
   并**显式写明 Athey & Imbens (2006) §5 对 QDiD 的批评**及何时该改用 CiC
   （诚实标注局限，而不是悄悄改名）。
2. 新增 `sp.qdid(method=...)` 分发：`{"qdid", "cic", "mdid", "ddid2"}`，
   对齐 R `qte` 包的四个函数。`"cic"` 委托到既有 `sp.cic`（不新起一份实现，§12），
   `distributional_te(method="cic")` 内部改为调用同一核心，三路径合一。
3. `sp.cic` 补齐 Athey-Imbens 的**离散结果界**（连续 CiC 在离散 Y 上只识别区间）。

**参考值**：WP-0 的 `qte::QDiD` / `CiC` / `MDiD` / `ddid2`（`lalonde.psid.panel`）。

**验收**：四个方法点估计对 R ≤1e-6；三条 CiC 路径在同一数据上互相 ≤1e-10（同一核心）；
`sp.cic` 离散界包含真值的仿真覆盖率 ≥95%。

---

## WP-4 · ⚠️ `distributional_te` 推断与估计修复 — 第 4–5 周

**问题**：D2（KS p 值无效）+ D14（LPM 估 CDF、网格离散化）+ D13（裸吞异常）。

**方案**
1. **KS p 值重新中心化**：bootstrap 统计量改为 `sup_y |DTE_b(y) − DTE_hat(y)|`，
   p 值 = `mean(boot_centered >= ks_stat)`。同时加 Cramér-von Mises 变体。
2. `_fit_cond_cdf_ctrl` 的 LinearRegression 换成 **distribution regression**
   （逐 y 的 logit），即 Chernozhukov-Fernández-Val-Melly (2013) 的原始方法，
   与模块 docstring 的引用对齐；输出跨 y 做 rearrangement 保证单调。
3. `_quantile_from_cdf` 改为对 CDF 做插值反演（不再吸附到 `n_grid` 网格点）。
4. DR 路径的 Hájek/Horvitz-Thompson 归一化统一（现在 IPW 项未归一、分母用 `n1`）。
5. `except Exception` 改走 `record_degradation`（§7），并把失败重抽次数写进
   `result.degradations`。
6. 接 WP-7 的一致置信带。

**验收**
- **H0 校准测试**（0.2 节 D2 的复现脚本转成测试）：无效应 DGP 下，
  200 个种子的 KS p 值 KS-检验对 Uniform(0,1) 不拒绝；5% 拒绝率 ∈ [0.02, 0.09]。
- **功效测试**：位置移动 δ=0.5 下拒绝率 ≥0.8（防止"改成永远不拒绝"这种假修复）。
- 反事实 CDF 对 R `Counterfactual::counterfactual`（需安装）≤1e-4，
  或在 R 不可得时对手写 distribution regression 的 R 实现对齐。
- CHANGELOG `⚠️ Correctness`：旧 KS p 值不可用，明确告知受影响版本。

---

## WP-5 · ⚠️ `qte_hd_panel` 重建 — 第 5–6 周

**问题**：D3（去均值毁掉分位数识别）+ D4（伪造 SE）+ LASSO 单方程选择。

**方案**
1. **估计量换成 Canay (2011) 两步法**：
   第一步用均值 FE 估 `α̂_i`（位置效应），第二步对 `Y_it − α̂_i` 跑分位数回归。
   Canay 的关键条件是**个体效应是纯位置移动**——这一条必须在 docstring
   和结果 `diagnostics` 里显式声明，并提供检验（比较不同 τ 下的 α̂ 稳定性）。
2. 同时提供 Koenker (2004) 惩罚化 FE-QR 作为 `method="koenker_fe"`
   （对个体效应不做位置移动假设，代价是 λ 需要调）。
3. **LASSO 改双重选择**（Belloni-Chernozhukov-Hansen）：分别对 `Y ~ X` 与 `D ~ X`
   选择，取并集；X 先标准化；`lasso_alpha` 默认改为理论惩罚或交叉验证。
4. **删除所有 `se = 0.1` 伪造路径**。statsmodels 不可用 → 抛 `ImportError`；
   QuantReg 不收敛 → 该 τ 的 SE 返回 `NaN` 并 `warnings.warn`（§7）。
5. 实现 bootstrap SE（cluster on unit）并真正使用 `seed`。

**验收**
- **扇形恢复测试**（0.2 节 D3 的 scale-shift DGP 转成测试）：
  Canay 路径在 5 个 τ 上 `|est − Φ⁻¹(τ)| < 0.15`（T=8, n=4000）。
  这是本 WP 的核心验收——旧实现在此测试上偏 45%，必然失败。
- 常数效应 DGP 仍恢复真值（不回归）。
- 双重选择在"X 只通过 D 方程相关"的 DGP 上选中该变量（单方程 LASSO 会漏）。
- 无任何路径返回硬编码 SE：`grep -n "0.1" hd_panel.py` 人工复核 + 测试断言
  fallback 抛异常而非返回数字。

---

## WP-6 · Callaway & Li (2019) 面板 QTT — 第 6 周

**缺口**：面板数据下的 QTT 需要 copula 假设来把两期边际分布拼成联合分布。
R `qte::panel.qtet` 实现了这一支，StatsPAI 完全没有。

**方案**：新增 `sp.panel_qtet`，实现 Callaway-Li (2019) 的 copula-stability 估计量；
接入 WP-7 的一致推断；registry 注册 + dispatcher 挂载。

**参考值**：WP-0 的 `qte::panel.qtet`（`lalonde.psid.panel`）。

**验收**：点估计对 R ≤1e-6；copula-stability 假设违背时发 `warnings.warn` 并写入 `diagnostics`。

**风险**：Callaway-Li 的 copula 步骤对 R 包实现细节敏感（分位数插值方式）。
若 1e-6 卡住，先交付 ≤1e-4 并在测试里注明差异来源；不放宽到"数量级一致"。

---

## WP-7 · 一致推断与分位数交叉修正（`qte/_core.py`） — 第 7 周

**缺口**：D10。这是把整个家族从"逐点估计"抬到"函数型推断"的一层，所有 WP 共用。

**方案**：新建 `qte/_core.py`，提供：
1. `_multiplier_bootstrap(influence_fn_matrix, ...)` — 乘子 bootstrap，
   复用 `did/` 已有的乘子 bootstrap 基元（§4 不重复实现）。
2. `_uniform_band(...)` — 基于 `sup_τ |·|` 临界值的一致置信带。
3. `_rearrange(q_curve)` — Chernozhukov-Fernández-Val-Galichon (2010) 重排，
   消除分位数交叉；作为所有分位曲线的后处理（可 `rearrange=False` 关闭）。
4. `_ks_test` / `_cvm_test` — 正确中心化的"无分布效应"/"常数效应"/
   "随机占优"函数型检验。

所有 result 对象加 `ci_lower_uniform` / `ci_upper_uniform` 字段，
`.plot()` 同时画逐点带与一致带。

**验收**
- 一致带的仿真覆盖率 ∈ [0.93, 0.97]（名义 95%，1000 次重复）；
  逐点带的**同时**覆盖率显著低于 95%（证明一致带确实更宽、有意义）。
- rearrange 后曲线严格单调（断言）；rearrange 只减小估计误差
  （CFG 2010 定理：重排的 Lᵖ 距离不增）——用仿真验证。
- KS / CvM 检验在 H0 下校准、H1 下有功效（同 WP-4 的双向验收）。

---

## WP-8 · Dispatcher 与结果契约 — 第 7 周

**问题**：D11 + D12。

**方案**
1. `sp.qte(method=...)` 升级为家族总入口，覆盖
   `firpo_qte` / `firpo_qtt` / `conditional_qr` / `distribution` /
   `qdid` / `cic` / `mdid` / `panel_qtet` / `iv`（→ `dist_iv`）/ `hd_panel`。
   现有专用函数全部保留（不破坏 API），dispatcher 只是统一入口。
2. 新增 `sp.qte_compare()`——对照 `sp.synth_compare()`，一次跑多个估计量出对照表。
3. 所有 5 个 result 类补齐 §3.3 契约：
   `.summary()` `.plot()` `.to_frame()` `.to_latex()` `.to_word()` `.to_excel()` `.cite()`
   `_repr_html_`。统一继承一个 `_QTEResultBase`（放 `_core.py`）。
4. registry 补全所有新函数 + `sp.function_schema` 自检通过。

**验收**：`tests/test_export_surface_contract.py` 扩展到 QTE 全家族并通过；
`python3 -c "import statspai as sp; sp.help('qte')"` 列出全部 method；
`scripts/registry_stats.py --check` 无漂移。

---

## WP-9 · 文档、CHANGELOG、覆盖率收口 — 第 8 周

1. `docs/guides/choosing_qte_estimator.md`——对照已有的
   `choosing_did_estimator.md` / `choosing_matching_estimator.md` 写：
   决策树（横截面/面板/IV/高维）、估计量-假设对照表、
   **每个估计量的已知局限**（含 QDiD vs CiC、Canay 的位置移动假设）。
2. `CHANGELOG.md`：D1-D4 四条全部标 `⚠️ Correctness`；
   `MIGRATION.md`：`dist_iv` 换算关系、`method="quantile_regression"` 弃用、
   `kan_dlate` 去留结论、KS p 值失效版本区间。
3. `paper.bib`：WP-0 列出的新引用**按 §10 双渠道核验后**入库；
   全模块 docstring 的 `References` 段只留 bib key。
4. 覆盖率：`pytest --cov=statspai.qte` ≥95%（核心估计器标准，§5）。
5. `tests/reference_parity/REFERENCES.md` + parity index 更新，
   QTE 家族从 `api_stable` 升到 `certified`。

---

## 2. 交付节奏与自检

| 周 | WP | 主产出 |
|---|---|---|
| 1 | WP-0 | R 夹具 `qte_R.json` |
| 1–2 | WP-1 | ⚠️ `dist_iv` 修复 |
| 2–3 | WP-2 | Firpo QTE/QTT + 解析 SE |
| 3–4 | WP-3 | QDiD/CiC/MDiD 正名与合流 |
| 4–5 | WP-4 | ⚠️ KS 推断 + 分布回归 |
| 5–6 | WP-5 | ⚠️ `qte_hd_panel` 重建 |
| 6 | WP-6 | Callaway-Li 面板 QTT |
| 7 | WP-7 + WP-8 | 一致推断 + dispatcher/契约 |
| 8 | WP-9 | 文档、CHANGELOG、覆盖率 |

**每个 WP 的完成定义（DoD）**
1. 参考对齐测试进 `tests/reference_parity/`，且**用非退化 DGP**
   （真值随 τ 变化）——这是本计划对"thin + unanchored"的直接答复。
2. `pytest tests/ -k qte -q` 全绿；`black` + `flake8` + `mypy src/statspai/qte` 干净。
3. 正确性修复必须同时写 CHANGELOG + MIGRATION（§12）。
4. 新引用必须 §10 双渠道核验，commit message 注明 `refs verified via <src1>, <src2>`。

**对 JOSS 审稿的影响**（审稿中，issue #10604）：本计划改动集中在 `qte` 子模块，
不改 `paper.md` 的对外能力叙述，不涉及 GitHub Release（不触发 Zenodo 归档）。
若 WP-1 决定删除 `kan_dlate`，会使对外函数计数 −1，届时需同步
`scripts/registry_stats.py` 的 canonical 数字并**主动提醒用户**是否影响审稿快照。

---

*创建：2026-07-31。0.2 节所有缺陷均有实测复现记录，非代码审读推测。*
