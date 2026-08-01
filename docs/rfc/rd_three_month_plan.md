# RD（断点回归）三个月工作计划（2026-08 起）

> 目标：把 `sp.rd*` 家族从"16,207 LOC / 53 函数 / 2 个函数有跨软件比对、且不进 CI"
> 推进到**与 R `rdrobust` 3.0.0 / `rddensity` 2.6 / `rdlocrand` 2.0 / `rdmulti` 2.0.0 /
> `rdpower` 3.0 数值对齐、且由 pytest 冻结夹具守住**的状态。
>
> 方法论沿用 QTE 计划：**先用可复现脚本测出缺陷，再动手**；
> 验收标准是对齐证据或已知真值恢复，不是"代码写完了"。

---

## 0. 现状基线（2026-08 实测）

### 0.1 规模与证据落差

| | QTE（已完成） | RD（本计划） |
| --- | --- | --- |
| LOC | 2,017 | **16,207** |
| 对外函数 | 7 | **53** |
| 有跨软件锚点的函数 | 0 → 8 | **2**（`rdrobust`/`rddensity`）|
| 锚点是否进 CI | — | **否**（`tests/r_parity/` 是脚本产物，不是 pytest）|

`tests/reference_parity/test_rd_parity.py`：8 个测试，**全部是解析测试**（fuzzy 恢复、
退化输入），没有任何 R 比对。`tests/r_parity/06_rd.py` / `09_rddensity.py` 各覆盖
**1 个**函数，产出 JSON 供 JSS 附录用，**回归时不会让 CI 变红**。

即：**51/53 个 RD 函数没有任何跨软件数值锚点。**

### 0.2 已实测确认的缺陷（复现脚本，不是代码审读）

数据：`rdrobust` 自带 `rdrobust_RDsenate`（n=1,297，`y=vote`, `x=margin`, `c=0`）。
规格网格：`bwselect` × `p ∈ {1,2}` × `kernel ∈ {triangular, uniform, epanechnikov}`。

#### 🔴 A — MSE 最优带宽公式错误（3–5 倍过窄，且与 p 无关）

| 规格 | R `h` | sp `h` | 倍数 |
| --- | ---: | ---: | ---: |
| `mserd` p1 triangular | 17.754 | **4.633** | 3.83× |
| `mserd` p2 triangular | 22.256 | **4.633** | 4.80× |
| `mserd` p1 uniform | 11.597 | **4.198** | 2.76× |
| `mserd` p2 uniform | 18.765 | **4.198** | 4.47× |
| `cerrd` p1 triangular | 12.407 | **3.775** | 3.29× |

**24/24 个规格带宽偏差 >5%。** 注意 p1 与 p2 的 sp 带宽**完全相同**——
选择器根本没用到多项式阶数。sp 全部 36 个规格只产出 12 个不同带宽值
（3.26–5.03），R 的取值域是 8.01–28+。**sp 的带宽只随 kernel 变，
R 随 kernel、p、bwselect 三者变。**

`sp.rdbwselect`（独立函数）与 `sp.rdrobust` 内部走同一份 `rd/bandwidth.py`，
所以两个对外函数同时错。

#### 🔴 B — 处理效应因此系统性错误

| 规格 | R conventional | sp conventional | R robust | sp robust |
| --- | ---: | ---: | ---: | ---: |
| `mserd` p1 triangular | 7.414 | **12.621** | 7.507 | **12.395** |
| `mserd` p1 uniform | 7.202 | **12.788** | 7.593 | **12.726** |
| `mserd` p2 triangular | 8.045 | **12.395** | 8.317 | **11.281** |

**24/24 conventional 偏差 >1%；23/24 robust 偏差 >1%。**
在 `rdrobust` 教科书数据集上，`sp.rdrobust` 默认输出 **12.39，正确答案是 7.41**——
高估 67%。

**关键定位证据**：把 R 的带宽显式传进去（`h=17.7544`），
sp 的 conventional 立刻变成 **7.4141，与 R 逐位一致**。
→ **局部多项式拟合引擎是对的，缺陷完全隔离在带宽选择。**

#### 🔴 C — 偏差带宽 b 从未计算（`b == h`）

24/24 规格中 `bandwidth_b == bandwidth_h`。R 的 b 显著大于 h
（p1 triangular：h=17.75, b=28.03）。
讽刺的是 `sp.rdbwselect` **确实**算出了一个不同的 b（7.142），
但 `sp.rdrobust` 把它丢掉、令 b = h。所以这是与 A 独立的第二个 bug：
即使 A 修好，稳健偏差修正仍然错。

#### 🟠 D — `bwselect` 取值与 R 不兼容

`msesum` / `cersum` 直接抛 `ValueError`（sp 用 `msecomb1`/`msecomb2`）。
R 的 6 个 MSE 变体里 sp 只认 4 个，命名还不同 → 迁移用户的脚本直接崩。

### 0.3 参考环境（已就绪，实测）

```
R 4.5.2
rdrobust  3.0.0     rddensity 2.6      rdlocrand 2.0
rdmulti   2.0.0     rdpower   3.0      RDHonest  未装（WP-6 需要）
```

---

## 1. 优先级原则（同 QTE）

1. **返回错数的函数** → 最高。A/B/C 影响 `rdrobust`/`rdbwselect`，
   是全仓被引用最多的 RD 入口。
2. **API 不兼容**（D）→ 次高，迁移即崩。
3. **无锚点的能力** → 第三，按使用频率排序。
4. **契约与文档** → 最后但必须做完。

**A/B/C 修复一律 ⚠️ correctness fix**，进 CHANGELOG + MIGRATION。

---

## WP-1 · ⚠️ 带宽选择器重建 — 第 1–3 周

**范围**：`rd/bandwidth.py` 的 MSE / CER 带宽，`sp.rdbwselect` 与 `sp.rdrobust` 共用。

**方案**
1. 按 Calonico-Cattaneo-Titiunik (2014) + Calonico-Cattaneo-Farrell-Titiunik (2019)
   重写：pilot 带宽 → 偏差常数 → 方差常数 → `h_MSE`，并让 `b` 走独立公式。
2. `bwselect` 取值与 R 对齐：`mserd/msetwo/msesum/msecomb1/msecomb2` +
   `cerrd/certwo/cersum/cercomb1/cercomb2`；旧名保留并发 `DeprecationWarning`。
3. `rdrobust` 停止令 `b = h`，改用选择器返回的 `b`。

**验收**
- 36 规格网格对 R `rdbwselect`：`h`、`b` 相对误差 ≤1e-6（同一闭式公式，应当逐位）。
- 36 规格对 R `rdrobust`：conventional / bias-corrected / robust 三个系数
  与 SE 相对误差 ≤1e-6。
- **带宽必须随 p 变化**（专门加断言：p1 与 p2 的 h 不得相等）。
- 冻结夹具 `rdrobust_R.json` 进 `tests/reference_parity/`，pytest 守住。

**风险**：CCT 带宽有多个实现细节（regularization 项 `scaleregul`、
`vce` 选择、边界处理）。若 1e-6 卡住，逐项二分定位，**不放宽到"数量级一致"**。

## WP-2 · `rdrobust` 全参数面对齐 — 第 4–5 周

`fuzzy` / `covs` / `cluster` / `weights` / `deriv` / `scalepar` / `scaleregul` /
`vce ∈ {nn, hc0..hc3}` / `bwcheck` / `masspoints`。逐个对 R 建网格。
**masspoints 是重点**：R 默认 `masspoints="adjust"`，离散 running variable 上
不处理会同时影响带宽和方差。

**验收**：每个参数至少 6 个规格，≤1e-6；`masspoints` 在离散 DGP 上单独一组。

## WP-3 · 密度与操纵检验 — 第 6 周

`rddensity` / `rdplotdensity` / `mccrary_test` 对 R `rddensity` 2.6。
覆盖 `bino`（二项检验）、`fitselect`、`kernel`、`h` 自选。

**验收**：检验统计量与 p 值 ≤1e-6；已有 `tests/r_parity/09_rddensity.py`
升级为 pytest 冻结夹具。

## WP-4 · 局部随机化与功效 — 第 7–8 周

`rdrandinf` / `rdwinselect` / `rdsensitivity` / `rdrbounds` 对 `rdlocrand` 2.0；
`rdpower` / `rdsampsi` 对 `rdpower` 3.0。

**注意**：`rdrandinf` 是随机化推断，**必须固定种子**并记录 `reps`；
点估计确定、p 值不确定 —— 锚点估计紧、p 值松，且在测试里写明理由。

## WP-5 · 多断点 / 多维 — 第 9–10 周

`rdmc` / `rdms` 对 `rdmulti` 2.0.0；`rd2d` / `multi_score_rd` / `geographic_rd`
按 Cattaneo-Titiunik-Yu 的 2D 设计对齐（若 R 侧无对应包，走已知真值仿真 + 论文数字）。

## WP-6 · Honest / bias-aware — 第 11 周

`rd_honest` / `rd_bias_aware_fuzzy` 对 `RDHonest`（需先装）。
Armstrong-Kolesár 的 FLCI 与 sp 现有实现比对。若 `RDHonest` 装不上，
退回论文表格数字（external_parity）。

## WP-7 · 其余 33 个函数分诊 + 收口 — 第 12 周

对剩余函数逐个判定：**有 R 对应 → 对齐；无对应 → 已知真值仿真；
两者都不可行 → 在 registry 与 docs 明确标注 `analytical-only`**，
不留"看起来验证过"的模糊状态。

文档：`docs/guides/choosing_rd_estimator.md` 更新（现有指南需按 WP-1 结论重写
带宽章节）；CHANGELOG + MIGRATION；parity index 重建。

---

## 2. 每个 WP 的完成定义

1. 冻结夹具进 `tests/reference_parity/`，**pytest 守住**（不是脚本产物）。
2. 生成器内含自检：手工复算与包输出不一致就报错，不静默产出错夹具
   （QTE 的 `panel_qtet` 生成器已验证这一模式有效）。
3. **非退化设计**：RD 的退化情形是"线性 DGP + 效应恒定"——任何带宽都对。
   测试必须用曲率明显、带宽敏感的 DGP。
4. `black` / `flake8` / `mypy` 干净；正确性修复进 CHANGELOG + MIGRATION。
5. 新引用按 §10 双渠道核验。

## 3. 对 JOSS 审稿的影响

RD 是 `paper.md` 明确列出的能力。WP-1 会**改变 `sp.rdrobust` 的默认数值输出**，
这是审稿期间需要主动披露的：改动是把错的改对（有 R 逐位证据），
但任何引用过旧 RD 数字的材料都需要重跑。落地时在 CHANGELOG 用
**⚠️ correctness fix** 标注，并提醒用户是否影响审稿快照。

---

*创建：2026-08。0.2 节所有数字均来自可复现脚本，非推测。*

---

## 附录 A · WP-1 实现规格（逆向自 rdrobust 4.0.0，2026-08-01）

> 这一节是 WP-1 的**实现说明书**。R 侧的逆向工程已经做完，
> 下一步是照着它写 Python 并对 36 规格网格验证。
> 沿用 `panel_qtet` 的做法：**先在 R 里手工复算这套级联、断言与
> `rdbwselect` 差 0，再移植**——不要跳过这一步直接写 Python。

来源：`rdrobust::rdbwselect`（636 行）+ `rdrobust:::rdrobust_bw`（177 行），
用 `deparse()` 导出后阅读。

### A.1 为什么现有实现必然错

现有代码是**单步经验公式**：

```python
h = (C_K * sigma2 / (f_c * m2**2 * n)) ** (1/5)      # rd/bandwidth.py
```

R 的是**三级级联**，每级都调一次 `rdrobust_bw`，且指数是 `1/(2o+3)` 而不是常数 `1/5`。
两个已确认缺陷由此直接推出：

- `1/5 == 1/(2p+3)` **当且仅当 p == 1** → 带宽不随 p 变（实测 p1/p2 都是 4.633）。
- `b` 是级联里**独立的一级**（stage 2），不是 h 的副产品 → 现有代码令 `b = h`。

### A.2 级联结构

**预处理（三个反直觉的默认值，全部实测确认）**：

1. `stdvars = FALSE` 是**默认**——`x` 根本不标准化，`x_sd` 保持 1，
   且 `BWp = min(sd(x), x_iq/1.349)` 走未标准化分支（不是 `min(1, ...)`）。
2. `masspoints = "adjust"` 是**默认**——`c_bw` 里的 `N` 被替换成**唯一值计数 `M`**；
   任一侧并列比例 ≥20% 时还会触发 `bwcheck = 10` 的下界夹紧。
3. **`scale` 参数在三级之间不同**：stage 1 传 `0`，stage 2/3 传 `scaleregul`。
   这是最隐蔽的一个——`rdrobust_bw` 内部 `R = scale * (2*(o+1-nu)) * BWreg`，
   所以 stage 1 的正则项被完全关掉。三级都传 `scaleregul` 会让 h **系统性偏低约 11%**
   （实测：17.754 → 16.026）。

```
q = p + 1                                   （默认；用户可覆盖）
x_iq = quantile(x,.75,type=2) - quantile(x,.25,type=2)
BWp  = min(1, (x_iq/x_sd)/1.349)            （标准化后）
C_c  = 2.576 triangular | 1.843 uniform | 2.34 epanechnikov
c_bw = C_c * BWp * N^(-1/5)                 （必要时按 bw_max/bw_min 夹紧）

stage 1  d_bw = rdrobust_bw(o=q+1, nu=q+1, o_B=q+2, h_V=c_bw, h_B=range_side)
stage 2  b_bw = rdrobust_bw(o=q,   nu=p+1, o_B=q+1, h_V=c_bw, h_B=d_bw)
stage 3  h_bw = rdrobust_bw(o=p,   nu=deriv, o_B=q, h_V=c_bw, h_B=b_bw)
```

每级的收敛式（`rate = 1/(2*o+3)`，`scaleregul` 默认 1）：

```
value = ( V / (B^2 + scaleregul * R) ) ^ rate
```

### A.3 三个变体的装配

设 `l`/`r` 为左右侧的 `rdrobust_bw` 输出：

| 变体 | 分子 | 分母的偏差项 |
| --- | --- | --- |
| `mserd`（默认） | `V_l + V_r` | `(B_r − B_l)^2` |
| `msesum` | `V_l + V_r` | `(B_r + B_l)^2` |
| `msetwo` | 各侧 `V` 单独 | 各侧 `B^2` 单独 |

正则项一律 `scaleregul * (R_l + R_r)`。最后 `h_mserd = x_sd * h_bw_d`。

CER 变体 = 对应 MSE 变体乘一个 `N^(-ε)` 收缩因子（见 `rdbwselect` 尾部）。

### A.4 `rdrobust_bw` 内部（stage 的被调方）

返回 `list(V, B, R, rate)`：

- 用 `rdrobust_kweight(X, c, h_V, kernel)` 取权重，`ind_V = w > 0` 子样本
- `R_V = vander(X - c, o)`，`invG_V = qrXXinv(R_V * sqrt(w))`
- `beta_V = invG_V %*% crossprod(R_V * w, D_V)`
- `V` 来自 `vce`（默认 `nn`，`nnmatch=3` 最近邻方差）
- `B` 来自 `o_B` 阶拟合在 `h_B` 上的高阶导数
- `R` 是正则项（避免 B≈0 时带宽爆炸）

**最容易踩的三个点**：`type=2` 分位数（不是 numpy 默认）、
`nn` 方差要 `nnmatch=3`、`vander` 的列是 `(x-c)^0..o` 未除阶乘。

### A.4b 级联已在 R 侧逐位验证

`tests/reference_parity/_fixtures/_verify_rdbwselect_cascade.R` 手工复算了整套级联
（调用内部 `rdrobust:::rdrobust_bw`），对 `rdrobust::rdbwselect` 在
3 kernel × 2 p 的 6 个规格上 **h 与 b 的最大相对偏差 = 0.000e+00**。

这意味着**装配逻辑已确认无误**，剩余移植工作只有 `rdrobust_bw` 本身
（V/B/R 的计算，177 行）。移植 Python 时应先让 Python 版复现这 6 个规格，
再跑 36 规格全网格。

### A.5 验收（已就位）

`tests/reference_parity/test_rdrobust_parity.py` 的 12 个 `xfail(strict=True)`
就是本 WP 的验收单。全部转绿即完成；`strict=True` 保证不会因为放宽容差而"假通过"。
另有 2 个**非 xfail** 的测试（引擎正确性、数据一致性）必须始终绿。

---

## 附录 B · WP-1 与 WP-2 执行结果（2026-08-01）

### B.1 已修复

| 缺陷 | 状态 | 证据 |
| --- | --- | --- |
| A 带宽公式错（窄 2.8–4.8 倍、与 p 无关） | ✅ | 36 网格 h 最大偏差 5.6e-08 |
| B conventional 系数错 | ✅ | 36 网格最大偏差 4.1e-12 |
| C `b == h`（偏差带宽从未计算） | ✅ | 36 网格 b 最大偏差 1.5e-08 |
| D `msesum`/`cersum` 抛异常 | ✅ | 6 个变体全部可用 |
| **E `tau_bc` 定义错**（新发现） | ✅ | 36 网格 robust 系数最大偏差 4.4e-12 |
| E′ 稳健 SE | ✅ | 36 网格最大偏差 3.6e-12 |

在 `rdrobust` 教科书数据上 headline 从 **12.39 → 7.5065**（R: 7.5065）。

**WP-1 + WP-2 完成。`sp.rdrobust` 现在对 R `rdrobust` 4.0.0 全量对齐**
（36 规格 × 6 个量，最大相对偏差 6.2e-12）：

| 量 | 最大相对偏差 |
| --- | ---: |
| 带宽 `h` / `b` | 5.6e-08 / 1.5e-08 |
| conventional 系数 / SE | 4.0e-12 / 3.6e-12 |
| bias-corrected 系数 | 4.4e-12 |
| robust SE | 3.6e-12 |
| robust 置信区间 | 6.2e-12 |

`test_rdrobust_parity.py` 的 12 个 `xfail(strict)` **全部转绿并已摘除标记**。

### B.2 独立确认：Lee (2008)

两个 `external_parity` 测试把 bug 当发表值 pin 住了，且与它们引用的文档矛盾：

| | 值 | 相对 Lee (2008) Table 2 的 0.080 |
| --- | ---: | ---: |
| 原 pin（自称 published） | 0.0616 | −23% |
| 修复后 | 0.0768 | −4% |
| R `rdrobust` 4.0.0 | 0.0763 | −5% |

第二个测试 pin 了 0.073，注释自己写着「paper 0.077」。现为 0.077545 对 R 的 0.077547。

### B.3 移植中踩到的坑（写给后续 WP）

1. `stdvars = FALSE` 是默认——不标准化，`BWp` 走原始尺度分支。
2. `masspoints = "adjust"` 是默认——`c_bw` 用唯一值计数 `M` 而非 `n`；
   任一侧并列 ≥20% 时触发 `bwcheck=10` 下界。
3. **stage 1 传 `scale = 0`，stage 2/3 传 `scaleregul`**。三级都传会让 h 偏低 11%。
4. **`bwrestrict` 夹紧作用在每一级中间带宽上**，不只是 `c_bw`。漏掉会让
   `msesum`/p=2/epanechnikov 偏 1.1%（其余 34 格正常）——是最难发现的一个。
5. **用户显式给 `h` 而不给 `b` 时，R 令 `b = h`**，不走选择器。我最初让 b 无条件
   走级联，导致 `sp.rdbwsensitivity` 的网格估计值恒定（因为 tau_bc 只由 b 决定）。
   这个回归是既有测试套抓到的，不是我的验收测试。
6. `tau_bc` **不是**在 b 窗口上跑 q 阶回归，而是
   `Q_q = R_p'W_h − h^(p+1)·L·e_{p+2}'·(invG_q R_q')W_b`，再 `beta_bc = invG_p Q_q' D`。

### B.4 E′ 的最后一个坑

稳健 SE 接近尾声时卡在 **conventional 和 robust 同时偏 4.2%** —— 共同因子。
原因是 nn 残差的 tie run：R 在**整侧**数据上算 `dups`/`dupsid`，再按窗口取子集
（`edups_l = dups_l[ind_l]`）；在窗口内重算会错数跨边界的并列值。改对后
两个 SE 同时降到 3.6e-12。

### B.5 已完成 / 未完成

**已完成**：WP-1（带宽级联）、WP-2（偏差修正 + 稳健方差）。
`sp.rdrobust` 的 sharp / 无协变量 / 无聚类 / `vce="nn"` 路径全量对齐。

**WP-2 剩余参数面：已量化并锁进 CI**（`test_rdrobust_params_parity.py`，
夹具 `_generate_rdrobust_params_R.R` / `rdrobust_params_R.json`）。

实测差距（相对偏差）：

| spec | h | conv | se_conv | robust | se_rob |
| --- | ---: | ---: | ---: | ---: | ---: |
| covs_p1 | 7.3e-03 | 9.5e-03 | 1.8e-03 | 9.7e-03 | 1.4e-03 |
| fuzzy_p1 | **1.2e-08** | **3.8e-14** | 1.1e-02 | 9.6e-03 | 2.5e-02 |
| cluster_p1 | 3.8e-02 | 2.3e-03 | 4.2e-02 | 3.2e-03 | 2.9e-02 |
| deriv1 | **6.4e-09** | **6.6e-14** | 1.5e-02 | 3.8e-02 | 7.6e-02 |

两个可直接读出的结论：

- **`fuzzy` 与 `deriv` 已经拿到正确带宽**（1e-8），`fuzzy` 连 conventional
  点估计也已精确（1e-14）——级联对它们已生效，缺的只是方差。
- **`covs` 与 `cluster` 没有**（h 差 7e-3 / 4e-2）——级联里没有协变量投影
  与聚类机制，带宽在形成任何估计之前就已经错了。

19 个 `xfail(strict)` 守住这些；另有一个**上界测试**（10%）防止未来退回到
sharp 路径曾经的 60% 量级——即使 strict 断言仍是 xfail 也会失败。

**未完成（按优先级）**：

1. **CCT 的协变量投影与聚类机制**：`_vbr` / `cct_bias_corrected` 需要 Z 列、
   `s` 向量与 gamma 投影（R 侧的 `dZ` 分支），以及 `rdrobust_vce` 的
   cluster 路径。这是 covs/cluster 带宽与全部 SE 的共同前提。
2. **`vce=` 参数不存在**：R 有 `hc0..hc3` / `cr*`，`sp.rdrobust` 一个都没有，
   R 脚本无法迁移。这是 API 缺口，不是数值问题。
2. **WP-3～WP-7**：密度检验、局部随机化、多断点、honest CI、其余 33 个函数分诊。
3. **evidence 分级**（1153 vs 147）——建议放独立 `_evidence.json`，避免
   `registry.py` 的并行冲突。
4. **文档**：quickstart card 仍显示旧的 0.073，需重新生成；
   `docs/guides/choosing_rd_estimator.md` 的带宽章节需按本次结论重写。
5. 5 个 `test_cov95_*` 已重新 pin，标注为**回归护栏而非正确性证据**
   （它们没有 R 对照；有对照的是它们的输入）。


---

## 附录 C · 我在 WP-2 里引入并修复的回归 —— `covs=` 静默失效

**更正**:上一轮我把这条当成「既有缺陷 F」汇报,那是错的。
经 `git show 76d06565` 逐版本比对确认,**它是我在 WP-2 里引入的回归**——
改动前 `covs` 工作正常（se 0.3765 → 0.0587,降 6.4 倍,与 R 行为一致）。

**成因**:WP-2 里我写的 CCT 替换只挡了 `fuzzy`,没挡 `covs`：

```python
if _tau_bc_cct is not None and fuzzy is None:      # 缺 and Z is None
    tau_conv, tau_bc, se_conv, se_robust = _tau_bc_cct
```

`cct_bias_corrected` 没有协变量机制,于是用**未调整**的 CCT 结果覆盖了
`_rd_estimate` 已经算好的协变量调整估计。表现是： 与不传 `covs` 的调用**逐位相同
（1e-12）**,在两个不同数据集上都是;`covs` 甚至没有进入 `model_info`。
用户传了协变量、不报错、拿到未调整的估计,且无从察觉。

判别性 DGP(`z` 系数 2.0,残差噪声 0.3):

| | est | se | h |
| --- | ---: | ---: | ---: |
| R 不调整 | 2.8901 | 0.2856 | 0.3131 |
| **sp 不调整** | **2.8901** | **0.2856** | **0.3131** ✓ |
| R 调整 | 3.0033 | **0.0444** | 0.2700 |
| sp 调整 | 2.8901 | 0.2856 | 0.3131 ✗ 与不调整相同 |

R 的 SE 降了 **6.4 倍**,StatsPAI 纹丝不动。

这也解释了参数面差距表里 covs 那两行的 ~1e-2 —— **从来不是带宽问题**。

**已修复**:替换条件加上 `and Z is None`。修复后 no-cov se=0.2856(精确匹配 R)、
covs se=0.0421(R: 0.0444),降 6.8 倍。两个 `xfail(strict)` 已转为常规回归测试。

### 为什么现有测试抓不到

`tests/test_rd_validation.py::test_covariate_adjustment_reduces_se` 断言的是
`r_cov.se < r_no_cov.se * 1.5`。两者完全相等时该断言**成立**。
容差方向写反了:它允许调整后 SE 更大 50%,却无法检测「调整根本没发生」。

### 我尝试过修,失败并回滚了

在 `_cct_bandwidth._vbr` 里实现了 Frisch-Waugh 式的协变量偏出(R 的 `dZ` 分支:
`gamma = (ZWZ)^-1 ZWY`,`s = [1, -gamma]`),并接进 `cct_bandwidth`。

在 senate 夹具上它「改善」了(h 从 7.3e-3 到 2.0e-3),**但那份数据的协变量几乎
不起作用**。换到上面这个判别性 DGP,它把 h 压到 **0.0915(R: 0.2700,窄 3 倍)**,
SE 从 0.044 恶化到 0.520。

**已回滚接线**(`_cct_bandwidth` 里的 Z 代码保留但不被调用),因为让 covs 留在
legacy 路径严格更好。这本身是个教训:我差点用一份退化夹具「验证」了一个错误实现
——正是本计划一路在批评的模式。

### 教训

两条,都指向同一件事——**验证数据必须具备判别力**：

1. 我用 senate 夹具「验证」了一个错误的 Z 投影(见 B.6),因为那份数据的协变量
   几乎不起作用。
2. 我引入的这个回归,在 senate 上只表现为 ~1e-2 的差距,看起来像「带宽还没接
   协变量」;换到 `z` 系数 2.0、噪声 0.3 的 DGP 上,立刻暴露为 6.4 倍的 SE 差异。

**现在这个判别性 DGP 已固化在 `test_rdrobust_params_parity.py` 里**,
两个常规测试守住:covs 必须改变估计、且必须让 SE 至少减半。

### 剩余

- 带宽的 Z / cluster 投影仍未接入(`h` 差 ~7e-3 / ~4e-2)——估计本身已调整,
  只有带宽选择没有。必须用判别性 DGP 验证后再动。
- `vce=` 参数缺失。
