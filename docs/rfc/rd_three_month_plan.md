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
