# StatsPAI (`sp`) 机器学习因果推断功能分析

## 一、ML 因果推断功能全景

`sp` 包在 ML 因果推断方面实现了 **8 大类方法**，全部原生实现（不依赖 EconML/CausalML 等外部包）：

| 类别 | 函数/类 | 核心方法 | 文献来源 |
|------|---------|----------|----------|
| **Double ML** | `dml()` / `DoubleML` | PLR（部分线性）+ IRM（交互回归），支持交叉拟合、多次重复中位数聚合 | Chernozhukov et al. (2018) |
| **Causal Forest** | `causal_forest()` / `CausalForest` | Honest random forest，双重去偏 + Bootstrap CI | Wager & Athey (2018) |
| **Meta-Learners** | `metalearner()` | S/T/X/R/DR-Learner 五种，支持任意 sklearn 兼容模型 | Kunzel et al. (2019), Kennedy (2023), Nie & Wager (2021) |
| **TMLE** | `tmle()` / `TMLE` | Targeted Maximum Likelihood，半参数高效估计 | van der Laan & Rose (2011) |
| **BCF** | `bcf()` / `BayesianCausalForest` | 贝叶斯因果森林，内置倾向得分调整 | Hahn, Murray & Carvalho (2020) |
| **DeepIV** | `deepiv()` / `DeepIV` | 深度神经网络 IV 估计（MDN + Response Network） | Hartford et al. (2017) |
| **Neural Causal** | `tarnet()`, `cfrnet()`, `dragonnet()` | TARNet / CFRNet / DragonNet 三大神经因果模型 | Shalit et al. (2017), Shi et al. (2019) |
| **Conformal CATE** | `conformal_cate()` / `ConformalCATE` | 无分布假设的 CATE 置信区间 | Lei & Candes (2021) |

此外还有：
- **Policy Learning**：`policy_tree()` + `policy_value()` — Athey & Wager (2021)
- **Causal Discovery**：`notears()` (Zheng et al. 2018) + `pc_algorithm()` (Spirtes et al. 2000)
- **AIPW**：`aipw()` — Robins, Rotnitzky & Zhao (1994)
- **Dose-Response**：`dose_response()` — Hirano & Imbens (2004) GPS 方法

---

## 二、各方法技术细节

### 2.1 Double/Debiased Machine Learning (`dml`)

**模型**：
- PLR: Y = θD + g(X) + ε, D = m(X) + v
- IRM: 二值处理的交互模型

**技术特点**：
- 交叉拟合（K-fold cross-fitting）消除正则化偏差
- 支持多次重复（n_rep）+ 中位数聚合，提高稳定性
- 默认使用 GradientBoosting，可替换为任意 sklearn 估计器
- 自动计算渐近正态推断的标准误和置信区间

### 2.2 Causal Forest (`causal_forest`)

**技术特点**：
- Honest estimation：分离建树样本和估计样本，避免过拟合
- 双重去偏：先估计 E[Y|X] 和 P(D=1|X)，再对残差建森林
- Bootstrap CI：基于自助法的个体预测置信区间
- 支持 formula 和 array 双接口
- 灵感来源于 EconML，但独立实现

### 2.3 Meta-Learners (`metalearner`)

五种学习器的数学框架：

| Learner | 核心思想 | CATE 估计公式 |
|---------|----------|---------------|
| S-Learner | 单模型 + 处理变量作特征 | τ(x) = μ(x,1) - μ(x,0) |
| T-Learner | 分组建模 | τ(x) = μ₁(x) - μ₀(x) |
| X-Learner | 两阶段插补 | 加权组合 imputed effects |
| R-Learner | Robinson 分解 + 损失最小化 | min E[(Ỹ - τ(X)D̃)²] |
| DR-Learner | 双稳健伪结果回归 | 基于 AIPW score 的回归 |

### 2.4 Neural Causal Models

| 模型 | 架构 | 损失函数 |
|------|------|----------|
| **TARNet** | 共享表示 φ(X) + 处理组特定输出头 | 事实结果 MSE |
| **CFRNet** | TARNet + 表示平衡惩罚 | MSE + α·MMD(φ_treated, φ_control) |
| **DragonNet** | 三头网络（outcome₀, outcome₁, propensity） | MSE + β·CE(e,D) + γ·targeted_reg |

### 2.5 TMLE

两阶段半参数估计：
1. 初始估计 Q(Y|A,W) 和 g(A|W)（可用 Super Learner）
2. 靶向步骤：用巧妙协变量 H(A,W) = A/g(W) - (1-A)/(1-g(W)) 更新
- 双稳健：结果模型或倾向得分模型任一正确即一致
- 达到半参数效率界

---

## 三、与现有包的详细对比

### 3.1 ML 因果方法覆盖度

| 方法 | **EconML** (Microsoft) | **CausalML** (Uber) | **DoubleML** (R/Python) | **DoWhy** (Microsoft) | **grf** (R) | **StatsPAI** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DML (PLR) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| DML (IRM) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| DML (PLIV) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Causal Forest | ✅ | ✅ (Uplift) | ❌ | ❌ | ✅ (原版) | ✅ |
| S-Learner | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| T-Learner | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| X-Learner | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| R-Learner | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| DR-Learner | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| TMLE | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| BCF | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| TARNet | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| CFRNet | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| DragonNet | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| DeepIV | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Conformal CATE | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Policy Tree | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| NOTEARS | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| PC Algorithm | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| AIPW | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Dose-Response (GPS) | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |

### 3.2 经典因果方法覆盖度（ML 包普遍缺失的领域）

| 方法 | EconML | CausalML | DoubleML | DoWhy | grf | **StatsPAI** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DID (2×2 + staggered) | ❌ | ❌ | ❌ | 部分 | ❌ | ✅ (14种变体) |
| RD (Sharp/Fuzzy) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (6种变体) |
| IV/2SLS | ✅ (部分) | ❌ | ✅ (PLIV) | ❌ | ❌ | ✅ (5种方法) |
| Synthetic Control | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (5种变体) |
| Matching | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ (PSM/CEM/Mahalanobis) |
| Entropy Balancing | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Lee/Manski Bounds | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Mediation Analysis | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Spillover Effects | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### 3.3 工程与生态系统对比

| 维度 | EconML | CausalML | DoubleML | DoWhy | grf (R) | **StatsPAI** |
|------|--------|----------|----------|-------|---------|-------------|
| 语言 | Python | Python | R/Python | Python | R | Python |
| 统一 API | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 经典计量覆盖 | ❌ | ❌ | ❌ | 部分 | ❌ | **✅ 完整** |
| Agent/LLM 原生 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| 智能推荐引擎 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| 出版级输出 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ (Word/Excel/LaTeX/HTML)** |
| HTE 诊断 | 有限 | 有限 | ❌ | ❌ | ✅ GATE/BLP | **✅ GATE/BLP + 可视化** |
| DAG 工具 | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| 结果对象标准化 | 中等 | 中等 | ✅ | ✅ | ✅ | **✅ (.summary/.plot/.cite)** |

---

## 四、StatsPAI 的核心差异化优势

### 4.1 一站式整合

现有 Python 生态中，ML 因果推断方法散布在 5+ 个独立包中：
- DML → `DoubleML` 或 `EconML`
- Meta-Learners → `EconML` 或 `CausalML`
- Causal Forest → `EconML` 或 `grf` (R)
- DAG/因果发现 → `DoWhy` + `pcalg` (R)
- 经典方法（DID/RD/IV） → `pyfixest`, `rdrobust` (R), `fixest` (R) 等

StatsPAI 用一个 `import sp` 覆盖全部 390+ 函数，API 风格统一。

### 4.2 独有方法（竞品中无 Python 实现）

- **TMLE**：半参数高效估计，Python 生态中几乎没有成熟实现
- **BCF**：贝叶斯因果森林，此前仅有 R 包 `bcf`
- **Neural Causal (TARNet/CFRNet/DragonNet)**：此前需要自行实现或使用零散代码
- **Conformal CATE**：无分布 CATE 推断，学术前沿方法

### 4.3 智能工作流引擎（竞品完全没有）

| 函数 | 功能 |
|------|------|
| `recommend()` | 根据数据 + 研究问题 → 自动推荐估计器 + 代码 |
| `compare_estimators()` | 多方法对比 + 符号/显著性一致性诊断 |
| `assumption_audit()` | 一键假设检验（SUTVA/Overlap/Unconfoundedness 等） |
| `sensitivity_dashboard()` | 多维敏感性分析面板 |
| `pub_ready()` | 期刊特定的发表就绪清单 |

### 4.4 Agent-Native 设计

- `list_functions()` — 列出全部 390+ 函数
- `describe_function()` — 函数描述
- `function_schema()` — 每个函数的 JSON Schema
- 所有结果对象自带结构化输出，天然适配 LLM 工作流

---

## 五、不足与改进建议

| 方面 | 现状 | 建议 |
|------|------|------|
| **PLIV（部分线性 IV）** | 未实现 | EconML/DoubleML 均支持，是 DML 框架的重要延伸，建议补充 |
| **连续处理 DML** | 仅 PLR 模型支持 | 可扩展 IRM 至连续处理场景 |
| **Auto-tuning** | 默认使用固定超参数（GBM 200 棵树） | 可引入 cross-validation 自动调参或 Super Learner 集成 |
| **GPU 加速** | Neural causal 依赖 PyTorch，其余 CPU only | 考虑 JAX 后端统一加速（已有 optional 依赖） |
| **公开 Benchmark** | 无标准数据集评测 | 建议在 IHDP / ACIC / LaLonde 等数据集上提供对比结果 |
| **IIVM（交互IV模型）** | 未实现 | DoubleML 支持，适用于二值 IV + 二值处理的场景 |
| **Orthogonal Forest** | 未实现 | EconML 的 OrthoForest 在高维场景有优势 |

---

## 六、模块文件结构

```
src/statspai/
├── dml/                    # Double ML (PLR + IRM)
├── causal/                  # Causal Forest (Honest + DML)
├── metalearners/            # S/T/X/R/DR-Learner + 诊断
├── tmle/                    # TMLE + Super Learner
├── bcf/                     # Bayesian Causal Forest
├── neural_causal/           # TARNet, CFRNet, DragonNet
├── deepiv/                  # Deep IV (MDN + Response Net)
├── conformal_causal/        # Conformal CATE
├── causal_discovery/        # NOTEARS + PC Algorithm
├── policy_learning/         # Policy Tree + Value
├── dose_response/           # GPS 连续处理效应
├── inference/               # AIPW, IPW, Bootstrap
├── did/                     # 14 种 DID 变体
├── rd/                      # 6 种 RD 变体
├── matching/                # PSM, CEM, Mahalanobis
├── synth/                   # SCM, SDID, GSynth
├── regression/              # IV (5种), OLS, GLM, ...
├── bounds/                  # Lee, Manski, Horowitz-Manski
├── mediation/               # 中介分析
├── interference/            # 溢出效应
├── dtr/                     # 动态处理机制
├── diagnostics/             # Sensemakr, E-value
├── smart/                   # 智能推荐引擎
└── core/                    # CausalResult 标准化结果
```

---

## 七、总结

StatsPAI 在 ML 因果推断领域实现了 Python 生态中**覆盖面最广的统一框架**：

1. **方法覆盖**：8 大类 ML 因果方法 + 完整经典计量方法，共 390+ 函数
2. **独有优势**：TMLE、BCF、Neural Causal、Conformal CATE 填补竞品空白
3. **工程优势**：统一 API、Agent-Native、出版级输出、智能工作流
4. **改进空间**：PLIV/IIVM、自动调参、GPU 加速、公开 Benchmark
