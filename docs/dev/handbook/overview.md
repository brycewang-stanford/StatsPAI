# 项目概览：领域分组、guide 清单、Memory 指针

> 从 CLAUDE.md §1、§2、§8、§13、§14 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；需要按领域找模块、找对应 guide 时先读完本文件。新的细则直接写在这里。

做法有三条：

1. **一次 `import statspai as sp`** 覆盖 DiD / IV / RD / 合成控制 / DML / Meta-Learner / 贝叶斯因果 / 因果发现 / 结构计量 / 面板 / 空间 / 时序。
2. **Agent 原生**——所有函数返回结构化结果、带自描述 schema，人和 Agent 用同一入口。**v1.6 起进入 P1 阶段**：`sp.causal_question`（estimand-first DSL）、`sp.llm_dag_propose / validate / constrained`（LLM-DAG 闭环）、`sp.paper()`（自动论文）、`sp.causal_text`（文本因果 MVP）。
3. **数值对齐 Stata / R**——已有参考实现的方法，先对齐再扩展。

领域分组（对外 API 按下列七类组织）：

- **因果 / 处理效应**：`causal did rd iv synth dml metalearners tmle bcf bayes mcmc causal_impact policy_learning ope dtr multi_treatment qte principal_strat proximal mediation mendelian assimilation bridge`
- **面板 / 结构**：`panel fixest structural frontier multilevel gformula gmm msm longitudinal`
- **空间 / 时序**：`spatial timeseries bartik`
- **因果发现 / ML**：`causal_discovery dag neural_causal deepiv conformal_causal matrix_completion bunching causal_llm causal_rl causal_text fairness`
- **设计 / 抽样 / 推断**：`matching power mht survey bounds dose_response interference selection censoring imputation transport target_trial epi survival surrogate doe`
- **分解 / 诊断 / 回归**：`decomposition regression nonparametric diagnostics robustness postestimation inference smart`
- **基础设施**：`core utils compat fast datasets output plots workflow agent experimental question`

## 8. 文档

- MkDocs（[`mkdocs.yml`](mkdocs.yml)），源在 `docs/`，`mkdocs serve` 本地预览。
- 当前 guide：`synth` / `choosing_did_estimator` / `choosing_iv_estimator` / `choosing_rd_estimator` / `choosing_matching_estimator` / `callaway_santanna` / `cs_report` / `honest_did` / `repeated_cross_sections` / `robustness_workflow` / `bayesian_econometrics` / `migration-from-r` / `mixtape_ch09_did` / `stock_watson_4e` / `design_based_econometrics` / `ding_first_course` / `xu_lan_causal_econometrics` / `facure_causal_inference_in_python` / `schuler_vanderlaan_modern_causal_inference` / `forecasting_fpp` / `time_series_econometrics` / `design_of_experiments` / `wager_causal_inference` / `regression_and_other_stories` / `yuksel_aydede_causal_ml` / `conditional_inference_trees` / `figure_editor`。相关估计器改动要同步对应 guide。
- JOSS 论文 [`paper.md`](paper.md) + [`paper.bib`](paper.bib)——对外 API 或项目范围变更时同步。

## 13. 参考 Memory

`~/.claude/projects/-Users-brycewang-Documents-GitHub-StatsPAI/memory/`：

- `user_bryce.md` — 用户画像（计量经济学背景，期望精准技术语言）
- `project_statspai_vision.md` — P0–P3 路线图
- `feedback_sp_alias.md` — 始终 `import statspai as sp`
- `feedback_no_pr.md` — 直推 main
- `reference_pypi_publish.md` — 发布流程

外部指针：[GitHub](https://github.com/brycewang-stanford/StatsPAI) · [PyPI](https://pypi.org/project/StatsPAI/) · [CoPaper.AI](https://copaper.ai)。

---

## 14. 速查

```bash
pip install -e ".[dev]"                       # 开发安装
pytest                                        # 测试
black src tests && flake8 src tests && mypy src   # lint / format / type
mkdocs serve                                  # 文档预览
python -m build && twine check dist/*         # 打包
python benchmarks/run_all.py                  # 性能基准
python -c "import statspai as sp; print(len(sp.list_functions()))"   # registry 自检
python scripts/registry_stats.py                # canonical 数字（README/docs/stats.md 同步用）
python scripts/registry_stats.py --check        # CI 漂移检查（function/submodule 计数）
python scripts/registry_stats.py --table        # 重生 docs/stats.md 的按模块表
(cd rust/statspai_hdfe && maturin develop --release)   # Rust 后端
```

## 分领域笔记索引（原 §11，含条目列）

| 文件 | 领域 | 条目 |
| --- | --- | --- |
| [`rd.md`](docs/dev/modules/rd.md) | 回归断点（RD） | `rd/`；`rd/optimized.py`（`sp.rd_optimized`，Imbens-Wager 2019） |
| [`synth.md`](docs/dev/modules/synth.md) | 合成控制 | `synth/`；`synth/_gsynth_multi.py`（`sp.gsynth(treat=)`，1.39 起） |
| [`did.md`](docs/dev/modules/did.md) | 双重差分与事件研究 | `did/did_forest.py`；`callaway_santanna(notyet_cutoff=)`；`did/es_inference.py`；CS 的聚类变量；`did/_arp_sets.py::rm_confidence_set`；`sp.drdid` 的面板路径；`sp.did_multiplegt`；`did/few_treated.py`；dCDH 家族（`did/did_multiplegt.py` / `did_multiplegt_dyn.py` / `twowayfeweights.py`）；`sp.sun_abraham`；DiD 的日期型时间列（`did/_core.py::calendar_time_aware`） |
| [`decomposition.md`](docs/dev/modules/decomposition.md) | 分解 | `decomposition/` |
| [`panel_multilevel.md`](docs/dev/modules/panel_multilevel.md) | 面板、多层模型、前沿、动态面板 GMM | `multilevel/` / `frontier/` / GLMM；`gmm/_dynpanel/`；`panel/` 的随机效应方差分量只有一份实现；`mackinnon1994_pvalue`（`panel/unit_root.py`）在拟合范围之上返回 1 |
| [`bayes_mcmc.md`](docs/dev/modules/bayes_mcmc.md) | 贝叶斯（PyMC 与 mcmc/） | `bayes/`；`mcmc/`（1.39 起）；`mcmc/` 的第二批（1.39 起）；`mcmc/` 的回归工作流（2026-10 Gelman-Hill-Vehtari 一轮）；`mcmc/semiparametric.py`（`sp.semiparametric_bayes`，2026-10-28） |
| [`forest.md`](docs/dev/modules/forest.md) | 因果森林与 GRF 家族 | `forest/`；`forest/` 的 GRF 家族 |
| [`regression.md`](docs/dev/modules/regression.md) | 回归族（似然模型、GEE、GAM、选择、非参数树） | `regression/mprobit.py`；`nonparametric/conditional_tree.py`（`sp.ctree`，2026-10-12 起，来自 Varian 2014 的复现）；`selection/elastic_net.py`（`sp.glmnet`，1.39 起）；共线回归元（`core/_collinear.py`，2026-10 起）；`regression/gee.py`（`sp.gee`，1.39 起）；`regression/gam.py`（`sp.gam`，1.39 起）；教材驱动的小估计器；`sp.fmm`；`sp.shapley_r2`；`rlasso(absorb=)` |
| [`iv.md`](docs/dev/modules/iv.md) | 工具变量与 MTE | `regression/iv.py`；`iv/mte.py`；`iv/recentered.py`（`sp.recentered_iv`，Borusyak-Hull，2026-10-28）；`panel/_iv_sw.py`（SW 第一阶段）；`sp.iv_mediate` |
| [`formulas_covariates.md`](docs/dev/modules/formulas_covariates.md) | 公式写法、协变量与处理变量的输入约定 | 公式里的非标识符列名；公式的 R 写法（同一轮）；协变量列表里的分类变量（`core/_covariates.py`，2026-10-05 起）；二值处理的校验；公式里的 `I(x^2)` |
| [`treatment_effects.md`](docs/dev/modules/treatment_effects.md) | 处理效应（TMLE、IPW/AIPW、匹配、反驳检验） | `tmle/`（2026-10 Schuler-van der Laan 一轮）；`matching/full.py`（`sp.full_match` / `sp.match(method='full')`，1.39 起）；`robustness/refute.py`（`sp.refute`）；`matching/`；`sp.ipw(se_method='sandwich')`；`sp.aipw(trim=)`（1.39 起）；`matching/`；`matching/residual_balance.py`（`sp.residual_balance`） |
| [`dml.md`](docs/dev/modules/dml.md) | 双重机器学习 | `dml/`；`dml/did.py`（`sp.dml_did`，1.39 起） |
| [`causal_discovery_dag.md`](docs/dev/modules/causal_discovery_dag.md) | 因果发现与 DAG | `causal_discovery/mmhc.py`（`sp.mmpc` / `sp.mmhc`）；`causal_discovery/`；`dag/`（图这一层） |
| [`structural.md`](docs/dev/modules/structural.md) | 结构模型（BLP、路径分析、生产函数） | `structural/blp.py`（`sp.blp` 的随机系数部分，2026-10 按 BLP 1995 复现文章修订）；`structural/path_analysis.py`（`sp.path_analysis`）；`structural/production/op_lp_acf.py`（OP / LP / ACF）；`structural/blp.py`（`sp.blp` / `sp.blp_instruments`，2026-10-12 起含 Berry 1994）；`structural/ddc/`（`sp.ddc`，2026-10-28） |
| [`stata_translation.md`](docs/dev/modules/stata_translation.md) | Stata 翻译（sp.from_stata / sp.stata） | `agent/_translation/`（`sp.from_stata` / `sp.stata`）；`sp.stata` 的数据管理 / 描述统计 / 抽样 / 编程层（2026-10 Kohler-Kreuter 一轮新增）；`i.f#1.d` 的翻译 |
| [`timeseries.md`](docs/dev/modules/timeseries.md) | 时间序列与预测 | `timeseries/`；`timeseries/dlm.py`（1.39 起）；`timeseries/` 预测工具箱（2026-10，来自 Hyndman FPP 审查）；`timeseries/` 的 2026-10 Neusser 一轮 |
| [`measurement_error.md`](docs/dev/modules/measurement_error.md) | 测量误差（回归元与协变量） | `measurement_error/eivreg.py`（`sp.eivreg`）；`measurement_error/simex.py`（`sp.simex`）；`measurement_error/noisy_covariates.py`（`sp.noisy_covariates`，Agarwal-Singh），均 2026-10-28 起 |
| [`adverse_selection.md`](docs/dev/modules/adverse_selection.md) | 逆向选择（保险市场） | `adverse_selection/`（`sp.positive_correlation_test` / `sp.efc_welfare` / `sp.pooled_price_ratio` / `sp.risk_class_selection` / `sp.private_info_bound`，2026-10-28 起） |
| [`survival.md`](docs/dev/modules/survival.md) | 生存分析 | `survival/models.py` |
| [`experiments_interference.md`](docs/dev/modules/experiments_interference.md) | 试验设计、干扰、自适应试验、离策略评估 | `doe/`（试验设计，1.39 起，来自 Joseph 2025 教材）；`interference/network_exposure.py`（2026-10 Wager 教材第 12 章重建）；`ope/mdp.py`（1.39 起）；`experimental/adaptive.py`；`spatial/treatments.py`（`sp.spatial_treatment`，Pollmann，2026-10-28）；实验论文的报告工具（sharpened q、summary_index、suest、ri_test、krippendorff_alpha、LLM 编码） |
| [`agent_mcp.md`](docs/dev/modules/agent_mcp.md) | MCP 服务器与随包 skill | `agent/mcp_server.py`（`statspai-mcp`，2026-10-14 一轮）；随包 skill（`agent/_skill/`，`statspai skill install`） |
| [`evidence.md`](docs/dev/modules/evidence.md) | 证据映射（validation_scope） | `sp.validation_scope`（`validation_scope.py`）是按配置 × 输出的证据映射 |
| [`plots.md`](docs/dev/modules/plots.md) | 图形编辑器 | `plots/` 的图形编辑器（`sp.interactive` / `FigureEditor`，2026-10-16 重建） |
| [`datasets.md`](docs/dev/modules/datasets.md) | 数据集 | 随包与按需下载（`datasets/_fetch.py`）；mpdta 模拟副本与原始数据；测试的联网规则 |
