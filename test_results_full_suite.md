# StatsPAI 全量测试结果报告

> **自动生成**: 2026-05-03 全量回归（含 ObjSense 修复后验证）
> **测试环境**: Python 3.13.5 · StatsPAI v1.12.2 · **scipy ObjSense 冲突已修复**
> **全仓覆盖**: 300 个测试文件，12 个批次，并行执行
> **最终结果**: ✅ **3032 passed, 0 failed, 23 skipped, 3 xfail**
> **全仓测试完成**: 主目录 12 批次 + 子目录 5 批次 = 17 批次全部通过
> **修复**: `tests/conftest.py` 预导入 `scipy.optimize` 稳定 PyO3 类型注册表

---

## 汇总仪表盘

| 批次 | 模块领域 | 通过 | 跳过 | 失败 | 告警 | 耗时 |
|:----:|---------|:----:|:----:|:----:|:----:|:----:|
| 1 | 核心 / 基础设施 | 223 | 0 | 0 | 14 | 21.08s |
| 2 | 双重差分 (DiD) | 281 | 0 | 0 | 607 | 9m 38s |
| 3 | 工具变量 (IV) | 99 | 0 | 0 | 0 | 7.99s |
| 4 | 断点回归 (RD) | 127 | 1 | 0 | 2 | 1m 14s |
| 5 | DML / 元学习器 | 136 | 1 | 0 | 1 | 11m 25s |
| 6 | 合成控制 (Synth) | 151 | 0 | 0 | 0 | 4m 14s |
| 7 | 匹配/面板/多层/GMM/分解 | 330 | 0 | 0 | 177 | 1m 30s |
| 8 | 贝叶斯 / BCF | 31 | 17 | 0 | 0 | 4m 13s |
| 9 | ML / 因果发现 / DAG / TMLE | 308 | 1 | 0 | 110 | 3m 43s |
| 10 | Agent / MCP / Workflow | 363 | 0 | 0 | 21 | 5m 17s |
| 11 | 回归表 / 输出 / 其他 | 359 | 1 | 0 | 25 | 1m 25s |
| 12 | 验证 / 对齐套件 | 75 | 0 | 0 | 18 | 1m 27s |
| 13 | IV 子目录 (贝叶斯IV/MTE/NPIV/弱IV) | 62 | 0 | 0 | 2 | 17.12s |
| 14 | 空间计量 (Spatial) | 67 | 1 | 0 | 0 | 77.45s |
| 15 | 参考对齐 (Reference Parity) | 109 | 1 | 3 xfail | 105 | 58.62s |
| 16 | 外部对齐 (External Parity) | 38 | 0 | 0 | 0 | 23.85s |
| 17 | 集成测试 + Monte Carlo | 12 | 0 (8 已选) | 0 | 0 | 15.55s |
| | **总计** | **2771** | **23** | **3 xfail** | **1082** | **~49m** |

---

## 批次详情

### 批次 1: 核心 / 基础设施 ✅

| 指标 | 数值 |
|------|------|
| 通过 | 223 |
| 失败 | 0 |
| 告警 | 14 |
| 耗时 | 21.08s |

**覆盖文件**: test_utils.py, test_help.py, test_registry.py, test_inference.py, test_ols.py, test_diagnostics.py, test_export.py, test_format.py, test_exceptions.py, test_citations.py, test_bibliography.py, test_collection.py, test_sumstats.py, test_session.py, test_brief.py, test_hausman.py, test_postestimation.py, test_quantile.py, test_ebalance.py, test_heckman.py, test_ri.py, test_rif.py, test_binscatter.py, test_modelsummary.py, test_sensemakr.py, test_spec_curve.py, test_subgroup.py, test_fast_*.py, test_detect_design.py, test_preflight.py, test_recommend.py, test_preregister.py, test_compat_sklearn.py, test_new_features.py, test_v06/09/10 migration suite, test_review_fixes, test_correctness_v150.py, test_dispatchers_v150.py, test_estimator_provenance (round 1-10), test_audit_*.py, test_cite_inline.py, test_article_aliases, test_escape_hatches.py, test_tidy_glance.py, test_gt_adapter.py, test_journal_presets.py, test_translation.py, test_unified_sensitivity.py, test_multi_se.py, test_multiway_and_subcluster.py, test_cluster_rct.py, test_frailty.py, test_exception_migrations.py, test_examples.py, test_lineage.py, test_repro_metadata.py, test_target_checklist.py, test_suggest_bibkey_backfills.py, test_auto_diagnostics.py, test_auto_estimators.py, test_causal_to_forest_rename.py, test_phase9to14.py, test_round3.py

---

### 批次 2: 双重差分 (DiD) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 281 |
| 失败 | 0 |
| 告警 | 607 (多为依赖库 deprecation warning) |
| 耗时 | 9m 38s |

**覆盖文件**: test_did.py, test_did_advanced.py, test_did_core_primitives.py, test_did_frontiers.py, test_did_multiplegt_dyn.py, test_did_multiplegt_joint.py, test_did_numerical_fixtures.py, test_did_summary.py, test_did_timevarying_covariates.py, test_cs_rcs.py, test_cs_report.py, test_cs_report_smoke.py, test_honest_did_aggte.py, test_honest_did_sdid.py, test_continuous_did_cgs.py, test_continuous_did_heuristics.py, test_harvest_did.py, test_lp_did.py, test_overlap_did.py, test_aggte.py, test_bjs_joint.py, test_sp_did_aggregation.py, test_gardner_2s.py

---

### 批次 3: 工具变量 (IV) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 99 |
| 失败 | 0 |
| 耗时 | 7.99s |

**覆盖文件**: test_iv.py, test_iv_dispatcher.py, test_iv_frontiers.py, test_kernel_iv.py, test_dist_iv_frontiers.py, test_weakiv_tobit.py, test_continuous_iv_late.py, test_front_door.py, test_front_door_integrate_by.py, test_bridge.py, test_bridge_full.py, test_mediation.py, test_mediate_interventional.py, test_mediation_sensitivity.py, test_principal_strat.py, test_proximal.py, test_proximal_frontiers.py

---

### 批次 4: 断点回归 (RD) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 127 |
| 跳过 | 1 |
| 取消选择 | 1 |
| 失败 | 0 |
| 告警 | 2 |
| 耗时 | 1m 14s |

**覆盖文件**: test_rd.py, test_rd_aliases.py, test_rd_dispatcher.py, test_rd_frontiers.py, test_rd_new_modules.py, test_rd_validation.py, test_rddensity_io.py, test_rdpower.py, test_bunching_unified.py, test_icp.py, test_ddd_heterogeneous.py

---

### 批次 5: DML / 元学习器 ✅

| 指标 | 数值 |
|------|------|
| 通过 | 136 |
| 跳过 | 1 |
| 失败 | 0 |
| 告警 | 1 |
| 耗时 | 11m 25s |

**覆盖文件**: test_dml.py, test_dml_iivm.py, test_dml_model_averaging.py, test_dml_panel.py, test_dml_split.py, test_metalearners.py, test_metalearner_frontiers.py, test_auto_cate.py, test_auto_cate_tuned.py, test_causal_forest_grf.py, test_forest_inference.py

---

### 批次 6: 合成控制 (Synth) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 151 |
| 失败 | 0 |
| 耗时 | 4m 14s |

**覆盖文件**: test_synth.py, test_synth_advanced.py, test_synth_extras.py, test_synth_new_methods.py, test_synth_survival.py, test_sequential_sdid.py

---

### 批次 7: 匹配 / 面板 / 多层 / GMM / 分解 ✅

| 指标 | 数值 |
|------|------|
| 通过 | 330 |
| 取消选择 | 1 |
| 失败 | 0 |
| 告警 | 177 |
| 耗时 | 1m 30s |

**覆盖文件**: test_matching.py, test_matching_optimal.py, test_match_dispatcher.py, test_panel.py, test_panel_dispatcher.py, test_frontier.py, test_multilevel.py, test_gmm.py, test_arima.py, test_garch.py, test_local_projections.py, test_econ_trinity.py, test_decomposition_tier_c.py, test_fixest.py, test_bartik.py, test_shift_share_political.py, test_epi.py, test_epi_diagnostic.py, test_evidence_synthesis.py, test_survey.py, test_survey_calibration.py, test_longitudinal.py, test_mr_diagnostics.py, test_mr_extensions.py, test_mr_extras.py, test_mr_frontier.py, test_gformula_ice.py, test_g_computation.py, test_msm.py, test_prod_fn.py, test_diag_themes.py, test_overlap_and_cbps.py, test_robustness_battery.py, test_robustness_report.py, test_diagnose_batteries_sprint_b.py, test_diagnose_result_closed_loop.py, test_smart_tools_sprint_b*.py, test_smart_workflow.py, test_workflow_sprint_b.py, test_predict_oos.py, test_paper*.py, test_replication_pack.py, test_hdfe_native.py, test_numba_kernels.py

---

### 批次 8: 贝叶斯 / BCF ✅

| 指标 | 数值 |
|------|------|
| 通过 | 31 |
| 跳过 | 17 (多为 pymc optional dependency) |
| 失败 | 0 |
| 耗时 | 4m 13s |

**覆盖文件**: test_bayes_advi.py, test_bayes_did.py, test_bayes_did_cohort.py, test_bayes_dml.py, test_bayes_fuzzy_rd.py, test_bayes_hdi_compat.py, test_bayes_hte_iv.py, test_bayes_iv.py, test_bayes_iv_per_instrument.py, test_bayes_mte.py, test_bayes_mte_bivariate_normal.py, test_bayes_mte_hv_latent.py, test_bayes_mte_multi_iv.py, test_bayes_mte_policy.py, test_bayes_mte_selection.py, test_bayes_mte_tidy.py, test_bayes_mte_uncertainty.py, test_bayes_rd.py, test_bvar.py, test_bcf_longitudinal.py, test_bcf_ordinal.py, test_causal_impact.py

> 注意: 17 个跳过是因为 pymc 为 optional dependency，CI 中未安装。非问题。

---

### 批次 9: ML / 因果发现 / DAG / TMLE ✅

| 指标 | 数值 |
|------|------|
| 通过 | 308 |
| 跳过 | 1 |
| 失败 | 0 |
| 告警 | 110 |
| 耗时 | 3m 43s |

**覆盖文件**: test_causal_discovery.py, test_causal_discovery_ts.py, test_dag_recommend_and_tte_report.py, test_dag_scm.py, test_neural_causal.py, test_deepiv.py, test_conformal_bcf_bunching_mc.py, test_conformal_extended.py, test_conformal_frontiers.py, test_tmle.py, test_hal_tmle.py, test_policy_learning.py, test_ope_cevae.py, test_ope_extensions.py, test_fairness.py, test_causal_llm.py, test_causal_rl.py, test_causal_rl_core.py, test_causal_text.py, test_causal_mas.py, test_causal_kalman.py, test_lingam.py, test_llm_dag_loop.py, test_llm_evaluator.py, test_llm_resolver.py

---

### 批次 10: Agent / MCP / Workflow ✅

| 指标 | 数值 |
|------|------|
| 通过 | 363 |
| 失败 | 0 |
| 告警 | 21 |
| 耗时 | 5m 17s |

**覆盖文件**: test_agent.py, test_agent_blocks_drift.py, test_agent_detail_levels.py, test_agent_docs.py, test_agent_result_methods.py, test_agent_schema.py, test_mcp_enrichment.py, test_mcp_error_envelope.py, test_mcp_image_content.py, test_mcp_pipelines.py, test_mcp_prompts_expanded.py, test_mcp_protocol.py, test_mcp_result_handle.py, test_mcp_runner.py, test_mcp_sampling.py, test_causal_workflow.py, test_question_dsl.py

---

### 批次 11: 回归表 / 输出 / 其他 ✅

| 指标 | 数值 |
|------|------|
| 通过 | 359 |
| 跳过 | 1 |
| 失败 | 0 |
| 告警 | 25 |
| 耗时 | 1m 25s |

**覆盖文件**: test_regtable_alpha.py, test_regtable_fmt_auto.py, test_regtable_publication_extensions.py, test_regtable_quarto.py, test_regtable_round2_extensions.py, test_regtable_round3_extensions.py, test_regtable_round4_extensions.py, test_regtable_snapshots.py, test_aer_word_style.py, test_export.py, test_tidy_glance.py, test_gt_adapter.py, test_journal_presets.py, test_translation.py, test_check_identification.py, test_sensitivity_frontier.py, test_target_trial.py, test_transport.py, test_surrogate.py, test_interference_extensions.py, test_auto_estimators.py, test_unified_sensitivity.py

---

### 批次 12: 验证 / 对齐套件 ✅

| 指标 | 数值 |
|------|------|
| 通过 | 75 |
| 失败 | 0 |
| 告警 | 18 |
| 耗时 | 1m 27s |

**覆盖文件**: test_validation_vs_stata_r.py, test_mixtape_ch09_guide.py, test_correctness_v150.py, test_dispatchers_v150.py, test_causal_to_forest_rename.py

---

## 第二批：子目录测试（第二轮）

---

### 批次 13: IV 子目录 ✅

| 指标 | 数值 |
|------|------|
| 通过 | 62 |
| 失败 | 0 |
| 告警 | 2 |
| 耗时 | 17.12s |

**覆盖文件**: tests/iv/test_bayesian_iv.py, test_ivmte_bounds.py, test_jive_variants.py, test_mte.py, test_npiv.py, test_plausibly_exogenous.py, test_plots.py, test_post_lasso.py, test_unified_fit.py, test_weak_identification.py, test_weak_iv_ci.py

---

### 批次 14: 空间计量 (Spatial) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 67 |
| 跳过 | 1 |
| 失败 | 0 |
| 耗时 | 77.45s |

**覆盖文件**: tests/spatial/ 全部 19 个测试文件（权重、ESDA、GWR、ML/SLX/SAC 模型、面板、诊断）

---

### 批次 15: 参考对齐 (Reference Parity) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 109 |
| 跳过 | 1 |
| 预期失败 (xfail) | 3 |
| 失败 | 0 |
| 告警 | 105 |
| 耗时 | 58.62s |

**覆盖文件**: tests/reference_parity/ 全部 17 个文件（DiD/RD/DML/IV/Synth/Matching/MR/HDFE/BCF 与 R/Stata 数值对齐）

---

### 批次 16: 外部对齐 (External Parity) ✅

| 指标 | 数值 |
|------|------|
| 通过 | 38 |
| 失败 | 0 |
| 耗时 | 23.85s |

**覆盖文件**: tests/external_parity/ 全部 3 个文件（CausalML book、Honest DiD 论文、已发表 replication）

---


---

### 批次 17: 集成测试 + Monte Carlo ✅

| 指标 | 数值 |
|------|------|
| 通过 | 12 |
| 取消选择 (slow) | 8 |
| 失败 | 0 |
| 耗时 | 15.55s |

> **StatsPAI v1.12.2 — 全量回归测试通过 ✅ | ObjSense 冲突已修复**
>
> - **全仓总计：3032 passed, 0 failed**
> - 第一批 12 批次：**2483 passed**, 21 skipped, 2 deselected
> - 第二批 5 批次（子目录）：**288 passed**, 2 skipped, 8 deselected, 3 xfail
> - 第三批（隔离运行）：**261 passed**, 7 skipped (**fast/** 149 + **bridge/smart/robustness** 112)
> - 3 个预期失败 (xfail) 为 R/Stata 已知数值偏差
> - ~2000 个告警几乎全部来自第三方依赖库的 DeprecationWarning，非 StatsPAI 代码问题
> - 全量标签覆盖率 **12.9%**，隔离覆盖率达 **18%**
> - **ObjSense 冲突已修复**：`tests/conftest.py` 预导入 `scipy.optimize` 稳定 PyO3 类型注册表；fast/bridge/smart/robustness 现可在同一进程中正常收集运行
> - 总执行时间约 **50 分钟**

### 覆盖率深度分析

#### 47 个零覆盖率文件的三层分类

**A 层 — 有测试但被 ObSense 阻塞（~18 文件，已修复）**
| 模块 | 文件数 | 测试状态 | 说明 |
|------|:------:|:--------:|------|
| `fast/` | 11 | 149 ✅ 7 ⏭️ | PyO3 冲突导致 pytest collect 失败，隔离运行全过 |
| `bridge/` | 6 | 10 ✅ | 同上 |
| `smart/benchmark`, `verify` | 2 | 51 ✅ | 通过 test_smart_tools* 覆盖 |
| `workflow/_robustness` | 1 | 23 ✅ | 通过 test_robustness_* 覆盖 |
| `compat/sklearn` | 1 | 已跳过 | sklearn import 触发 ObjSense |
| **小计** | **~18** | **261 ✅** | **全部通过** |

**B 层 — 有间接覆盖但工具无法追踪（~2 文件）**
| 文件 | 说明 |
|------|------|
| `agent/pipeline_tools.py` | 通过 test_agent.py / test_mcp_pipelines.py 间接覆盖 |
| `agent/workflow_tools.py` | 通过 test_mcp*.py 间接覆盖 |

**C 层 — 真正无专门测试（~21 文件，已覆盖 6 个）**
| 文件 | 行数 | 优先级 | 备注 |
|------|:----:|:------:|------|
| `agent/_enrichment.py` | 84 | 低 | 内部 MCP enrichment 工具 |
| `agent/_result_cache.py` | 75 | 低 | 内部缓存 |
| `agent/_translation/*` (4 文件) | 834 | 低 | R/Stata 翻译器，边缘功能 |
| `agent/auto_dispatch.py` | 55 | 低 | 自动分发 |
| `output/_excel_style.py` | 125 | 低 | Excel 导出格式 |
| `panel/hdfe_rust.py` | 19 | 低 | Rust HDFE 包装器 |
| `panel/panel_diagnostics.py` | 173 | 低 | 面板诊断 |
| `plots/_jupyter_editor.py` | 652 | 低 | Jupyter 编辑器 UI |
| `plots/_script_editor.py` | 106 | 低 | 脚本编辑器 UI |
| `spatial/models/_base.py` | 12 | 低 | 基类 |
| `target_trial/ccw_internal.py` | 41 | 低 | 内部工具 |
| `multilevel/mixed.py` | 2 | 低 | 占位符 |
| **小计** | **~2178** | | **已覆盖 622 行（6 个文件）** |

#### 新加测试（2026-05-03）

| 模块 | 文件 | 测试数 | 覆盖率 |
|------|------|:------:|:------:|
| `output/_aer_style.py` | `test_aer_style.py` | 20 | 98% |
| `core/next_steps.py` | `test_next_steps.py` | 41 (+1 fix) | 95%+ |
| `agent/auto_tools.py` | `test_auto_tools.py` | 29 | — |
| `cli.py` | `test_cli.py` | 31 | — |

#### 根本问题：scipy ≥ 1.14 的 PyO3 冲突 ✅ 已修复

scipy 1.16.1 的 `_highspy._core` 是 PyO3 模块，其 `generic_type: type "ObjSense"` 在 C 级类型注册表中注册。如果该模块从 `sys.modules` 卸载后重新导入，PyO3 拒绝重复注册：

```
ImportError: generic_type: type "ObjSense" is already registered!
```

**根因**：PyO3 的类型注册表是 C 级全局状态，即使 `sys.modules` 被清理也不会释放。`coverage` 的模块追踪或 pytest 大规模收集时的 import 顺序可能触发模块卸载→重载路径。

**修复方案**：在 `tests/conftest.py` 中预导入 `scipy.optimize`（需用其子模块 `_highspy._core`），确保 PyO3 类型在 pytest 收集任何测试文件之前稳定注册，且 `sys.modules` 持有强引用防止卸载。

**影响范围**（已解除）：11 个 `test_fast_*.py` + bridge + robustness + smart 共 ~20 个测试文件在集体收集时曾崩溃。

**修复验证**：`pytest tests/test_fast_*.py tests/test_robustness_*.py tests/test_bridge_*.py tests/test_smart_*.py ...` → **233 passed, 0 failed**；全量 batch：**459 passed, 0 failed**。

#### 覆盖率 ROI 排序（可写测试）

| 优先级 | 模块 | 行数 | 难度 | 影响 |
|:------:|------|:----:|:----:|:----:|
| — | `output/_aer_style.py` | 91 | 低 | ✅ 20 个测试 [覆盖] |
| — | `agent/auto_tools.py` | 139 | 中 | ✅ 29 个测试 [覆盖] |
| — | `core/next_steps.py` | 162 | 低 | ✅ 41 个测试 [覆盖] |
| — | `agent/_runner.py` | 77 | 中 | ✅ 11 个测试 [已有] |
| — | `agent/_sampling.py` | 84 | 中 | ✅ 10 个测试 [已有] |
| — | `cli.py` | 69 | 低 | ✅ 31 个测试 [覆盖] |
| 1 | 其余 ~20 个文件 | ~2100 | 低-高 | 边缘功能（待补） |
