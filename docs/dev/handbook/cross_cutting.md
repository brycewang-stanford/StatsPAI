# 跨领域约定

> 从 CLAUDE.md §11 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；新增回归入口、`predict(newdata)`、staggered 估计器、采样器或排序类统计量之前先读完本文件。新的细则直接写在这里。

跨领域的通用约定留在这里：

- **2026-10 月度审查留下的约定**（第三轮选项扫描 + 对本月约 210 个新函数的边界探测；清单与未决项在 CHANGELOG 该节）：
  - **权重的读法**：`weights_as=`（`core/_agent_summary.resolve_weights_as`）加在 13 个回归入口上。`'sampling'` 只在用户没另选方差时换成稳健方差，与手动指定逐位一致；每个入口只接受自己原生的那一种基于模型的读法。新增回归入口接受 `weights=` 时接上它，不要另起参数名。
  - **预测时没见过的因子水平必须拒绝**：检查在 `core/results.py::_prediction_design`（`postestimation/_design.reject_unseen_levels`），它读 `data_info['factor_levels']`。参照水平没有哑变量，所以只能靠拟合时记录（`fitted_factor_levels`）。已接 regress / logit / probit / glm / poisson / nbreg；**新增支持 `predict(newdata)` 的公式型估计器要在拟合时写这一项**，否则拼错的水平静默得到参照水平的预测。
  - **整数时间有间隔**：`sun_abraham` / `stacked_did` 的参照期是相对时间 −1，间隔面板上它不存在；两者经 `calendar_time_aware(numeric_gaps=True)` 按观测期的秩编号。新增以"相对时间 −1"为参照的估计器照此办理，并加"间隔编码等于连续编码"的测试。
  - **默认先验是否够 vague 的事后检查只有一份**：`mcmc/_core.note_default_prior`，所有带固定默认先验的采样器都要调用（1.39 时只有 `bayes_regress` 调了，IV 系数被压到真值的六分之一而无告警）。告警不得消耗随机数。
  - **并列**：凡是"按得分排序后取前若干"或"取 k 近邻"的统计量，并列的单位必须一起进入（`cate_gain_curve`、`factor_importance`），否则结果随行序变。`sp.match(ties='all')` 的**标准误**（2026-10-15 起）同臂邻居也全取并列者，即 Stata `psmatch2 , ai(1) ties`：Lalonde 1178.209472851439、NSW 640.4053566440037，任意行序都逐位一致（`test_psm_ties_se_parity.py`；Stata 要 `set type double`，默认 float 存条件方差时差 1e-8）。`ties='first'` 仍是不带 `ties` 的 psmatch2（随行序变，Stata 本身也如此）。此前记的 1186.871 是 J/(J+1) 换成 K/(K+1) 的错误候选。
  - **`.loc[y_df.index]` 是 bug**：索引标签重复时（`pd.concat` 不带 `ignore_index`）按标签取行会把行数翻倍。按位置取行；这一轮在 gam / gee / threshold / series / xthtaylor / abnormal_returns / dlm / mcmc 各入口都修过一遍。
  - **`sp.ljungbox(lags ≤ model_df)` 的 p 值缺失是约定**（R `Box.test(fitdf=)`、statsmodels 相同），只告警、不拒绝。
  - **探测代理的简报里要写明**：zsh 下未加引号的 `$(grep -rl ...)` 会作为**一个**参数传给 pytest，跑 0 条测试而不报错；用 `files=("${(@f)$(...)}")`，并核对报告里的测试条数。

- **输入卫生（2026-11-05，rOpenSci 统计软件标准审计的第一组缺口）**：
  - **`inf` 一律拒绝，NaN 照各估计器的缺失规则**。共享入口是 `_input_validation.require_finite`（`DataInsufficient`，诊断里有 `infinite_by_column`，提示文本是 `INFINITE_VALUES_HINT`）。已接入的层：`core/utils.create_design_matrices`（所有走它的公式估计器，按设计矩阵列名报，`np.log(0)` 这类派生的 `inf` 也拦）、`did/_core.drop_unusable_rows`（DiD 家族；队列列不在检查范围内，`g = inf` 仍是从未处理）、`rdrobust` 的 `_parse_data`、DML 基类 `_validate`、`fixest/_input_checks.check_fixest_input`、`clean_frame`、合成控制的结果面板。**不要**把 `inf` 当缺失值悄悄删掉，也不要放进求解器。新估计器读列时调用 `require_finite`。
  - **`sp.feols` / `fepois` / `feglm` 在分发到任何后端之前先查输入**（`fixest/_input_checks.py`）：零行、没有完整行、`inf`、无固定效应的裸列公式 `nobs <= k`，报文与 `sp.regress` 相同；pyfixest 的 `ZeroDivisionError` / `LinAlgError` / "All variables are collinear" 由 `translate_backend_failure` 转成带入口名的类型化异常。
  - **存储类型**：公式读作数值的 object 列若全是数（`Decimal`、Python int）按 float64 读并告警，`C(col)` 保持因子（`core/_formula_dtypes.coerce_formula_value_columns`，`create_design_matrices` 与 fixest 包装共用）；int64 超过 2**53 的列告警并给出 float64 后剩几个不同值。
  - **有序结果变量**：顺序未声明的文本标签（字符串 / object 列、无序 `Categorical` 的文本类别）一律拒绝，入口是 `regression/multinomial.require_ordered_outcome`（ologit / oprobit / gologit / meologit / `bayes_regress` 的有序模型共用）。`mlogit` 的基准类别按 `Categorical` 声明的顺序取首个水平。
  - **`missing=` 与共享入口检查（2026-11-06，审计缺口 4 / 16）**：25 个核心估计器（`regress` … `aipw`，清单见 `docs/guides/missing_values.md`）都挂 `_input_validation.checks_input` 装饰器，签名里写 `missing: MissingPolicy = "drop"`（`_input_validation.MissingPolicy`），函数体不看它。装饰器在拟合前依次拒绝：零行、被用到的列全缺、文本结果变量、`inf`、单一聚类、`missing="raise"` 下的缺失值（`DataInsufficient`，诊断 `missing_by_column`）。用到的列从调用参数里读（列名、列名列表、公式记号、`cluster=` / `vce="cluster v"` / `vcov={"CRV1": v}`）；带字符串默认值或 `Literal` 注解的参数、`_OPTION_PARAMS` 里的名字是选项，不扫。缺失有含义的列用 `nan_ok=` / `inf_ok=` 豁免（staggered 的 cohort 列，`heckman` 的结果变量）。装饰器放在 `accepts_aliases` / `accepts_vce` 之下、`markout_clusters` / `_expands_categorical` / `_calendar_time_aware` 之上。2026-11-07 起（RE2.2）`missing=` 也收按角色的字典 `{"response": ..., "predictors": ...}`：`response` 是装饰器 `outcome=` 参数指的列（公式左边），其余用到的列（含权重、聚类、单位与时间列、权重数组）都是 `predictors`，字典里没写的角色按 `"drop"`；拒绝时诊断多 `missing_by_role` / `missing_roles`，拟合后在 `result.diagnostics["missing_by_role"]` 记下按 `"drop"` 处理的角色有哪些缺失。字符串形式不变、什么都不记。registry 对所有被装饰的函数（`__statspai_checks_input__`）统一换成 `_MISSING_PARAM` 一张卡，schema 导出为字符串或按角色的字典（`_enum_or_role_mapping`）。新核心估计器照此接入，并把它加进 `tests/test_edge_case_contract.py` 的 `SPECS`。DiD 家族的完全共线协变量走 `did._core.independent_covariates`（省略后者并告警）。
  - **导出文件名**：写文件的 `to_excel` / `to_word` / `to_docx` / `to_latex` 都套 `_export_path.checks_export_path`：无后缀补上，后缀是别的格式就拒绝，缓冲区和 `None` 原样通过。新的导出方法也要套上它。
