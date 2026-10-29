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
