# StatsPAI Skill for Claude Code

This folder is a **Claude Code Skill** that teaches Claude (or any
compatible agent harness) how to drive [StatsPAI](https://github.com/brycewang-stanford/StatsPAI)
end-to-end through the **canonical pipeline of an applied AER /
QJE / AEJ empirical paper** — pre-analysis plan, Table 1 summary
statistics, written-out estimating equation and identifying
assumption, identification graphics (event-study / first-stage F /
McCrary / SCM trajectory / love plot), the AER multi-regression
"Table 2" gauntlet (progressive controls / design horse-race /
multi-outcome / IV first-stage triplet), heterogeneity (subgroup
table + CATE figure), mechanisms, and a full robustness battery
(placebo / Oster / honest_did / E-value / Conley / two-way SE /
spec curve / sensitivity dashboard) — ending in a single editable
Word / Excel / LaTeX replication bundle.

## What this skill ships (paper-ready, in three formats)

Every numbered table in the AER pipeline emits parallel
**`.docx` + `.xlsx` + `.tex`** (Markdown / HTML on demand) so
co-authors can edit in Word, journal editors can review in Excel,
and the build system can assemble LaTeX — all from one source.

| Tier | When | API |
| --- | --- | --- |
| **1. Single multi-column table** (Stata `outreg2` / R `modelsummary` equivalent) | One Table 2 / Table 3 / Table A1 with progressive columns | `rt = sp.regtable(M1, M2, ..., template="aer", title=...)` then `rt.to_word("table2.docx") · rt.to_excel("table2.xlsx") · rt.to_latex()` |
| **2. Multi-panel paper format** | Tables 2 + 3 + A1 + A2 in one document | `sp.paper_tables(main=, heterogeneity=, robustness=, placebo=, template="aer").to_docx("paper_tables.docx")` |
| **3. Full session bundle** (Stata 15 `collect` equivalent) | Replication appendix mixing summary + balance + multiple regression tables + headings + prose | `sp.collect("Paper").add_summary(...).add_balance(...).add_regression(...)...save("paper.docx")` (auto-detects `.docx`/`.xlsx`/`.tex`/`.md`/`.html`) |

Journal templates auto-apply the right SE label, star levels, and
footer notes:

```python
sp.list_journal_templates()
# → ('aer', 'qje', 'econometrica', 'restat', 'jf', 'aeja', 'jpe', 'restud')

rt = sp.regtable(M1, M2, M3, template="qje")     # QJE styling
rt.to_word("table2_qje.docx")
```

Inline citations drop a coefficient straight into prose:

```python
sp.cite(M3, "training")           # → "1.239*** (0.153)"
```

## Default behavior the agent enforces

| Convention | Default action | When to override |
| --- | --- | --- |
| **Show every estimated parameter verbatim** (controls AND intercept) | `sp.regtable(*models, template="aer", ...)` with NEITHER `keep=` NOR `drop=` | Add `drop=["Intercept"]` to suppress the constant; `keep=[focal]` for intentionally focal-only tables (IV first-stage triplet, interaction-form heterogeneity) |
| **Route FE regressions through `sp.feols`** (pyfixest backend) | `sp.feols("y ~ x \| firm_id + year", df, vcov={"CRV1": "firm_id"})` | `sp.regress` for pure OLS only — it does **NOT** parse `\|` as FE |
| **Two-way cluster** | `sp.feols(..., vcov={"CRV1": "firm_id+year"})` | `sp.twoway_cluster(...)` is for `sp.regress` / `sp.ivreg` (statsmodels) results only |
| **Always export Word + Excel + LaTeX in parallel** | `rt.to_word(...) · rt.to_excel(...) · open(...).write(rt.to_latex())` | Skip `.tex` if you're not building a journal manuscript; always include `.docx` for human review |

## Install

```bash
# Option 1: copy
cp -r StatsPAI_full_data_analysis_skill ~/.claude/skills/StatsPAI_skill

# Option 2: symlink (auto-follow upstream updates)
ln -s "$(pwd)/StatsPAI_full_data_analysis_skill" ~/.claude/skills/StatsPAI_skill
```

Then install the Python package itself:

```bash
pip install statspai          # >= 1.6.6 (Word/Excel export stack)
```

## Activate

The skill auto-activates on natural-language triggers such as
*"run a DID analysis"*, *"AER empirical analysis"*, *"applied
microeconomics pipeline"*, *"instrumental variables regression"*,
*"event-study plot"*, *"first-stage F"*, *"Oster bound"*, *"export
regression table to Word"*, *"outreg2 in Python"*, *"sp.collect"*,
*"sp.feols"*, *"estimand-first DSL"*, etc. See the `triggers:`
block in `SKILL.md` for the full list.

## Scope

**In scope** — the canonical AER 8-section empirical pipeline:

```text
§-1 Pre-Analysis Plan      →  §0 Data construction & contract
§1  Table 1 (descriptives) →  §2 Empirical strategy (equation + identifying assumption)
§3  Identification graphics (event-study / first-stage / McCrary / SCM trajectory / love plot)
§4  Multi-regression main tables (progressive controls / design horse-race / multi-outcome /
                                  panel A-B / IV first-stage triplet) + coefplot
§5  Heterogeneity (subgroup table + CATE / dose-response figure)
§6  Mechanisms (mediation / decomposition)
§7  Robustness gauntlet (placebo / Oster / honest_did / E-value / 2-way & Conley SE /
                         spec curve / sensitivity dashboard) + robustness master table
§8  Replication package — Word / Excel / LaTeX / Markdown bundle via sp.collect()
                          + reproducibility stamp
```

**Out of scope** — data cleaning (use pandas first) and end-to-end
paper drafting (`sp.paper()`; the skill stops at diagnostics and
hands the `CausalResult` back to you).

## Files

- `SKILL.md` — frontmatter + full agent playbook
  - 8 paper sections (§-1 – §8) with end-to-end code
  - **8 multi-regression `regtable` patterns** (A. progressive controls · B. design horse race · C. multi-outcome · D. stacked panel A/B · E. IV first-stage triplet · F. `sp.causal()` orchestrator · G. subgroup heterogeneity · H. robustness master Table A1)
  - **3-tier export cookbook** (single table / paper-format multi-panel / full session bundle)
  - **17 standard AER figures** (raw trends, rollout heatmap, event-study, Bacon, CS-DID, RD plot, McCrary, love plot, SCM trajectory, coefplot, dose-response, CATE, robustness forest, spec curve, sensitivity dashboard)
  - **Method Catalog** (classical OLS / `feols` / IV / panel / DID / RD / matching / SCM / ML / neural / text / mediation / robustness)
  - **Common Mistakes table** (24 anti-patterns with corrections)
- `README.md` — this file

---

## StatsPAI Skill for Claude Code（中文版）

本文件夹是一份 **Claude Code Skill**，教 Claude（或任何兼容的 agent
运行时）端到端地驱动 [StatsPAI](https://github.com/brycewang-stanford/StatsPAI)
按照 **典型的应用经济学（AER / QJE / AEJ）实证论文套路** 完成一次完整
分析——从预分析计划、样本构造、Table 1 描述统计、写出回归方程与识别
假设，到事件研究图 / 一阶段 F / McCrary 密度 / SCM 轨迹 / love plot
等识别图、AER 多回归主表火力全开（progressive controls / 设计赛马
/ 多结果 / 面板 A-B / IV 三件套）、异质性（子样本表 + CATE 图）、
机制分解、完整 robustness gauntlet（placebo / Oster / honest_did /
E-value / Conley / 二维聚类 / spec curve / sensitivity dashboard），
最终交付一份 **同时是 Word / Excel / LaTeX 三种格式** 的可复现
replication bundle。

### 这份 skill 给你产出什么（论文级，三种格式同步）

AER 流程里每张编号的表格都同步产出 **`.docx` + `.xlsx` + `.tex`**
（Markdown / HTML 按需），合作者用 Word 编辑、期刊编辑用 Excel
审稿、构建系统拼 LaTeX——一份源、三种格式。

| 档位 | 适用场景 | 主 API |
| --- | --- | --- |
| **1. 单张多列表格**（Stata `outreg2` / R `modelsummary` 等价物） | 一张 Table 2 / Table 3 / Table A1，多列对比 | `rt = sp.regtable(M1, M2, ..., template="aer", title=...)` 然后 `rt.to_word(...) · rt.to_excel(...) · rt.to_latex()` |
| **2. 论文级多面板** | Table 2 + 3 + A1 + A2 整本一份文件 | `sp.paper_tables(main=, heterogeneity=, robustness=, placebo=, template="aer").to_docx(...)` |
| **3. 整套 session bundle**（Stata 15 `collect` 等价物） | replication 附录：描述统计 + balance + 多张回归表 + 标题 + 正文混排 | `sp.collect("Paper").add_summary(...).add_balance(...).add_regression(...)...save("paper.docx")`（按扩展名自动路由 `.docx`/`.xlsx`/`.tex`/`.md`/`.html`） |

刊物模板自动套对应的 SE label / 星号档 / 脚注：

```python
sp.list_journal_templates()
# → ('aer', 'qje', 'econometrica', 'restat', 'jf', 'aeja', 'jpe', 'restud')
```

正文里要现挂一个系数：

```python
sp.cite(M3, "training")           # → "1.239*** (0.153)"
```

### Agent 默认行为对齐

| 规则 | 默认 | 何时显式覆盖 |
| --- | --- | --- |
| **完整变量列表导出**（含 Intercept + 全部控制） | `sp.regtable(*models, template="aer", ...)`，不传 `keep=` 也不传 `drop=` | 想去掉常数 → `drop=["Intercept"]`；想 focal-only → `keep=[focal]`（IV 三件套 / 交互项才用） |
| **FE 回归走 `sp.feols`**（pyfixest 后端） | `sp.feols("y ~ x \| firm_id + year", df, vcov={"CRV1": "firm_id"})` | `sp.regress` 只用于纯 OLS——它**不解析** `\|` 为 FE 吸收 |
| **双向聚类** | `sp.feols(..., vcov={"CRV1": "firm_id+year"})` | `sp.twoway_cluster(...)` 只兼容 `sp.regress` / `sp.ivreg`（statsmodels 系） |
| **一次产 Word + Excel + LaTeX 三件套** | `rt.to_word(...) · rt.to_excel(...) · open(...).write(rt.to_latex())` | 不出 LaTeX 的话跳过 `.tex`；`.docx` 必出（人审用） |

### 安装

```bash
# 方式 1：复制
cp -r StatsPAI_full_data_analysis_skill ~/.claude/skills/StatsPAI_skill

# 方式 2：软链（自动跟随上游更新）
ln -s "$(pwd)/StatsPAI_full_data_analysis_skill" ~/.claude/skills/StatsPAI_skill
```

再装 Python 包本体：

```bash
pip install statspai          # >= 1.6.6（含 Word/Excel 导出栈）
```

### 激活

Skill 会被自然语言触发词自动激活，例如 *"run a DID analysis"*、
*"AER empirical analysis"*、*"applied microeconomics pipeline"*、
*"instrumental variables regression"*、*"event-study plot"*、
*"first-stage F"*、*"Oster bound"*、*"导出回归表到 Word"*、
*"outreg2 in Python"*、*"sp.collect"*、*"sp.feols"*、
*"estimand-first DSL"* 等。完整列表见 `SKILL.md` 开头 frontmatter
的 `triggers:` 字段。

### 适用范围

**覆盖**——AER 风格的 8 段式实证分析闭环：

```text
§-1 预分析计划          →  §0 样本构造与 data contract
§1  Table 1 描述统计     →  §2 实证策略（方程 + 识别假设）
§3  识别图（event-study / 一阶段 / McCrary / SCM 轨迹 / love plot）
§4  多回归主表（progressive controls / 设计赛马 / 多结果 / 面板 A-B
              / IV 一阶段+简化式+2SLS 三件套） + 系数图
§5  异质性（子样本表 + CATE / 剂量反应图）
§6  机制（中介 / 分解）
§7  Robustness gauntlet（placebo / Oster / honest_did / E-value /
                       二维 & Conley SE / spec curve / sensitivity 面板）
                       + Robustness master 主表
§8  Replication package — sp.collect() 一键产出 Word / Excel / LaTeX / Markdown
                          四种格式 + 复现戳
```

**不覆盖**——数据清洗（先用 pandas 处理）和端到端论文生成
（`sp.paper()`；skill 在诊断这一步停住，把 `CausalResult`
交还给用户）。

### 文件

- `SKILL.md` — frontmatter + 完整 agent 操作手册
  - 8 个论文 section（§-1 – §8）含端到端代码
  - **8 套 `regtable` 多回归 pattern**（A. 渐进控制 · B. 设计赛马 · C. 多结果 · D. 面板 A/B · E. IV 三件套 · F. `sp.causal()` 编排器 · G. 子样本异质性 · H. Robustness master Table A1）
  - **3 档导出 cookbook**（单表 / 论文级多面板 / 整套 session bundle）
  - **17 张标准 AER 图**（原始趋势、rollout 热图、event-study、Bacon、CS-DID、RD plot、McCrary、love plot、SCM 轨迹、coefplot、剂量反应、CATE、robustness 森林图、spec curve、sensitivity 面板）
  - **Method Catalog**（经典 OLS / `feols` / IV / 面板 / DID / RD / 匹配 / SCM / ML / 神经 / 文本 / 中介 / robustness）
  - **Common Mistakes 反模式表**（24 条带正确写法）
- `README.md` — 本文件
