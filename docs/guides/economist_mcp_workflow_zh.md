# 面向实证经济学的 MCP 工作流

这份指南面向常见的社科实证场景：研究者手里有本地 `.dta`、CSV、
Parquet 或 Arrow 数据，希望在 Claude Code、Codex、Cursor 等 MCP 客户端
里完成估计、诊断、稳健性检查和引用整理，而不是在聊天窗口里手动复制数据。

StatsPAI 的 MCP server 是 local-first 的：它读取你显式给出的
`data_path`，运行 StatsPAI 估计器，返回严格 JSON，并用 `result_id`
缓存拟合结果，方便后续工具复用同一个对象。

## 配置 MCP server

安装 StatsPAI 后，可以用 console script：

```json
{
  "mcpServers": {
    "statspai": {
      "command": "statspai-mcp",
      "args": []
    }
  }
}
```

也可以用 Python 模块入口：

```json
{
  "mcpServers": {
    "statspai": {
      "command": "python",
      "args": ["-m", "statspai.agent.mcp_server"]
    }
  }
}
```

## 数据交接

数据型工具都接受 `data_path`。支持 `.dta`、`.csv`、`.tsv`、`.parquet`、
`.feather`、`.xlsx`、`.json`、`.jsonl`，以及 `file://`、`https://`、
`http://`、`s3://`、`gs://` 等 URL。

大文件建议同时传：

```json
{
  "data_columns": ["y", "d", "id", "year", "x1", "x2"],
  "data_sample_n": 50000
}
```

`data_columns` 会尽量做列投影；`data_sample_n` 用固定随机种子做快速抽样。
默认数据大小上限由 `STATSPAI_MCP_MAX_DATA_BYTES` 控制。

当 MCP server 加载本地文件时，返回结果会带 `data_provenance`：源路径、
格式、请求列、抽样设置、文件大小、mtime 和 SHA-256。用
`statspai://result/<id>` 读取缓存结果时，也能在 provenance 区块看到同一份
来源摘要。远程 URL 会去掉 query token 后记录来源；StatsPAI 不会为了 provenance
再次下载远程数据计算 hash。

## 核心循环

推荐让 agent 使用 result handle：

```text
detect_design -> preflight/recommend -> fit(as_handle=true)
              -> audit_result -> sensitivity_from_result / plot_from_result
              -> bibtex
```

拟合工具返回 `result_id` 后，可继续调用：

| 工具 | 作用 |
| --- | --- |
| `audit_result` | 生成 reviewer-style 诊断清单 |
| `brief_result` | 结构化结果摘要 |
| `interpret_result` | 基于结果的解释，可在支持 MCP sampling 的客户端中增强 |
| `plot_from_result` | 返回内联 PNG 诊断图 |
| `sensitivity_from_result` | E-value / Oster / Cinelli-Hazlett 等敏感性分析 |
| `honest_did_from_result` | DID/event-study 的 Rambachan-Roth 敏感性分析 |
| `bibtex` | 从 `paper.bib` 读取真实引用，不让 agent 编造参考文献 |

## 常见实证 pipeline

StatsPAI 暴露了面向 MCP 的一键 pipeline：

| 工具 | 适用场景 |
| --- | --- |
| `pipeline_did` | DID、事件研究、Callaway-Sant'Anna 异质处理效应 |
| `pipeline_iv` | IV / 2SLS，并给出弱工具变量后续检查 |
| `pipeline_rd` | RD / fuzzy RD，并给出图形和密度检查 |

例子：

```json
{
  "name": "pipeline_did",
  "arguments": {
    "data_path": "/abs/cfps_panel.dta",
    "y": "lwage",
    "treat": "treated",
    "time": "year",
    "id": "pid",
    "cohort": "first_treat",
    "controls": ["age", "age2", "edu", "industry"],
    "as_handle": true
  }
}
```

## Stata 和 R 迁移

StatsPAI 提供单命令翻译工具：

| 来源 | MCP 工具 | 常见命令 |
| --- | --- | --- |
| Stata | `from_stata` | `regress`、`xtreg`、`reghdfe`、`ivreg2`、`ivreghdfe`、`csdid`、`did_imputation`、`synth`、`rdrobust`、`psmatch2` |
| R | `from_r` | `feols`、`felm`、`lm`、`glm`、`plm`、`matchit`、`att_gt`、`did` |

MCP prompts 中也有三个快捷入口：

| Prompt | 用途 |
| --- | --- |
| `stata_command_workflow` | 翻译一条 Stata 命令，运行对应 StatsPAI 工具，然后 audit |
| `r_command_workflow` | 翻译一条 R/fixest/did 表达式，运行并 audit |
| `cross_language_command_check` | 先比较 Stata 与 R 命令的 estimand、固定效应、聚类和协方差约定，再决定是否拟合 |

如果翻译器返回 `ok=false`，agent 应该报告错误和建议，而不是猜一个替代命令。
如果 Stata 与 R 片段隐含不同样本、控制变量、固定效应、处理时点或协方差约定，
应先报告 convention mismatch。

对于 `psmatch2`，`from_stata` 会把常见 nearest-neighbor、kernel、radius、
`common` 和 `ai()` 路径映射到 `sp.psmatch2`。会改变约定的选项，例如 Stata 的
`probit` 倾向得分或 ATE 取向请求，会以 notes 形式返回，而不是静默宣称完全对齐；
ATE 取向的 matching 应直接使用 `sp.match`。

对于 `ivreghdfe`，`from_stata` 会映射到与 R `feols(... | fe | endog ~ instr)`
一致的 StatsPAI/fixest 形状：公式里保留 IV block，`fe=[...]` 保留吸收固定效应。
这只是命令迁移合同，不是 live Stata 执行。

## 跨软件验证

StatsPAI 在仓库里保存了已提交的 Python/R/Stata parity artifacts。MCP 客户端可以读：

```text
statspai://parity/track-a-summary
```

这个资源返回精简 JSON：strictness-tier 计数、module id、Stata command 标签、
convention notes，以及按常见 StatsPAI tool name 建索引的 `tool_evidence`。它只是
总结已提交工件，不是 live Stata/R 执行。只有当你另外配置并实际调用 Stata MCP 或
R MCP server 时，才应声称完成了 live cross-software run。

## 外部数据 MCP

World Bank、OECD、FRED、IMF 或 OpenEcon 类数据 MCP 应负责检索指标和保存数据；
StatsPAI 负责分析收到的字节。推荐边界：

1. 用数据 MCP 查询并下载指标。
2. 保存成 CSV、Parquet、Arrow 或 `.dta`。
3. 把保存后的路径传给 StatsPAI 的 `data_path`。
4. 在 notebook、表注或附录中记录 provider、indicator id、query、retrieval date
   和转换代码。

StatsPAI 不会替你编造缺失的指标值、来源名或检索 provenance。
