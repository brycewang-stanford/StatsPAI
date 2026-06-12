# Examples-coverage campaign

> 月度 campaign(2026-06-12 启动):把公开 registry 函数的 docstring
> `Examples` 段覆盖率从 35.9% 推向 ≥90%(函数型条目),配合 JOSS 审稿
> (openjournals/joss-reviews#10604,"example usage" 是 checklist 显式条目)。

## Baseline(2026-06-12,v1.17.0)

- 注册符号 1031 个;370 个(35.9%)docstring 带 `Examples` 段。
- 缺口 661 = **421 函数型**(本 campaign 主目标)+ 240 类型(Result 类
  等,deferred tier——多数经函数入口返回,单独示例价值低)。
- 测量工具:`python scripts/examples_coverage.py`(`--missing CAT` 出
  单类清单;`--check --max-missing N` 做 CI ratchet)。
- 完整清单:[worklist.md](worklist.md)(按类分组,checkbox 跟踪)。

## 红线(审稿期间)

1. **只加 docstring,不改任何函数签名/数值行为。**
2. 每条 Example 必须在 `.venv` 下实际运行通过后才能落盘;输出值只在
   固定种子下确定性成立时展示(CI 不收集 doctest,但示例错了被审稿人
   跑出来更糟)。
3. 一律 `import statspai as sp` + `sp.xxx`(CLAUDE.md §4)。
4. Examples 里不引文献;References 段维持现状(§10 零幻觉红线)。
5. 不在 Example 里调用网络 / LLM / R / Stata;只用 numpy/pandas 模拟
   数据或 `sp.datasets` 内置数据。
6. **并行会话协调**:本仓库可能有多个会话同时推进(见 Log 2026-06-12
   条目)。动手前先 `git log --oneline -5` + `git status`,避免重复
   补同一批函数;不要 stash 别人的未提交工作;CHANGELOG.md 有他人
   WIP 时不要并发编辑。

## 批次规划(优先级序)

| # | 范围 | 数量 | 状态 |
| --- | --- | --- | --- |
| 1 | causal 头部:DML 配套 + RD 家族 + matching + DiD/SDID | 28 | ✅ 2026-06-12 |
| 2 | causal 余量(qte/cate/policy/iv 配套) | ~30 | pending |
| 3 | causal 收尾 + bayes(整类 0%) | ~30 | pending |
| 4 | inference + panel + survey + mediation/gformula | ~30 | pending |
| 5 | output + postestimation + smart/agent | ~35 | pending |
| 6 | decomposition + spatial | ~40 | pending |
| 7 | mendelian + epi + dag + timeseries | ~45 | pending |
| 8 | 其余长尾(conformal/neural/interference/…) | ~60 | pending |
| 9 | class-style deferred tier 复盘(决定哪些值得加) | — | pending |
| 10 | CI ratchet 收紧至最终值 + docs 同步 | — | pending |

每批交付定义:示例运行验证 → `worklist.md` 勾选 → ratchet 预算下调 →
单独 commit(`docs(examples): batch N — <范围>`)。

## CI ratchet

`parity-guards.yml` 的 registry-drift job 挂
`scripts/examples_coverage.py --check --max-missing <budget>`。预算只
降不升;新注册函数若不带 Examples 会撞预算失败。当前预算:**633**。

## Log

- **2026-06-12** campaign 启动。扫描器 `scripts/examples_coverage.py`
  落地;baseline 370/1031(缺口 661 = 421 函数型 + 240 类型);
  worklist 生成。同日完成 DML "Spindler 防御":guide
  (`sp_dml_vs_doubleml.md`)新增 "Scope and known limitations" +
  "Companion tooling" 两节、parity 表更新到 1.17.0(4/4 doubleml-for-py
  0.11.3 重验通过,PLR/PLIV 机器精度)、`sp.dml` 双层 docstring Notes、
  registry 增加 multi-instrument failure mode、schema bundle 重生。
  这些编辑由并行会话随其 batch 一起提交(`60a0268`,该 commit 另含
  27 个高频函数的 Examples——与本 campaign 同方向的并行推进)。
- **2026-06-12 batch 1**(本会话):28 个头部因果函数 Examples 全部
  验证落地——DML 配套(dml_sensitivity/dml_diagnostics)+ IV
  (anderson_rubin_ci/iv_compare/kernel_iv/continuous_iv_late)+ RD
  (rdd/rdplotdensity/rdpower/rdsampsi/rdsensitivity/rdbalance/
  rdplacebo/rdsummary)+ matching/weighting(genmatch/optimal_match/
  cardinality_match/ps_balance/overlap_plot/trimming/
  stabilized_weights)+ DiD/SDID/QTE(gardner_did=did_2stage/cic/qdid/
  synthdid_estimate/synthdid_placebo/overlap_weighted_did)。
  缺口 661 → 633;ratchet 预算同步下调。已知边界:cic 与 qdid 的
  `n_boot=0` 会崩(示例用 n_boot=50 规避;修复属数值行为变更,留待
  审稿后走 ⚠️ 流程)。
