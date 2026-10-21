# 上游问题报告发送计划（2026-10-21）

给 Bryce 本人看。所有消息都还**没有发出**，由你亲自发送。正文在
`docs/dev/upstream-send/`，每个目标一个文件，只含消息本身：GitHub issue 的
文件第一行是标题、空一行后是正文；邮件和网页表单的文件只有正文，收件人和
主题写在下面。原始草稿（含给你的说明与仓库内证据路径）仍在
`docs/dev/2026-10-2*-upstream-*-draft.md`。

## 优先级与渠道

| 优先级 | 目标 | 渠道（已核实） | 文件 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | DRDID | GitHub issue，`pedrohcgs/DRDID`（CRAN BugReports） | `drdid.md` | |
| 2 | hdm | GitHub issue，`MartinSpindler/hdm`（仓库所有者即 CRAN 维护者 Martin Spindler） | `hdm.md` | |
| 3 | fect | **私人邮件**给 Yiqing Xu（CRAN Maintainer） | `fect-email.md` | 拟合作的 JSS 共同作者，语气友好，不发公开 issue |
| 4 | PanelMatch | GitHub issue，`insongkim/PanelMatch`（CRAN BugReports） | `panelmatch.md` | 新增第 4 条（ATC 倾向得分权重） |
| 5 | CMAverse | GitHub issue，`BS1125/CMAverse`（DESCRIPTION BugReports） | `cmaverse.md` | |
| 5 | GRAPPLE | GitHub issue，`jingshuw/GRAPPLE`（DESCRIPTION：Jingshu Wang，cre） | `grapple.md` | |
| 5 | xtevent | GitHub issue，`JMSLab/xtevent` | `xtevent.md` | README 已确认是作者的官方仓库，见下 |
| 6 | leebounds | 邮件给 Harald Tauchmann（SSC Support 行） | `leebounds-email.md` | |
| 暂缓 | StataCorp | 网页表单 <https://www.stata.com/support/tech-support/contact/> | `statacorp-{1..4}-*.md` | 先修好 Stata 授权与更新，见下 |

## GitHub issue 的命令

在仓库根目录执行。`tail -n +3` 去掉文件的标题行和空行，正文经标准输入交给
`--body-file -`；标题用单引号，里面的反引号与括号不会被 shell 解释（六个标题
都不含单引号）。

```bash
# 1 DRDID
tail -n +3 docs/dev/upstream-send/drdid.md | gh issue create --repo pedrohcgs/DRDID \
  --title '`drdid_rc` / `drdid_rc1`: sign of the pre-period control regression term in the influence function?' \
  --body-file -

# 2 hdm
tail -n +3 docs/dev/upstream-send/hdm.md | gh issue create --repo MartinSpindler/hdm \
  --title 'rlassoIV(select.X = FALSE) omits the second-stage intercept' \
  --body-file -

# 4 PanelMatch
tail -n +3 docs/dev/upstream-send/panelmatch.md | gh issue create --repo insongkim/PanelMatch \
  --title 'Four behaviours in PanelMatch 3.1.5 (forbid.treatment.reversal, placebo_test near the panel end, ps.match ties, ATC propensity weights)' \
  --body-file -

# 5 CMAverse
tail -n +3 docs/dev/upstream-send/cmaverse.md | gh issue create --repo BS1125/CMAverse \
  --title 'cmest delta-method SEs for ERcde and ERintref with a normal mediator and a log-linear outcome' \
  --body-file -

# 5 GRAPPLE
tail -n +3 docs/dev/upstream-send/grapple.md | gh issue create --repo jingshuw/GRAPPLE \
  --title 'Tukey loss: tau2.se looks about 3.7 times too small' \
  --body-file -

# 5 xtevent
tail -n +3 docs/dev/upstream-send/xtevent.md | gh issue create --repo JMSLab/xtevent \
  --title 'xtevent trend() standard errors when the trend starts after the window start' \
  --body-file -
```

想先看效果，可以把 `--body-file -` 换成 `--web`（在浏览器里预填后再手动提交），
或者先 `tail -n +3 <文件> | less` 读一遍。

## 邮件

**fect（优先级 3）**

- To：`yiqingxu@stanford.edu`
- Subject：`fect: the jackknife F test in diagtest`
- 正文：`docs/dev/upstream-send/fect-email.md`（开头 “Dear Yiqing,”，结尾署名 “Bryce”，提出可以改开 issue 或发 PR）

**leebounds（优先级 6）**

- To：`harald.tauchmann@fau.de`
- Subject：`leebounds with weights, one variance term appears unweighted`
- 正文：`docs/dev/upstream-send/leebounds-email.md`（署名 “Bryce Wang”，需要的话补上单位）

**xtevent 的备用渠道**（只在不用 GitHub issue 时）

- To：`simon.freyaldenhoven@phil.frb.org, chansen1@chicagobooth.edu, jorgepp@banxico.org.mx, jesse_shapiro_1@brown.edu`
- Subject：`xtevent trend() standard errors when the trend starts after the window start`
- 正文：`docs/dev/upstream-send/xtevent.md` 去掉第一行标题

**xtevent 仓库核实结果**：`gh api repos/JMSLab/xtevent/readme`（只读 README，没看代码）。README 写明
“Current SSC version: 3.1.0 (07jul2024)”，安装方式含 `github install JMSLab/xtevent`，
引用块列出的正是 SSC 上的四位作者（Freyaldenhoven、Hansen、Pérez Pérez、Shapiro），
各版本更新说明里修掉的 bug 都链接到这个仓库的 issue。可以认定是作者的官方开发仓库，
所以首选 GitHub issue。仓库没有归档、issue 开着，最近一次推送是 2025-12-18；搜索
“trend” 没有重复的 issue。

## StataCorp：暂缓

本机 Stata 18 是 2023-06-07 的修订版，`update query` 报 “bad serial number”，
不能更新。现在提交，四条里可能有已经在后续更新里修掉的。要做的事：

1. 修好授权（序列号 / 授权码），让 `update query` 能用，然后 `update all`。
2. 在更新后的 Stata 上把四个文件里的代码各跑一遍（`statacorp-1-stteffects.md`、
   `-2-xtintreg.md`、`-3-gsdesign.md`、`-4-xtregar.md`）。
3. 不再复现的条目直接删掉，不发；复现的把数字换成新版的输出，并把每条开头的
   “Stata/MP 18.0, macOS.” 改成带修订日期的版本（`about` 的输出）。
4. 一条一个表单，各自的主题：
   - 1：`stteffects ipwra, atet appears to average over the whole sample`
   - 2：`xtintreg standard errors under adaptive quadrature differ from the curvature of e(ll)`
   - 3：`gsdesign with nonbinding futility reports an expected sample size above the maximum`
   - 4：`xtregar, fe reports a standard error of _cons that is too small by a factor 1 - rho`

## PanelMatch 新增的第 4 条

来自 clean-room 重写（`ec626fdd9`）：`qoi = "atc"`（以及 `qoi = "ate"` 的 ATC 一半）在
`ps.weight` / `CBPS.weight` 下，R 给对照的权重正好是我们权重的倒数（集合内重新归一，
`test_atc_ps_weight_uses_the_odds_of_the_target_r_uses_the_reciprocal`，容差 1e-10），
平衡因此变差。2026-10-21 用 CRAN 装到临时库的 PanelMatch 3.1.5（R 4.5.2）实跑了
issue 里的代码（四段按顺序连跑，前三条的输出与草稿一致），第 4 条打印：

```
      none  ps.weight recomputed   inverted
 -0.822679  -1.282965  -1.282965  -0.353649
```

即 x1 在 t 期的标准化差：不精炼 -0.82，R 的权重 -1.28，权重取倒数后 -0.35（与
`sp.panel_match` 的 -0.353649 一致）。`CBPS.weight` 同样：-1.275957 对 -0.360662。
代码只用 PanelMatch 的公开 API 和 commit 钉住的公开 CSV，没有读 PanelMatch 源码。

## 发送前的检查

- [ ] 每份正文再通读一遍，确认语气、数字、链接都对；链接都钉在 commit 上，2026-10-21 全部能打开。
- [ ] fect：Yiqing 是拟邀的 JSS 共同作者。先想好这封邮件和合作邀请的先后，以及要不要在信里顺带提 JSS；正文现在只谈技术问题。
- [ ] StataCorp：授权修好、`update all`、在新版上重跑，不复现的删掉，之后再提交（见上）。
- [ ] PanelMatch 是 GPL-3：后续往来只看文档、输出和维护者的文字回复，不看源码。仓库里已有的 #127（“Bug in placebo_test”）正文引用了源码，没有读完，看标题与第 2 条不是一回事；#137（placebo.test 的数据要求，已关闭）标题看起来与第 2 条相关，只读维护者的文字回复即可。
- [ ] 用你自己的 GitHub 账号发；同一目标不要重复开 issue。
- [ ] 发出后把 issue 链接 / 邮件日期记到对应草稿顶部的说明里，方便跟进。
