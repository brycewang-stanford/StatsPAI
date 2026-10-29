# Git 协作细则

> 从 CLAUDE.md §9 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；推送、开 worktree、或从远程 runtime 回传结果之前先读完本文件。新的细则直接写在这里。

## 9. Git 协作

- **🚨 提交闸门（本节最高优先级，压过下面所有条款）**：**2026-09-28 起，用户已给出常设授权**：agent 判断合适时可以直接 `git commit` + `git push origin HEAD:main`，不必每次再问。"合适"指同时满足：(1) 相关测试与 pre-commit / pre-push 闸门全绿（不得 `--no-verify`）；(2) 只暂存自己这条线的改动，逐文件 add，绝不 `-A` / `-a`（§9.2）；(3) 改动完整、自洽，不是半成品；(4) 触及 JSS 冻结产物时已在 `docs/dev/jss_review_changes.md` 记录。拿不准就先问。**以下仍须用户在当前会话明确授权**：打 tag、发 PyPI / TestPyPI、GitHub Release、`--force` / 改写已推送历史、删除远程分支。每段工作结束时照常用中文总结，写明推送了哪些 commit。本地 `PreToolUse` hook（[`.claude/hooks/block-git-commit-push.py`](.claude/hooks/block-git-commit-push.py)）仍会拦截，判断合适后在命令前置 `STATSPAI_ALLOW_GIT=1` 放行，这是一道有意识的确认，不是绕过。
- **获得授权之后**才适用以下条款：默认分支 `main`，**直推 main**、默认不开 PR，除非明确要求（见 `memory/feedback_no_pr.md`）。
- Commit 风格：`feat:` / `fix(<area>):` / `docs(<area>):` / `chore:`，摘要 ≤ 72 字符。
- **禁止**：`--no-verify` / `--no-gpg-sign` / `--force`（除非明确授权）；对已推送 commit `--amend`。出错用 `git revert`。
- **push 前必须让 `python3` 指向本仓库 venv**，否则 pre-push 必然红。`.pre-commit-config.yaml` 里那批闸门（`flake8-count` / `registry-drift` / `schema-drift` / `error-taxonomy` / `orchestration-assertions` / `examples-coverage` / `parity-traces` / `registry-params` / `cold-import budget`）都是 `language: system` + 裸 `python3`，会按 PATH 解析；在没激活 venv 的 shell 里解析到系统 Python（如 Homebrew 3.14），直接 `ModuleNotFoundError: No module named numpy`，看起来像代码坏了，其实是解释器不对。

  ```bash
  source .venv/bin/activate && git push origin HEAD:main
  # 或（worktree 里不想污染环境时）
  PATH="/path/to/StatsPAI/.venv/bin:$PATH" git push origin HEAD:main
  ```

  `flake8-count`（2026-10-03 起）就是 CI 的 `python scripts/quality_gate.py flake8`：`src/statspai` 的违规总数不得超过 `DEFAULT_FLAKE8_MAX`。超长的字符串拆成相邻字面量，black 不会替你拆。这道闸门以前只在 CI 跑，结果 main 一周涨了 191 条没人察觉，而它在 mypy 和 pytest 之前失败，CI 从 09-28 到 10-03 一个测试都没跑。

  **不要**因此改 hook 的 `entry`：CI 里 `python3` 本来就是对的，把它钉死到 `.venv/` 反而会让 CI 红。这是本地环境问题，不是配置问题。**更不要**用 `--no-verify` 绕过——这六道闸门是真在挡东西。

### 9.2 并发：多窗口同时开工必须各自占一个 worktree

**开工前先看这条。** 如果另一个 Claude 窗口（或同事）正在本仓库作业，**不要**两边都在主工作树的 `main` 上改。

实测代价（2026-08-01，两条线并行一晚）：多个文件进入永久争用。每次提交要手工做「备份共享文件 → 还原到 HEAD → 重生成派生产物 → 提交 → 还原」五步，做了四轮；推送闸门按*已提交*状态检查而工作区混着两边改动，两个视角每次都打架。更糟的是有一次 `-A` 式全量提交把另一条线未提交的工作整个扫了进去，代码上了 main 却挂在毫不相关的 commit message 下。

**争用点清单**（2026-09-24 更新；原先只列了八个，实测不止）：

| 类别 | 文件 | 为什么争 |
| --- | --- | --- |
| 手工追加点 | `registry.py`、`__init__.py`（三处：import 块 / `__all__` / `_register_lazy`）、`CHANGELOG.md`、`MIGRATION.md`、`CLAUDE.md` 本身 | 两边都往同一段尾部追加 |
| **计数行** | `README.md`（英文，默认，PyPI 长描述用它）、`README_CN.md`（中文）、`docs/index.md`、`docs/reference/index.md` | 四处手写的「N 个注册函数」，由 `registry_stats.py --check` 门控。**对方加一个函数，你这四行同时作废** |
| 生成产物 | `schemas/*` **与** `src/statspai/schemas/*`（包内镜像）、`_parity_index.json`、`docs/parity.md`、`docs/stats.md`（两类冲突：at-a-glance 行 + 按模块行） | 纯粹因为两边都重新生成 |
| 证据视图（2026-10 起） | `docs/evidence_inventory.{json,md}`、`docs/reproduction_manifest.json`、`docs/dev/agent_card_audit.{json,md}`、`docs/dev/2026-10-02-review-status.md` | 都是生成的，各有 `--check`。**何时重生**：改了 `validation_scope.py`、任何 `tests/stata_parity/option_parity/` 或 `tests/stata_translation_holdout/` 下的文件、Track B 结果 → `python scripts/build_evidence_inventory.py`（在 `build_parity_index.py` 之前）；重推导了 R / Stata golden 或改了 `r-parity.yml` 的模块列表 → `python scripts/build_reproduction_manifest.py`；给那 60 个被审查的入口（`scripts/agent_card_audit.py::AUDITED`）加了 schema 枚举值 → **不必立刻重跑**：`tests/test_agent_card_audit.py` 会把报告没见过的新值现场真实调用一遍，只在函数拒绝该值时失败；方便时再 `python scripts/agent_card_audit.py`（约 10 分钟）刷新报告，新值积压超过 25 个时测试会要求刷新；改了 `docs/dev/review_backlog.json` → `python scripts/build_review_status.py` |
| 字节同步对 | `paper.bib` ↔ `src/statspai/paper.bib` | 必须逐字节一致 |
| 棘轮基线 | `scripts/signature_house_style_baseline.json`、`quality_gate` 的 mypy / flake8 基线 | 只降不升，两边都想动 |

**计数行不要自己算。** 别拿「我加了 2 个函数所以 1,190 → 1,192」去改——并行期间对方也在加。跑 `python scripts/registry_stats.py --check`，**它报什么数字就填什么**，再用 `--table` 重生 `docs/stats.md` 的对应模块行。2026-09-23/24 一晚上这个数被推了四次（1,189 → 1,190 → 1,192 → 1,197 → 1,198）。

**派生产物有生成顺序：`build_parity_index.py` 必须在 `dump_schemas.py` 之前。** registry 的证据备注由 parity index 派生（§5.1 末句），schema 包又把备注嵌进去；顺序反了，推送时 `schema-drift` 闸门会拦，而报错只说 schemas 陈旧，不会提示是索引的锅。

**rebase 撞到生成产物时不要手工合并冲突**——取对方的版本再重生（`checkout origin/main -- <那批派生文件>`，然后按上面的顺序重跑两个脚本，最后按 `registry_stats.py --check` 报的数字改那四行计数）。只有手工追加点（`CHANGELOG.md` / `CLAUDE.md` / `registry.py`）才逐块合。注意对方**发版**时会把你的 `## [Unreleased]` 段整个提升成 `## [X.Y.Z]`——你的条目要另起一个新的 Unreleased 段，不要塞回已发布的版本里。

**提交被 hook 改文件而失败之后，不要顺手 `--amend`。** pre-commit 的 black / isort 会重写文件并让本次提交失败；此时最自然的下一步「重新 add 再 `--amend --no-edit`」会把新改动并进**上一个、很可能已经推送过的** commit，挂在毫不相干的 message 下。2026-09-23 真发生过，靠 `reset --soft <那个已推送的 sha>` 才救回来。正确做法是重新暂存后**新起一次**提交。

**做法**：

```bash
git worktree add .claude/worktrees/<线名> -b wt/<线名>
cd .claude/worktrees/<线名>
```

完成后**不必切回主树**即可并入 main（主树可能压着别人未提交的工作，绝不要在那里 merge/checkout）——推送时用 `HEAD:main` 引用即可快进；非快进先 `git rebase origin/main`。

**必须带 PYTHONPATH。** 仓库是 editable 安装，`statspai` 被钉死在**主**工作树，裸跑 `import statspai` 仍会加载主树代码——测试跑在别人的改动上，隔离形同虚设：

```bash
PYTHONPATH="$(pwd)/src" python3 -m pytest ...
PYTHONPATH="$(pwd)/src" python3 scripts/dump_schemas.py
```

自检：worktree 内 `len(sp.list_functions())` 必须等于 `python scripts/registry_stats.py --check` 报的数字，不等就是没生效。**不要**为此往 `pyproject.toml` 加 pytest `pythonpath`——主树的 editable 安装对另一条线是正确的，改共享配置等于把刚消除的争用造回去。

**如果只能留在主树**：提交前务必 `git status` 确认哪些改动不是自己的，**逐文件 / 逐 hunk** 暂存（`git add <file>` 或 `git apply --cached`），绝不用 `-A` / `-a` 全量提交；派生产物要先把共享源文件还原到 HEAD 再重新生成，否则会把别人未提交的内容一起固化进去。

### 9.1 例外：远程 runtime（Colab / Lambda / RunPod / CI）回传结果走 PR

直推 main 的前提是"操作在本地，作者审过"。从远程 runtime（**最典型的是 [`Paper-JSS/colab_gpu_bench.ipynb`](Paper-JSS/colab_gpu_bench.ipynb)**）自动回传 benchmark 结果时，本地审视环节缺失，**必须改走 PR**：

- **允许的 payload**：
  - `tests/perf/results/05_*.json`（或对应 bench 编号的 JSON）
  - `tests/perf/results/_provenance_*.txt`（git SHA / JAX 版本 / GPU 型号）
  - `tests/perf/results/_log_full_*.txt`（subprocess stdout/stderr）
  - 可选：同步更新的 `paper.md` / `Paper-JSS/manuscript/sections/06-performance.tex` 表格数字
- **禁止的 payload**：源码改动、依赖变更、新增其他模块——这些走常规直推 main，不能搭 benchmark PR 顺风车。
- **分支命名**：`bench/<bench-name>-<gpu-tag>-<yyyymmdd>`，例：`bench/05-feols-t4-20260518`。
- **PR 标题**：`bench(<bench-name>): <gpu-tag> results — n=..., B=..., commit=<short-sha>`。
- **PR body 必填**：
  - 跑这次 benchmark 用的 git commit SHA（应与 `_provenance_commit.txt` 一致）
  - GPU 型号 / JAX 版本（与 `_provenance_gpu.txt` / `_provenance_jax.txt` 一致）
  - 验收点：`cell 16` 输出的 speedup 表格 + 自动生成的 LaTeX snippet 贴在 body 里
- **身份**：远程 runtime 用 `GITHUB_TOKEN`（fine-grained，仅 `contents:write` + `pull-requests:write`，scope 限本仓库），不要复用 user PAT。Token 通过 Colab Secrets 或 runtime env 注入，**禁止**写进 notebook 源码或 commit message。
- **合并策略**：本地审视过 JSON 数字合理（speedup 不离谱 / 没退化）后 **squash merge**；不合理则 close + 排查。
- **频率**：同一 bench 同一 GPU 同一天**只允许一个 open PR**，避免噪音。重跑覆盖旧 PR 用 force-push 同分支（这一条是 §9 "禁止 force" 的例外，因为 PR 本身就是审视点）。
