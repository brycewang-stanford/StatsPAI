# 对外 API 的细则

> 从 CLAUDE.md §4 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；给已注册函数加参数、改 agent 卡片、改核心层签名或 `_frame_input.py` 之前先读完本文件。新的细则直接写在这里。

- 给已注册函数**加参数也要登记**：在它的 `params=[...]` 里补一条 `ParamSpec`（描述就是 agent 看到的文档），再 `python scripts/dump_schemas.py`。漏了的话 `sp.function_schema()` 里没有这个参数，agent 不知道它存在；pre-push hook `registry-params`（2026-10-05 起）会拦。
- **agent 卡片里能从代码读出来的内容不要手写（2026-10-14 起）。** 三处各有一个来源和一道闸门：(1) 选择型参数的 `enum`：注解是 `Literal`、或 docstring 里的列表被模块源码印证时自动得到；只由代码陈述的（函数体里的校验、模块常量、原样转交给另一个函数）登记在 `src/statspai/_enum_seeds.py`，`tests/test_enum_seeds.py` 会回到那个位置重新核对，**不要**把记忆里的取值写进去，语法型参数（`vce='cluster firm'`、lavaan 模型串）不给枚举。(2) 没有任何失败模式的函数，取自它自己抛出的带 `recovery_hint` 的类型化异常：改了这类 `raise` 之后跑 `python scripts/gen_source_failure_modes.py`（`tests/test_source_failure_modes.py` 查漂移），所以**新函数用 taxonomy 异常并写 `recovery_hint`，卡片就自动有内容**。(3) `sp.search_functions` 的中文词表在 `src/statspai/_discovery_zh.py`，条目写成 `\uXXXX` 转义、上一行注释写汉字（JSS 归档会把源码里的非 ASCII 转写掉），加词时两处都要写，测试会比对。工具层（MCP / `execute_tool`）对未传的 `seed=None` 类参数注入默认种子并在返回里写 `seed_injected`（`agent/_default_seed.py`）；库函数自己的默认值不动。
- **核心层签名冻结（2026-10-17 起）。** `CORE_FUNCTIONS`（`sp.list_functions(core=True)` 的 32 个）的签名记在 `tests/core_api_snapshot.json`，删参、改默认值、可选变必选、调换位置参数、新增无默认值的参数都会被 `tests/test_core_api_freeze.py`（pre-push hook `core-api-freeze`）拦住。新增带默认值的参数不算破坏。有意的破坏先走弃用流程，再 `python scripts/core_api_snapshot.py --write --ack "fn.param=<MIGRATION.md 里的原文>"`。
- **`sp.*` 统一接受 polars / Arrow（2026-10-17 起）。** `statspai/__init__.py` 末尾用 `_frame_input.install` 包装每个公开函数（惰性导出在 `__getattr__` 里包装），并在定义模块里同名换绑，所以 `模块.函数 is sp.函数`、pickle 按引用可用、`__module__` 不变。polars / pyarrow / `__arrow_c_stream__` / `__dataframe__` 对象在调用时转成 pandas；pandas 参数**以同一对象**直通（`tests/test_frame_input.py` 守着）。正因为直通，`_frame_input.py` 被两个 provenance tracer 排除在估计路径之外；**改这个文件时不得让 pandas 输入走任何计算**，否则排除就不成立。新函数不用做任何事；不要在估计器里再写 polars 分支。

### 模块内部

- 共享基元放模块级 `_core.py` / `_common.py`（参考 `rd/_core.py`、`decomposition/_common.py`）。**不要**在多个文件重复实现 kernel / WLS / sandwich / 影响函数。
- 私有函数 `_` 前缀。
- 单文件 ~800 行以内，按关注点拆（estimator / inference / diagnostics / plots），不是单纯按行数拆。

### 依赖

- 核心依赖精简（见 `pyproject.toml`）。重依赖归入 extras：`dev` / `performance` (jax) / `bayes` (pymc) / `neural` `deepiv` (torch) / `fixest` (pyfixest) / `plotting`。
- `torch` / `jax` / `pymc` 必须**惰性 import**，用户没装 extra 不应触发 `ImportError`。
- 禁止引入 GPL / AGPL 依赖（和 MIT 冲突）。
