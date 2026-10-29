# Rust 组件（statspai-hdfe）

> 从 CLAUDE.md §6 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；改 `rust/statspai_hdfe/`、`_rust_backend.py` 或发布 `statspai-hdfe` 之前先读完本文件。新的细则直接写在这里。

## 6. Rust 组件

路径 [`rust/statspai_hdfe/`](rust/statspai_hdfe/)：HDFE demean、GLM（Poisson / logit / probit）的整段 IRLS、聚类 meat 的 Rust 内核，PyO3 + maturin，**单独发布为 PyPI 包 `statspai-hdfe`**。StatsPAI 本体仍是纯 Python wheel，不改 build-backend；**2026-10-18 起 `statspai-hdfe` 是核心依赖，但只限有 wheel 的平台**（`pyproject.toml` 里的环境标记：macOS arm64/x86_64、Linux x86_64/aarch64（glibc 与 musl）、Windows AMD64/ARM64），这样 pip 永远不需要编译 Rust；其他平台走 NumPy / Numba，`statspai[rust]` 在那里从 sdist 编译。加平台先在 `rust-hdfe.yml` 出 wheel，再放宽标记。`sp.feols` / `sp.fepois` / `sp.feglm` 默认（`backend="auto"`）走这套内核，见 `fixest/_native.py`。行方向的循环（`par.rs`）按问题规模固定分块、按块序相加，**结果与线程数无关**（逐位一致，有测试守着）；改并行代码时保持这一点。

```bash
uvx maturin build --release -o /tmp/w && pip install /tmp/w/statspai_hdfe-*.whl   # 本地装
PYO3_PYTHON=python cargo test --release                                         # crate 单测（rust/statspai_hdfe 下）
```

- **唯一入口是 `statspai/_rust_backend.py`**：每个想用 Rust 的模块都从这里拿 `rust`（模块或 `None`），不要再写 `import statspai_hdfe`。它认 `STATSPAI_SKIP_RUST=1`（2026-10-18 之前这个变量在 CI 里到处设，却从没有代码读它），并检查 `statspai_hdfe.API_VERSION >= MIN_API_VERSION`；入口语义一变就升 `lib.rs::API_VERSION`，并同步 `MIN_API_VERSION`。
- **谁用它**：`sp.fast.*`、`sp.hdfe_ols` / `sp.Absorber` / `sp.demean`（纯 FE、`solver="map"`、无 slope 时）、`sp.ppmlhdfe`（K≥2）、`sp.iv(absorb=)`、`didregress`。一维聚类的 `sp.regress`（`core/_numba_kernels.cluster_meat`）2026-10-19 起不再用 Rust：它要求按聚类排好序的行，排序比求和贵，现在是一次 `bincount`。`sp.feols`（pyfixest）、`sp.panel(method="fe")`（linearmodels）不用。
- **数值约定**：Rust 与 NumPy 解的是同一个投影，只在停止容差内一致，**不是逐位一致**；测试对照稠密最小二乘的精确投影（`tests/test_rust_backend.py`），不要写"两条路径 atol=1e-14"那种断言。结果与线程数无关（demean 按列独立；`cluster_meat` 固定分块后按序相加）。
- **Parity 一律在无 Rust 路径上生成与复现**：`verify_reproduce_py.py` 与 `trace_parity_provenance.py` 给子进程设 `STATSPAI_SKIP_RUST=1`（`setdefault`）。装了扩展的环境里 `sp.hdfe_ols` 等的末位会变（容差内），已提交的 `*_py.json` 只对无扩展路径成立。
- **共线检测不能靠 `np.linalg.inv` 抛错**：被 FE 吸收的列从任何迭代 demean 回来都是舍入噪声，不是精确的 0。`fast/_validation.py::require_identified_regressors` 是共用检查，新的 `sp.fast` 估计器在解正规方程前调用它。
- **安全**：所有 PyO3 入口在进入 Rust 前校验长度 / 连续性 / 编码范围并抛 `ValueError`；`panic = "unwind"`（以前是 `abort`，任何越界都会直接杀掉解释器）。线程池是 crate 自己的（`sp.fast.set_num_threads` / `STATSPAI_NUM_THREADS`），fork 后在子进程里重建。
- **CI**：`.github/workflows/rust-hdfe.yml` 在三个平台跑 `cargo test` 和带 `STATSPAI_REQUIRE_RUST=1` 的 Rust 路径测试；wheels（8 个平台 + sdist；musllinux aarch64 与 Windows ARM64 从 0.9.1 起）在手动触发或 `hdfe-v*` tag 时构建。**工作流从不上传。**
- **发布 `statspai-hdfe`（2026-10-18 用户定的规矩）：只用 `~/.pypirc` 里的 token + 本地 twine，不用浏览器登录 PyPI，不配 trusted publishing / CI 发布任务。** 步骤：升 `Cargo.toml` 版本（入口语义变了同步 `API_VERSION` / `MIN_API_VERSION`）并推送 → `gh workflow run rust-hdfe.yml --ref main` → `gh run download <id>` 收齐 8 个 wheel + sdist → `twine check` + 干净 venv 冒烟 → `twine upload -r testpypi` 并从 TestPyPI 装回验证 → `twine upload -r pypi`。上传属于 §9 需要当次授权的操作；同一版本号不能重传。0.8.0、0.9.0（API 3）均按此于 2026-10-18 发布，0.9.1（同一代码，补 musllinux aarch64 与 Windows ARM64 wheel；后者在原生 `windows-11-arm` runner 上构建，x64 上交叉编译会因缺 ARM64 解释器失败）于 2026-10-20 发布（均未打 tag）；StatsPAI 的依赖下限随之升到 0.9.1。
- **Track C**：`run_when_idle.sh` 把 `STATSPAI_NUM_THREADS` / `RAYON_NUM_THREADS` 钉为 1；`trace_perf_path.py` 只在计时时加载了 Rust 的记录里绑定 crate 源码哈希（`rust_loaded`），所以没装 Rust 时改 crate 不会让计时过期。
