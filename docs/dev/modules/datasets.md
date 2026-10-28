# 数据集（sp.datasets）

> 2026-10-22 新建。改 `src/statspai/datasets/` 之前先读本文件。

- **随包与按需下载（`datasets/_fetch.py`，2026-10-21 起）**：只有许可证明确允许再分发、体积又小的数据才进 wheel。其余（JEL-DiD 县级数据 6.3 MB、pped.org 的 `bacon_example.dta`、friosavila 的 drdid `lalonde.dta`、原始 mpdta）由加载函数在**第一次调用时**从规范的公开地址下载，用钉死的 SHA-256 校验后放进用户缓存：`$STATSPAI_DATA_DIR`，否则 `$XDG_CACHE_HOME/statspai/datasets`（Windows 是 `%LOCALAPPDATA%\statspai\datasets`）。import 和 `sp.list_datasets()` 都不下载。哈希不符的文件一律不用（缓存里的重新下载，新下载的丢弃）。离线时报 `DataInsufficient`，消息里给出 URL、哈希和应放的路径，用户可以手工放置；`download=False` 禁止联网。新增这类数据集时在 `RemoteFile` 里写 `licence` 和 `origin`，并在 `sp.list_datasets()` 的 `delivery` / `available_offline` 列里出现。
- **pped.org 的 HTTPS 证书已过期**，`bacon_example.dta` 走明文 HTTP，安全性靠钉死的哈希。不要为此关闭证书校验。
- **`sp.datasets.mpdta()` 默认仍是模拟副本**（R `did` 是 GPL-3，原始数据不随包），`df.attrs['simulated']` 为真并告警；`mpdta(simulated=False)` 下载原始面板。两者的 `treat` 含义不同：模拟副本是随时间变化的 D_it，原始数据是"曾经处理"的组哑变量。测试里需要原始数据时要能离线跳过；`tests/test_recommend_synth_detection.py` 曾因 `loader(simulated=False)` 在加了这个参数后悄悄改为下载原始数据而失败（2026-10-22 修）。
- **测试**：需要下载数据的测试在数据未缓存且 `STATSPAI_FETCH_DATASETS` 未设时跳过，并写明原因。nightly 的全量覆盖率任务设了这个变量（约 7 MB 下载），push 触发的 CI 不下载。目录类测试只加载随包数据。
- **sdist**：缓存目录在仓库之外，不会进 sdist；若有人把数据下载到仓库内，`MANIFEST.in` 的 `recursive-include tests *` 会把它打包，发版前按 CLAUDE.md §7 第 4 步核对文件清单。
- **随包数据与原文数据不一定相同**：`nsw_lalonde` 是 MatchIt 的 614 行截面，不是 Sant'Anna-Zhao 用的 NSW + CPS + PSID 面板（那是 `lalonde_panel()`）；`castle_doctrine` 是 Mixtape `castle.dta` 的 29 列子集，`popwt` 等 float32 列有 1e-7 量级的差。论文复现画廊见 `docs/guides/did_paper_gallery.md`，对数测试在 `tests/external_parity/test_did_paper_gallery.py`。
- **`rust_bus`（2026-10-28）。** 从 Zenodo 3385050（zurcher-data v0.14，OpenSourceEconomics，MIT）按需下载 319 kB 的 zip，钉 SHA-256 `8b1c267c…f16f`；九个 `.asc` 与 Rust 的 `nfxp.zip` 逐值相同。数据本身无单独许可证，因此只下载、不随包。测试在 `tests/external_parity/test_rust1987_bus.py`，未缓存且未设 `STATSPAI_FETCH_DATASETS` 时跳过；离线部分在 `tests/test_rust_bus_loader.py`。
- **`china_shock`（2026-10-28）。** Autor-Dorn-Hanson (2013) 的 `workfile_china.dta`（1,444 行 × 191 列，722 个通勤区 × 两个十年），从 Agarwal-Singh 的 Zenodo 复现包（10.5281/zenodo.19866751，9.6 MB，包内 `data/LICENSE.txt` 为 CC-BY-4.0，© 2013 AEA）按需下载、钉 SHA-256，只下载不随包。名字不用 `adh`，因为 `california_prop99` 是另一个 ADH（Abadie-Diamond-Hainmueller）。Stata 的单精度列转成 float64，取的是 float32 的值本身，不是十进制重舍入；整数和哑变量保持原类型。`RemoteFile` 定义在 `datasets/_china_shock.py`，ADH parity 测试从这里导入。
