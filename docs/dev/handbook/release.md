# 发布流程与归档

> 从 CLAUDE.md §7 与「其它关键事项」 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；发版之前先读完本文件。新的细则直接写在这里。

## 7. 发布

PyPI 凭据在 `~/.pypirc`——**不要**提交仓库、不要写进 memory。完整流程见 `memory/reference_pypi_publish.md`。

简流程：

1. Bump `pyproject.toml` + `__version__`。
2. 更新 [`CHANGELOG.md`](CHANGELOG.md)（`Added / Changed / Fixed / ⚠️ Correctness`）。
   版本号一改、tag 还没打，就进入**发版窗口**：`python scripts/release_gate.py --fix-census` 把 `docs/guides/stability.md` 与 `docs/jss_source_audit_dossier.md` 里的 registry 统计改成当前值（同时把根目录 `server.json` 的两处版本号改成新版本），再 `python scripts/release_gate.py --check` 确认（pre-push hook `release-gate` 与 `tests/test_release_gate.py` 在窗口内强制统计一致，窗口外只查另外两项）。这三项是 1.33.0 / 1.34.0 漏掉、靠构建论文复制包才发现、各赔了一个补丁版本的问题：统计过时、正则被 ASCII 转写改了语义（模式里的非 ASCII 字符一律写 `\uXXXX`）、包内文本出现宣传措辞。
3. `pytest -q` 全绿 + `pytest tests/reference_parity/ -q` 必过。
4. `rm -rf dist/ && python -m build && twine check dist/*`。**再把 sdist 的文件清单对照 `git ls-files`**：`MANIFEST.in` 的 `recursive-include tests *` 会把工作树里被 gitignore 的本地产物一起打包——1.32.0 发版时先后抓到复现检查的临时目录、Track C 的 38 MB 共享输入、以及可由包内 `sp.datasets.nhefs()` 逐字节重生、因而不入库的 NHEFS 副本，sdist 一度从 25 MB 涨到 63 MB。新出现的 gitignore 产物目录要在 `MANIFEST.in` 里 `prune`。
5. 干净 venv 装 wheel 冒烟测试。
6. `git tag vX.Y.Z && git push && git push --tags`。
7. `twine upload dist/*`。

**默认不发 GitHub Release（2026-09-26 起）。** 发版 = TestPyPI + PyPI（本地 twine）+ 打 tag，到此为止；**不要**跑 `gh release create`，也不要在网页上 Publish Release。原因：Publish Release 会同时触发两个不可逆动作——Zenodo 铸出永久不可删的 version DOI，以及 `ci-cd.yml` 的 `release: published` 再上传一次 PyPI（与本地 twine 重复）。只推 tag 两者都不触发。只有用户在**当前会话**明确要求（如期刊需要 Zenodo version DOI）时才发，并先按下文「Zenodo 归档」一条做核对；发版总结里要写明「未创建 GitHub Release」。

## Zenodo 与 JOSS 归档

- **Zenodo 归档：只有 "Publish 一个 GitHub Release" 会触发。** commit / push / 打 tag / 发 PyPI 新版都**不触发**归档。Publish Release 会同时触发 GitHub↔Zenodo 自动归档（铸出**永久不可删的 version DOI**）和 `ci-cd.yml` 的 `publish-prod`（上传 PyPI），一次两个不可逆动作——发之前务必先手动跑一轮 `ci-cd.yml` 的 `test_scope=full`，并核对 `.zenodo.json` / `CITATION.cff` 的标题、作者、ORCID、单位、license 与 `paper.md` 逐项一致。
- **JOSS #10604 的归档已完成（2026-08-24）。** 时序上要注意：JOSS 要求**在 `recommend-accept` 之前**交出 archive DOI，不是接收之后——编辑 8/18 的 post-review checklist 明确索要 DOI 和版本号。本次交付的是 **v1.23.0**，version DOI `10.5281/zenodo.22085759`；concept DOI 仍是 `10.5281/zenodo.19933900`（永远指向最新归档）。此后再发 Release 会继续铸新的 version DOI，**不影响本次投稿**（JOSS 锚的是已提交的那一个）。
