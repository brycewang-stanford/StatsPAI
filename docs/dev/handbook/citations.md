# 引用与文献：全文

> 从 CLAUDE.md §10 原样迁来（2026-10-30）。CLAUDE.md 只留硬约束和指向本文件的链接；新增引用、改 `paper.bib` 或抽取论文 bib 之前先读完本文件。新的细则直接写在这里。

## 10. 引用与文献（零幻觉红线）

> 捏造一条引用，代价是整个包数值正确性的可信度。一位计量经济学读者点开 DOI 发现不存在，会立刻怀疑 StatsPAI 的所有估计器——**这是最便宜的质量杀手，必须零容忍**。

### 四要素核验

任何**新增**引用（docstring / README / `CHANGELOG.md` / `MIGRATION.md` / `paper.md` / `docs/guides/` / commit message 均适用）落地前必须独立核验：

1. **作者**——完整名单、拼写、顺序、大小写
2. **年份**——正式发表年（若引预印本，显式标注 `arXiv preprint, YEAR`）
3. **标题**——完整标题，不省略副标题
4. **期刊 / 会议 / 出版方** + **DOI 或 arXiv ID**

核验来源至少 **2 个独立渠道**（Crossref / doi.org / arXiv / 期刊官网 / Google Scholar 取其二），单一二手来源不作数。**禁止凭 LLM 或训练语料记忆补全**——哪怕是 Abadie (2003)、Callaway & Sant'Anna (2021)、Chernozhukov et al. (2018) 这类你"非常确定"的论文，也一律现场核验后再写入文件。

### `paper.bib` 单一来源

- 核验通过的引用一律落到 [`paper.bib`](paper.bib)，bib key 用 `lastnameYEARkeyword` 规范（例：`abadie2003economic`、`callaway2021difference`、`chernozhukov2018double`）。
- docstring 的 `References` 段、`docs/` 教程、`paper.md` 正文**只引 bib key 或引用一份规范字符串**；禁止在多处手写同一条引用，避免格式漂移。
- 发现 `paper.bib` 已有条目信息错漏 → 修一处全仓受益；同时 `git grep` 扫手写副本一并修正，防止旧字符串残留。

### master 与派生子集（2026-09-04 起）

- 根目录 `paper.bib` 是 **master**：全仓唯一可手工编辑的 bib。JOSS 论文（`paper.md`）直接引用它；docstring、`sp.bibtex()`、MCP `bibtex` 工具也读它。
- **各篇论文的 bib 一律是派生子集，禁止手改**。JSS 手稿的 `Paper-JSS/manuscript/jss-bib.bib` 由 `python tools/bib_subset.py extract --roots Paper-JSS/manuscript/main.tex --out Paper-JSS/manuscript/jss-bib.bib --keep ...` 从 master 抽取（`make -C Paper-JSS bib-split`）；手稿要引新文献，先按四要素核验加进 master，再重新抽取。长版章节独占的引用走第二个派生文件 `jss-bib-archival.bib`（`--roots main.tex sections/*.tex --minus jss-bib.bib`）。`bib_subset.py check` 是漂移闸门（pre-push hook `bib-subset-jss` / `bib-subset-jss-archival`）。
- **wheel 里带一份 master 副本** `src/statspai/paper.bib`，必须与根目录字节一致（`python tools/bib_subset.py packaged --sync` 同步；pre-commit hook `bib-packaged-sync` 和 citation-audit 的 Gate 1b 校验）。运行时通过 `statspai._bibpath.master_bib_path()` 解析：源码树优先，其次包内副本，两者都没有则抛 `FileNotFoundError`——**不再**回退到当前目录的 `paper.bib`。
- **条目必须 BibTeX 安全**：字段内**禁止原始非 ASCII 字符**（`Ørregaard` 要写 `{\O}rregaard`，`é` 写 `{\'e}`，破折号写 `--`，`R²` 写 `$R^2$`），否则 BibTeX 做姓名缩写会切断多字节序列，PDF 里出现乱码。**核验记录写在 `annote`**（所有 .bst 样式都忽略它），`note` 只放读者可见的短句（如 "arXiv preprint, first posted 2025-06-21"），因为 `note` 会被打印进参考文献，而且里面的 `_` `&` `%` 会直接让 LaTeX 报错。`tests/test_bib_subset.py` 有守卫测试。
- 同一文献只能有一个 key。发现某篇论文用了不同 key 引同一文献，改论文的 key，不加重复条目（`audit_bib_duplicates.py --strict` 会挡）。

### 交付前自检

- Commit / PR 触及新引用 → 在 message / 描述里注明 "refs verified via `<source1>`, `<source2>`"。
- Review / self-review 对任何陌生引用默认按"未核验"处理——要求 DOI 可点开、arXiv 可访问，或 Crossref 能直接搜到。
- **宁缺毋滥**：拿不准时写 `（citation needed）`、只引 bib key 占位、或干脆不引——**捏造一条引用比缺失糟糕一百倍**。
