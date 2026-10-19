"""Turn ``stata_margins.tsv`` (written by ``stata_margins.do``) into JSON."""

import json
from collections import defaultdict

out: dict = defaultdict(dict)
with open("stata_margins.tsv", encoding="utf-8") as fh:
    for line in fh:
        tag, name, b, se = line.rstrip("\n").split("\t")
        out[tag][name.strip()] = [float(b), float(se)]
with open("stata_margins.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)  # keep Stata's order
    fh.write("\n")
