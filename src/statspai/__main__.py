"""``python -m statspai``: the ``statspai`` command, plus ``--where``.

``python -m statspai --where`` prints which StatsPAI this interpreter
imports, from where, and in which environment, without loading the
estimators. It is the quickest answer to "which Python has StatsPAI?".
Every other argument list is handed to the ``statspai`` CLI, so
``python -m statspai doctor`` is the same as ``statspai doctor``.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--where"], ["where"]):
        from ._doctor import where_text

        print(where_text())
        return 0
    from .cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
