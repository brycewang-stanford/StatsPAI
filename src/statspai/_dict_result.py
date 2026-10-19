"""Result objects for functions that historically returned a plain ``dict``.

``sp.bacon_decomposition``, ``sp.evalue``, ``sp.oster_bounds`` and
``sp.mi_estimate`` returned bare dicts, so they had no ``summary()``,
``to_latex()``, ``tidy()`` ... and ``sp.regtable`` could not take them. A
:class:`DictResult` **is** a ``dict`` (every key, ``isinstance(r, dict)``,
``json.dumps(r)``, ``r == {...}`` and ``**r`` behave exactly as before) and
also carries the shared result protocol. Keys are readable as attributes
(``r.beta_twfe``) unless a method or property of the same name exists, so
``r["summary"]`` stays the stored table while ``r.summary()`` renders text.
"""

from __future__ import annotations

from typing import Any, ClassVar, List

import numpy as np
import pandas as pd

from ._result_serialize import ResultProtocolMixin


class DictResult(ResultProtocolMixin, dict):
    """A ``dict`` result with ``summary`` / ``tidy`` / ``glance`` / exports."""

    #: Heading of :meth:`summary`.
    _title: ClassVar[str] = "Result"

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails; never for dunders, so
        # pickling and copying keep their default behaviour.
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute or key {name!r}"
            ) from None

    def __dir__(self) -> List[str]:
        return sorted(set(super().__dir__()) | {str(k) for k in self})

    def summary(self) -> str:
        """Plain-text rendering: scalar entries, then any tables."""
        lines = [self._title, "=" * max(40, len(self._title))]
        tables = []
        for key, val in self.items():
            if isinstance(val, pd.DataFrame):
                tables.append((key, val))
            elif isinstance(val, (bool, np.bool_)):
                lines.append(f"{key:<36} {bool(val)}")
            elif isinstance(val, (int, float, np.integer, np.floating)):
                lines.append(f"{key:<36} {float(val):.6g}")
            elif isinstance(val, str) and "\n" not in val and len(val) <= 80:
                lines.append(f"{key:<36} {val}")
            elif isinstance(val, tuple) and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in val
            ):
                lines.append(
                    f"{key:<36} (" + ", ".join(f"{float(v):.6g}" for v in val) + ")"
                )
        for key, frame in tables:
            lines += ["", f"{key}:", frame.to_string(index=False)]
        return "\n".join(lines)
