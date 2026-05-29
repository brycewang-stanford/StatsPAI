"""
Visualization module for StatsPAI.

Provides publication-quality academic plots:
- binscatter: Binned scatter plots with residualization (Cattaneo et al. 2024)
- coefplot: Coefficient comparison forest plots
- event_study_plot: DID event study (via CausalResult)
- rdplot: RD visualization (via rd module)
- marginsplot: Marginal effects (via postestimation)
- interactive: Interactive plot editor with data protection
"""

from .binscatter import binscatter
from .themes import set_theme, list_themes, use_chinese, _register_cjk_fallback


def __getattr__(name):
    if name in {"interactive", "get_code", "FigureEditor"}:
        from .interactive import FigureEditor, get_code, interactive
        bindings = {
            "interactive": interactive,
            "get_code": get_code,
            "FigureEditor": FigureEditor,
        }
        globals().update(bindings)
        return bindings[name]
    raise AttributeError(f"module 'statspai.plots' has no attribute {name!r}")

# --------------------------------------------------------------------------
# CJK font fallback — registered LAZILY, the first time matplotlib.pyplot is
# imported anywhere in the process.
#
# Why lazy: calling _register_cjk_fallback() at import time eagerly imports
# matplotlib (and scans the font manager), adding ~50-200ms to every
# `import statspai` even for the (vast) majority of agent calls that never
# plot. binscatter/themes already import matplotlib lazily inside their
# functions, so this auto-register was the *only* thing forcing matplotlib at
# import.
#
# Why a pyplot import hook (and not "register on first statspai plot call"):
# the CJK fallback mutates the GLOBAL rcParams['font.family'], so it must be
# active before the first figure is rendered by ANY path — including
# sp.rdplot / DID / synth plots that go straight to matplotlib without
# touching statspai.plots. Hooking the single chokepoint that every render
# path passes through (`import matplotlib.pyplot`) preserves the original
# out-of-the-box behavior exactly, with zero import-time cost when no plot is
# ever drawn.
#
# Opt-out: STATSPAI_NO_AUTO_CJK=1. User's later rcParams[...] assignment wins.
# --------------------------------------------------------------------------
def _ensure_cjk_fallback():
    # Idempotent: _register_cjk_fallback guards on its own _registered flag.
    _register_cjk_fallback()


def _install_cjk_pyplot_hook():
    import sys

    if "matplotlib.pyplot" in sys.modules:
        # pyplot already loaded (someone imported it before statspai) — the
        # render chokepoint is already past, so register immediately.
        _ensure_cjk_fallback()
        return

    import importlib.abc
    import importlib.util

    class _PyplotCJKHook(importlib.abc.MetaPathFinder):
        """One-shot finder: register CJK right after matplotlib.pyplot loads."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "matplotlib.pyplot":
                return None
            # Remove ourselves first so the find_spec below (and any future
            # imports) bypass us — no recursion, no per-import overhead.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_loader = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return real_loader.create_module(spec)

                def exec_module(self, module):
                    real_loader.exec_module(module)
                    try:
                        _ensure_cjk_fallback()
                    except Exception:
                        # Never let CJK registration break a pyplot import.
                        pass

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _PyplotCJKHook())


_install_cjk_pyplot_hook()

__all__ = [
    'binscatter',
    'set_theme',
    'list_themes',
    'use_chinese',
    'interactive',
    'get_code',
    'FigureEditor',
]
