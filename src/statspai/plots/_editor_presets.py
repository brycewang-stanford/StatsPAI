"""Font and size presets for the figure editor.

Moved out of ``interactive.py``; everything here is re-exported there so
``from statspai.plots.interactive import FONT_PRESETS`` keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exceptions import MethodIncompatibility


def _detect_chinese_fonts() -> Dict[str, List[str]]:
    """
    Detect available Chinese fonts on the current system.

    Returns dict with 'serif' and 'sans-serif' lists, ordered by
    priority (most common/reliable first).
    """
    try:
        from matplotlib.font_manager import fontManager
    except ImportError:
        return {"serif": [], "sans-serif": []}

    available = {f.name for f in fontManager.ttflist}

    # Candidate fonts in priority order per platform
    serif_candidates = [
        # macOS
        "Songti SC",
        "Noto Serif CJK SC",
        "Hiragino Mincho ProN",
        # Windows
        "SimSun",
        "NSimSun",
        # Linux
        "WenQuanYi Micro Hei",
        # Cross-platform
        "STSong",
        "AR PL UMing CN",
    ]
    sans_candidates = [
        # macOS
        "PingFang SC",
        "PingFang HK",
        "Heiti TC",
        "Hiragino Sans GB",
        "Hiragino Sans",
        # Windows
        "Microsoft YaHei",
        "SimHei",
        # Linux
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        # Cross-platform
        "STHeiti",
        "Kaiti SC",
    ]

    found_serif = [f for f in serif_candidates if f in available]
    found_sans = [f for f in sans_candidates if f in available]

    return {"serif": found_serif, "sans-serif": found_sans}


def _get_chinese_serif() -> List[str]:
    """Get the best available Chinese serif font list for this system."""
    detected = _detect_chinese_fonts()
    result = detected["serif"]
    if not result:
        # Fallback: try sans-serif Chinese fonts + generic
        result = detected["sans-serif"]
    # Always append English serif as final fallback
    result.extend(["Times New Roman", "DejaVu Serif"])
    return result


def _get_chinese_sans() -> List[str]:
    """Get the best available Chinese sans-serif font list."""
    detected = _detect_chinese_fonts()
    result = detected["sans-serif"]
    if not result:
        result = detected["serif"]
    result.extend(["Helvetica", "Arial", "DejaVu Sans"])
    return result


FONT_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- Serif fonts ---
    "Times New Roman": {
        "family": "serif",
        "fonts": ["Times New Roman", "DejaVu Serif"],
        "title_size": 11,
        "label_size": 10,
        "tick_size": 9,
        "venue": "AER / Econometrica / APA / IEEE",
    },
    "Palatino": {
        "family": "serif",
        "fonts": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
        "title_size": 11,
        "label_size": 10,
        "tick_size": 9,
        "venue": "Econometrica / Book typesetting",
    },
    "Charter": {
        "family": "serif",
        "fonts": ["Charter", "XCharter", "DejaVu Serif"],
        "title_size": 11,
        "label_size": 10,
        "tick_size": 9,
        "venue": "R default / Academic papers",
    },
    "Computer Modern": {
        "family": "serif",
        "fonts": ["CMU Serif", "Computer Modern", "Latin Modern Roman", "DejaVu Serif"],
        "title_size": 11,
        "label_size": 10,
        "tick_size": 9,
        "venue": "LaTeX / Beamer default",
    },
    # --- Sans-serif fonts ---
    "Helvetica": {
        "family": "sans-serif",
        "fonts": ["Helvetica", "Arial", "DejaVu Sans"],
        "title_size": 10,
        "label_size": 9,
        "tick_size": 8,
        "venue": "Nature / Science / Elsevier / Springer",
    },
    "Arial": {
        "family": "sans-serif",
        "fonts": ["Arial", "Helvetica", "DejaVu Sans"],
        "title_size": 10,
        "label_size": 9,
        "tick_size": 8,
        "venue": "General / Office documents",
    },
    "Calibri": {
        "family": "sans-serif",
        "fonts": ["Calibri", "Arial", "DejaVu Sans"],
        "title_size": 11,
        "label_size": 10,
        "tick_size": 9,
        "venue": "Office / PowerPoint",
    },
    # --- Chinese fonts (auto-detect) ---
    "SimSun / 宋体": {
        "family": "serif",
        "fonts": None,  # filled at runtime by _get_chinese_serif()
        "title_size": 12,
        "label_size": 10.5,
        "tick_size": 9,
        "venue": "中文论文 / 学位论文",
    },
    "SimHei / 黑体": {
        "family": "sans-serif",
        "fonts": None,  # filled at runtime by _get_chinese_sans()
        "title_size": 12,
        "label_size": 10.5,
        "tick_size": 9,
        "venue": "中文 PPT / 幻灯片",
    },
}

# Size presets — independent of font choice.
# Users pick a font, then pick a size context.
SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Journal (compact)": {
        "title_size": 9,
        "label_size": 8,
        "tick_size": 7,
        "desc": "Nature / Science / small figures",
    },
    "Journal (standard)": {
        "title_size": 11,
        "label_size": 10,
        "tick_size": 9,
        "desc": "AER / Econometrica / most journals",
    },
    "Journal (large)": {
        "title_size": 12,
        "label_size": 12,
        "tick_size": 11,
        "desc": "APA / full-page figures",
    },
    "Thesis": {
        "title_size": 12,
        "label_size": 10.5,
        "tick_size": 9,
        "desc": "学位论文 / Dissertation",
    },
    "Slides": {
        "title_size": 16,
        "label_size": 14,
        "tick_size": 12,
        "desc": "Beamer / Keynote / PowerPoint",
    },
    "Poster": {
        "title_size": 20,
        "label_size": 16,
        "tick_size": 14,
        "desc": "Conference poster",
    },
}

# Backward compatibility aliases (old keys → new keys)
_PRESET_ALIASES: Dict[str, str] = {
    "AER / Econometrica": "Times New Roman",
    "APA (7th ed.)": "Times New Roman",
    "Nature / Science": "Helvetica",
    "IEEE": "Times New Roman",
    "Elsevier": "Helvetica",
    "Springer": "Helvetica",
    "CJK Thesis": "SimSun / 宋体",
    "CJK Journal": "SimSun / 宋体",
    "CJK Slide": "SimHei / 黑体",
    "Beamer / Slides": "Helvetica",
}


def _resolve_preset_fonts(preset: Dict[str, Any]) -> List[str]:
    """Resolve None font lists to auto-detected Chinese fonts."""
    fonts = preset["fonts"]
    if fonts is not None:
        return list(fonts)
    if preset["family"] == "serif":
        return _get_chinese_serif()
    return _get_chinese_sans()


# Common fonts — auto-detect Chinese availability
def _build_font_choices() -> Dict[str, List[str]]:
    """Build font choice list with system-detected Chinese fonts."""
    detected = _detect_chinese_fonts()

    # Build Chinese choices with display names
    cn_fonts: List[str] = []
    _cn_display = {
        "Songti SC": "Songti SC (宋体)",
        "SimSun": "SimSun (宋体)",
        "NSimSun": "NSimSun (新宋体)",
        "STSong": "STSong (华文宋体)",
        "Noto Serif CJK SC": "Noto Serif CJK SC (思源宋体)",
        "Hiragino Mincho ProN": "Hiragino Mincho ProN (ヒラギノ明朝)",
        "PingFang SC": "PingFang SC (苹方)",
        "PingFang HK": "PingFang HK (蘋方)",
        "Heiti TC": "Heiti TC (黑体)",
        "Hiragino Sans GB": "Hiragino Sans GB (冬青黑体)",
        "Hiragino Sans": "Hiragino Sans (ヒラギノ角ゴ)",
        "Microsoft YaHei": "Microsoft YaHei (微软雅黑)",
        "SimHei": "SimHei (黑体)",
        "STHeiti": "STHeiti (华文黑体)",
        "Kaiti SC": "Kaiti SC (楷体)",
        "Noto Sans CJK SC": "Noto Sans CJK SC (思源黑体)",
        "WenQuanYi Micro Hei": "WenQuanYi Micro Hei (文泉驿)",
    }
    seen: set[str] = set()
    for f in detected["serif"] + detected["sans-serif"]:
        if f not in seen:
            seen.add(f)
            cn_fonts.append(_cn_display.get(f, f))

    return {
        "English Serif": [
            "Times New Roman",
            "Palatino",
            "Georgia",
            "Garamond",
            "Computer Modern",
            "DejaVu Serif",
        ],
        "English Sans-serif": [
            "Helvetica",
            "Arial",
            "Calibri",
            "Verdana",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "CJK": (
            cn_fonts
            if cn_fonts
            else [
                "(No CJK fonts detected)",
            ]
        ),
        "Monospace": [
            "Courier New",
            "Consolas",
            "DejaVu Sans Mono",
            "Source Code Pro",
        ],
    }


# Lazy-initialized on first access
_font_choices_cache: Optional[Dict[str, List[str]]] = None


def get_font_choices() -> Dict[str, List[str]]:
    """Get available font choices (cached, auto-detects Chinese fonts)."""
    global _font_choices_cache
    if _font_choices_cache is None:
        _font_choices_cache = _build_font_choices()
    return _font_choices_cache


# Backward compatibility alias — call get_font_choices() directly
FONT_CHOICES = get_font_choices

# Name fragments that mark a CJK font; used to decide whether a font list
# needs a Latin fallback for the minus sign.
CJK_KEYWORDS = (
    "Song",
    "Hei",
    "Kai",
    "Fang",
    "PingFang",
    "Hiragino",
    "Noto",
    "WenQuanYi",
    "STH",
    "STS",
    "SC",
    "TC",
    "HK",
    "JP",
    "KR",
)


def clean_font_name(name: str) -> str:
    """Strip the display suffix: ``"Songti SC (宋体)"`` -> ``"Songti SC"``."""
    return name.split(" (")[0].strip()


def resolve_preset_name(name: str) -> str:
    """Map a legacy preset alias to its current key, or raise."""
    if name not in FONT_PRESETS and name in _PRESET_ALIASES:
        name = _PRESET_ALIASES[name]
    if name not in FONT_PRESETS:
        available = ", ".join(FONT_PRESETS.keys())
        raise MethodIncompatibility(f"Unknown preset '{name}'. Available: {available}")
    return name
