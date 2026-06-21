import ast
from pathlib import Path

MATPLOTLIB_TEXT_METHODS = {
    "set_xlabel",
    "set_ylabel",
    "set_title",
    "suptitle",
    "xlabel",
    "ylabel",
    "title",
}


def _string_literals(node):
    values = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        values.append(node.value)
    elif isinstance(node, ast.JoinedStr):
        values.extend(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    else:
        for child in ast.iter_child_nodes(node):
            values.extend(_string_literals(child))
    return values


def test_matplotlib_text_literals_use_ascii_minus():
    """Avoid glyph warnings for CJK-capable fonts that lack U+2212."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "statspai"
    offenders = []

    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name not in MATPLOTLIB_TEXT_METHODS:
                continue

            for arg in [*node.args, *[kw.value for kw in node.keywords]]:
                for text in _string_literals(arg):
                    if "\u2212" not in text:
                        continue
                    offenders.append(
                        f"{path.relative_to(src_root)}:{node.lineno}: {text!r}"
                    )

    assert offenders == []
