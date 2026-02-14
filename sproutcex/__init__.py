from pathlib import Path

_readme = Path(__file__).resolve().parent.parent / "README.md"

if _readme.exists():
    text = _readme.read_text(encoding="utf-8")

    # Remove the first top-level heading
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        text = "\n".join(lines[1:]).lstrip()

    __doc__ = text
