"""retrocore が NumPy と Pillow 以外（標準ライブラリを除く）に依存せず、retro98 を import しないことを確認する。"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "retrocore"
ALLOWED = {"numpy", "PIL", "retrocore"}


def _top_level_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_retrocore_imports_only_numpy_pillow_stdlib():
    stdlib = set(sys.stdlib_module_names)
    bad = {}
    for path in sorted(CORE.glob("**/*.py")):
        extra = _top_level_imports(path) - ALLOWED - stdlib
        if extra:
            bad[path.name] = sorted(extra)
    assert not bad, f"retrocore に許可されていない依存があります: {bad}"


def test_import_retrocore_does_not_load_gui():
    code = (
        "import sys, retrocore, retrocore.analyzer, retrocore.optimize, retrocore.preview, "
        "retrocore.screenshot, retrocore.style, retrocore.sprite;"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('retro98', 'PySide6', 'scipy', 'cv2')];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"retrocore の import で読み込まれたモジュール: {r.stdout} {r.stderr}"
