import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_uses_isolated_hardware_supervisor():
    tree = ast.parse((ROOT / "tools/rtx4050_portable.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "hardware_preflight")
    assert any(isinstance(node, ast.ImportFrom) and node.module == "rtx4050_hardware"
               for node in ast.walk(function)), "hardware checks still run inside the desktop supervisor"
    assert not any(isinstance(node, ast.Import) and any(item.name == "torch" for item in node.names)
                   for node in ast.walk(function))


def test_source_update_entry_is_available():
    for name in ("tools/update_rtx4050_source.py", "deployment/rtx4050-windows/apply-update.py",
                 "deployment/rtx4050-windows/Update-From-Source.ps1"):
        assert (ROOT / name).is_file(), f"missing source update entry: {name}"
