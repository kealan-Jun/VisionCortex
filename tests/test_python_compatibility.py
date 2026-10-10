import ast
from pathlib import Path


def test_source_tree_parses_with_declared_python_311_minimum():
    source_root = Path(__file__).parents[1] / "src"
    for path in sorted(source_root.rglob("*.py")):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 11),
        )
