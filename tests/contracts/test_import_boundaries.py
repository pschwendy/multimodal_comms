import ast
from pathlib import Path


def test_methods_never_import_benchmarks():
    root = Path(__file__).parents[2] / "src" / "multimodal_comms" / "methods"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(value and "multimodal_comms.benchmarks" in value for value in imports), path


def test_no_sys_path_mutation():
    root = Path(__file__).parents[2] / "src"
    for path in root.rglob("*.py"):
        assert "sys.path" not in path.read_text(), path
