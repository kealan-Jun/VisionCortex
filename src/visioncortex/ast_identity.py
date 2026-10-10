"""Portable serialization for receipts authored with Python 3.12's AST."""
import ast
from copy import deepcopy


def dump_python312(node: ast.AST) -> str:
    """Match the established receipt recipe on 3.11 without accepting new code.

    Python 3.12 added an empty ``type_params`` field to non-generic definitions.
    Supplying that field on 3.11 fixes representation drift; all actual fields
    and their values remain bound, including future or nonempty type parameters.
    """
    node = deepcopy(node)
    for definition in ast.walk(node):
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if "type_params" not in definition._fields:
                definition._fields = (*definition._fields, "type_params")
                definition.type_params = []
    return ast.dump(node, include_attributes=False)
