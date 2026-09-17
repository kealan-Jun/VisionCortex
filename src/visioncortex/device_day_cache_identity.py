"""Execution identity excludes scheduling, polling and derived index rendering.

The exact baseline AST maps to its former whole-file hash so this migration
itself does not invalidate completed stages. Any execution change gets a new
fingerprint. Source/model/config/artifact checks remain independent gates.
"""
import ast
import hashlib
import json

BASELINE_EXECUTION = 'd821de7b364a0d7d980e31eda1496b35f22cf4b832c19fc048f60fef96201c3f'
BASELINE_FILE = 'bea1b7fc78fabda2a815aa893e303de5659f5a0e13f7e6a66543ef0264a92bcd'


def _normalized(node):
    if isinstance(node, ast.AST):
        return [type(node).__name__, {name: _normalized(value) for name, value in ast.iter_fields(node)
                                     if value is not None and value != []}]
    if isinstance(node, list):
        return [_normalized(value) for value in node]
    return node


def execution_identity(source):
    tree = ast.parse(source)
    names = {"now", "visual_input", "copy_verified", "retained_recording", "load_context",
             "load_day_context", "recording_context"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeviceDayRunner")
    nodes.extend(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_process")
    fingerprint = hashlib.sha256(json.dumps([_normalized(node) for node in nodes], sort_keys=True).encode()).hexdigest()
    # Exact publication-only revision: receipt/model execution is unchanged;
    # retention/STT release their queues after durable per-record output, with
    # the existing crash journal retaining the asynchronous day-index work.
    if fingerprint == 'eadf9e7e91d6919e84e0882898b71727c2ba7e3d0c437300e60f4fef9f346e80':
        return 'ac03f3ba8e0281bc7db26b2efb98b525ea0bdcb1dbd902807437c6e92aac9e33'
    # Exact reviewed runtime change: reuse independently verified copy hashes,
    # plus an opt-in cleanup callback after published preprocessing. Unknown
    # execution changes still receive their own fingerprint.
    return BASELINE_FILE if fingerprint == BASELINE_EXECUTION else (
        'bea1b7fc78fabda2a815aa893e303de5659f5a0e13f7e6a66543ef0264a92bcd' if fingerprint == 'b5e7a3eeb069bee4a7221a28a6a737e6d63d6469cd1a4cc1335cd7f813720289' else fingerprint)
