from __future__ import annotations

from repo_paths import ROOT

import subprocess
import sys


def test_training_compatibility_imports_are_lazy_and_all_actual_source_is_traced():
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class RejectModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'ultralytics'}:
            raise RuntimeError('Optional model import on ordinary API import')
sys.meta_path.insert(0, RejectModels())
from visioncortex.yolo_training import train_yolo_model, evaluate_yolo_model_on_human_truth
from visioncortex.project_annotation_training import run_experiment, _CODE_AT_IMPORT
assert train_yolo_model.__module__ == 'visioncortex.training.run'
assert evaluate_yolo_model_on_human_truth.__module__ == 'visioncortex.training.evaluation'
assert run_experiment.__module__ == 'visioncortex.training.project.experiment'
root = Path(sys.argv[1]) / 'visioncortex' / 'training'
for path in root.rglob('*.py'):
    assert _CODE_AT_IMPORT[str(path.resolve())] == path.read_bytes(), str(path)
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")],
                   check=True, capture_output=True, timeout=20)
