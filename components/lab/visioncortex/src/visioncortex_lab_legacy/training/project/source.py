"""Import-time source snapshots for experiment evidence, without model imports."""
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = {
    *_PACKAGE.joinpath("training").rglob("*.py"),
    *_PACKAGE.joinpath("training_runtime").rglob("*.py"),
    *(_PACKAGE / name for name in (
        "project_annotation_training.py", "yolo_training.py", "yolo_evaluation.py",
        "project_ignore_training.py", "project_sampling.py", "project_training_evidence.py",
        "project_augmentation.py", "project_branch_loss.py",
    )),
}
_CODE_AT_IMPORT = {str(path.resolve()): path.read_bytes() for path in sorted(_SOURCE_PATHS)}
