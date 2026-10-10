"""Compatibility API for explicitly reviewed project experiments.

Project annotation evidence never substitutes for independent human ground truth.
"""
from .training.project.common import sha256 as sha256
from .training.project.common import canonical_sha as canonical_sha
from .training.project.common import write_json as write_json
from .training.project.trace import OptimizationTrace as OptimizationTrace
from .training.project.trace import optimization_trace_usage as optimization_trace_usage
from .training.project.class_head import transfer_class_outputs as transfer_class_outputs
from .training.project.class_head import restrict_to_class_outputs as restrict_to_class_outputs
from .training.project.class_head import freeze_normalization_statistics as freeze_normalization_statistics
from .training.project.class_head import frozen_state_sha256 as frozen_state_sha256
from .training.project.class_head import synchronize_frozen_ema as synchronize_frozen_ema
from .training.project.cohort import validate_export as validate_export
from .training.project.cohort import training_sampling_sources as training_sampling_sources
from .training.project.evaluation import project_evaluation as project_evaluation
from .training.project.experiment import run_experiment as run_experiment
from .training.project.recovery import recover_evaluated_experiment as recover_evaluated_experiment
from .training.project.source import _CODE_AT_IMPORT as _CODE_AT_IMPORT
