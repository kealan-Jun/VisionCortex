"""Compatibility API for training, datasets and independent human evaluation.

Implementations live in training/; importing these APIs never loads model stacks.
"""
from .training.public_datasets import build_public_yolo_training_view as build_public_yolo_training_view
from .training.public_datasets import build_mapped_public_yolo_union as build_mapped_public_yolo_union
from .training.reviewed_dataset import build_yolo_training_dataset as build_yolo_training_dataset
from .training.run import train_yolo_model as train_yolo_model
from .training.evaluation import evaluate_yolo_model_on_human_truth as evaluate_yolo_model_on_human_truth

from .training import dataset_common as _common

def __getattr__(name):
    return getattr(_common, name)
