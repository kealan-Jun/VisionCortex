"""Explicit callbacks between task orchestration and per-panel completion."""

from dataclasses import dataclass
from typing import Callable

from field_recognition.recognition import aliyun_vision
from field_recognition.recognition.config import RecognitionConfig
from field_recognition.recognition.vision_display_rules import configured_rules


@dataclass(frozen=True)
class ReadoutPipeline:
    save_document: Callable
    acquire_primary_slot: Callable
    recognition: RecognitionConfig | None = None

    def public_config(self):
        return aliyun_vision.public_config(self.recognition)

    def primary_photos(self):
        return aliyun_vision.primary_photos(self.recognition)

    def no_digits_seconds(self):
        return aliyun_vision.no_digits_seconds(self.recognition)

    def display_rules(self):
        if self.recognition is None:
            return configured_rules()
        return [rule.as_dict() for rule in self.recognition.display_rules]

    def read_panel(self, panel, *, display_rules=None):
        options = {}
        if display_rules:
            options["display_rules"] = display_rules
        if self.recognition is not None:
            options["config"] = self.recognition
        return aliyun_vision.read_panel(panel, **options)
