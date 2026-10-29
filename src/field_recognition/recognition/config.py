"""Immutable inference settings captured when a runtime is constructed."""

import math
import os
from dataclasses import dataclass, field
from collections.abc import Mapping

from field_recognition.recognition.vision_display_rules import DisplayRule, configured_rules


DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
NIANFENG_BASE_URL = "https://api.nianfeng.tech/v1"
DEFAULT_NO_DIGITS_SECONDS = 5.0


def parse_no_digits_seconds(value):
    """Keep the existing bounded runtime fallback for invalid environment values."""
    try:
        seconds = float(value)
        if math.isfinite(seconds) and 0.1 <= seconds <= 120:
            return seconds
    except (TypeError, ValueError):
        pass
    return DEFAULT_NO_DIGITS_SECONDS


@dataclass(frozen=True)
class RecognitionConfig:
    provider: str = "aliyun"
    enabled: bool = False
    api_key: str = field(default="", repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    photo_mode: str = "fallback"
    no_digits_seconds: float = DEFAULT_NO_DIGITS_SECONDS
    ocr_service_url: str | None = None
    display_rules: tuple[DisplayRule, ...] = ()
    enable_thinking: bool = True

    def __post_init__(self):
        if not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean")
        seconds = float(self.no_digits_seconds)
        if not math.isfinite(seconds) or not 0.1 <= seconds <= 120:
            raise ValueError("no_digits_seconds must be finite and between 0.1 and 120")
        object.__setattr__(self, "no_digits_seconds", seconds)
        if len(self.display_rules) > 1 or any(
            not isinstance(rule, DisplayRule) for rule in self.display_rules
        ):
            raise ValueError("display_rules must contain at most one validated DisplayRule")
        object.__setattr__(self, "display_rules", tuple(self.display_rules))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None):
        env = os.environ if environment is None else environment
        provider = env.get("FIELD_VISION_PROVIDER", "aliyun").strip().lower()
        nianfeng = provider == "nianfeng"
        thinking = env.get("FIELD_ALIYUN_ENABLE_THINKING", "1").strip().lower()
        if thinking not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError("FIELD_ALIYUN_ENABLE_THINKING must be 0/1, false/true or no/yes")
        return cls(
            provider=provider,
            enabled=env.get(
                "FIELD_VISION_ENABLED" if nianfeng else "FIELD_ALIYUN_FALLBACK_ENABLED", "0"
            ).lower()
            in {"1", "true", "yes"},
            api_key=env.get(
                "FIELD_VISION_API_KEY" if nianfeng else "DASHSCOPE_API_KEY", ""
            ).strip(),
            base_url=env.get("FIELD_VISION_BASE_URL", NIANFENG_BASE_URL)
            if nianfeng
            else env.get("FIELD_ALIYUN_BASE_URL", DEFAULT_BASE_URL),
            model=env.get("FIELD_VISION_MODEL", "")
            if nianfeng
            else env.get("FIELD_ALIYUN_MODEL", DEFAULT_MODEL),
            photo_mode=env.get("FIELD_VISION_PHOTO_MODE", "fallback").strip().lower(),
            enable_thinking=thinking in {"1", "true", "yes"},
            no_digits_seconds=parse_no_digits_seconds(
                env.get("FIELD_ALIYUN_NO_DIGITS_SECONDS", "5")
            ),
            ocr_service_url=env.get("FIELD_OCR_SERVICE_URL"),
            display_rules=tuple(
                DisplayRule(**rule)
                for rule in configured_rules(env.get("FIELD_VISION_DISPLAY_RULES", "[]"))
            ),
        )


def runtime_recognition_config(core):
    """Use injected runtime settings; plain legacy adapter mappings remain supported."""
    config = core.get("config")
    if config is None:
        return None
    if isinstance(config, RecognitionConfig):
        return config
    return config.recognition
