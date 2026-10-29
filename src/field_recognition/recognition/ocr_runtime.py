"""Explicit, single-device PaddleOCR configuration; GPU errors never select CPU."""

import os
import re
from pathlib import Path


class OcrDeviceUnavailable(RuntimeError):
    pass


class OcrModelUnavailable(RuntimeError):
    pass


def local_model_options():
    """Use complete local weights directly, bypassing PaddleX host discovery."""
    cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(Path.home() / ".paddlex")))
    options = {}
    for kind, suffix in [("detection", "det"), ("recognition", "rec")]:
        configured = os.environ.get(f"FIELD_OCR_{suffix.upper()}_MODEL_DIR")
        path = (
            Path(configured).expanduser()
            if configured
            else cache / "official_models" / f"PP-OCRv5_mobile_{suffix}"
        )
        complete = all(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in ("inference.json", "inference.pdiparams", "inference.yml")
        )
        if complete:
            options[f"text_{kind}_model_dir"] = str(path.resolve())
        elif configured:
            raise OcrModelUnavailable(f"FIELD_OCR_{suffix.upper()}_MODEL_DIR 缺少完整模型文件")
    return options


def configured_device():
    device = os.environ.get("FIELD_OCR_DEVICE", "cpu").strip().lower()
    if not re.fullmatch(r"cpu|gpu:\d+", device):
        raise ValueError("FIELD_OCR_DEVICE must be cpu or gpu:<index>, for example gpu:0")
    return device


_ENVIRONMENT = object()


def create_model(device, *, remote_config=_ENVIRONMENT):
    # Unqualified calls retain the legacy environment adapter. A runtime passes
    # its captured configuration, including explicit None for local inference.
    if remote_config is _ENVIRONMENT and os.environ.get("FIELD_OCR_SERVICE_URL"):
        from field_recognition.recognition.remote_ocr import RemoteOcr

        return RemoteOcr()
    if remote_config is not _ENVIRONMENT and remote_config is not None:
        from field_recognition.recognition.remote_ocr import RemoteOcr

        return RemoteOcr(remote_config)
    options = local_model_options()
    if device.startswith("gpu:"):
        # Set before Paddle is imported. Allocate as needed alongside other GPU services.
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        import paddle

        if not paddle.is_compiled_with_cuda():
            raise OcrDeviceUnavailable(
                "GPU OCR requires paddlepaddle-gpu in this project environment"
            )
        index = int(device.split(":")[1])
        if index >= paddle.device.cuda.device_count():
            raise OcrDeviceUnavailable(f"Configured OCR device {device} is not visible to Paddle")
        paddle.set_device(device)
    from paddleocr import PaddleOCR

    return PaddleOCR(
        device=device,
        cpu_threads=2,
        enable_mkldnn=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        **options,
    )
