"""Private stdin/stdout inference protocol; no listening port or frame retention."""

import base64
import contextlib
import json
import re
import sys


def inference_device(value="cpu"):
    if value == "cpu":
        return value
    if not re.fullmatch(r"gpu:\d+", value):
        raise ValueError("Panel detector device must be cpu or gpu:<index>")
    return int(value.split(":")[1])


def main():
    wire = sys.stdout
    device = inference_device(sys.argv[4] if len(sys.argv) > 4 else "cpu")
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO, settings
        from field_recognition.recognition.panel_recovery import detect

        settings.update({"sync": False})
        torch.set_num_threads(2)
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable")
        model = YOLO(sys.argv[1])
        imgsz, confidence = int(sys.argv[2]), float(sys.argv[3])
        model.predict(np.zeros((800, 1280, 3), np.uint8), device=device, imgsz=imgsz, verbose=False)
    wire.write(json.dumps({"status": "ready", "names": model.names}) + "\n")
    wire.flush()
    for request in sys.stdin:
        try:
            payload = json.loads(request)
            raw = base64.b64decode(payload["image"], validate=True)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("invalid image")

            def infer(pixels, size, threshold):
                with contextlib.redirect_stdout(sys.stderr):
                    result = model.predict(
                        pixels, device=device, imgsz=size, conf=threshold, max_det=12, verbose=False
                    )[0]
                return {
                    "boxes": [
                        {"class_id": int(c), "confidence": float(s), "xyxy": b}
                        for b, c, s in zip(
                            result.boxes.xyxy.cpu().tolist(),
                            result.boxes.cls.cpu().tolist(),
                            result.boxes.conf.cpu().tolist(),
                        )
                    ],
                    "speed_ms": result.speed,
                }

            response = detect(
                image, infer, imgsz=imgsz, confidence=confidence, photo=payload.get("photo") is True
            )
        except Exception as exc:
            response = {"error": type(exc).__name__}
        wire.write(json.dumps(response) + "\n")
        wire.flush()


if __name__ == "__main__":
    main()
