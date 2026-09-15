"""Desktop child: accept credentials through stdin and emit a redacted receipt."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from rtx4050_portable import verify_package


def main(root: Path | None = None) -> int:
    root = root or Path(__file__).resolve().parents[1]
    key, stage = "", "package"
    started = stage_started = time.monotonic()
    timings = {}
    last_progress = float("-inf")

    def emit(current, **counts):
        print("VISIONCORTEX_PROVIDER_PROGRESS " + json.dumps({"stage": current, **counts}), flush=True)

    def package_progress(counts):
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress >= .5 or counts["checked_files"] == counts["total_files"]:
            emit("package", **counts)
            last_progress = now

    def next_stage(current):
        nonlocal stage, stage_started
        now = time.monotonic()
        timings[stage] = round(now - stage_started, 3)
        stage, stage_started = current, now
        emit(stage)

    try:
        value = json.loads(sys.stdin.readline(16384))
        key = value.get("api_key", "")
        emit("package")
        verify_package(root, progress=package_progress)
        next_stage("loading")
        from visioncortex.provider_connection import verify_connection
        next_stage("request")
        receipt = verify_connection(value["connection"], key, root / "Runtime/Temp")
    except Exception as error:
        message = str(error)
        if isinstance(key, str) and key:
            message = message.replace(key, "[REDACTED]")
        reason = {"package": "本地离线包校验未完成，请检查文件是否完整解压；尚未调用 AI 服务。",
                  "loading": "本地验证组件加载失败；尚未调用 AI 服务。",
                  "request": "AI 调用验证未完成，请检查接口与模型配置。"}[stage]
        receipt = {"status": "failed", "model_invocation": "NOT_PROVEN", "message": reason, "detail": message[:1000]}
    timings[stage] = round(time.monotonic() - stage_started, 3)
    receipt.update(verification_stage=stage, phase_seconds=timings,
                   verification_elapsed_seconds=round(time.monotonic() - started, 3))
    print("VISIONCORTEX_PROVIDER_RESULT " + json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0 if receipt["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
