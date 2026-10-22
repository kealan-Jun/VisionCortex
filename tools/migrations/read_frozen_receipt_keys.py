"""Key-only local IPC endpoint, imported against a verified frozen producer.

This endpoint cannot execute a stage, invoke a model, or publish an artifact.
The publication caller separately validates every receipt and artifact byte.
"""

import json
from pathlib import Path
import sys


MAXIMUM_MESSAGE = 64 * 1024**2


def main():
    from visioncortex.config import load_config
    from visioncortex.ai_settings import apply_active
    from visioncortex.device_day import DeviceDayRunner

    config = apply_active(load_config(Path(sys.argv[1])))

    class KeyReader(DeviceDayRunner):
        # Context is an explicit source-bound input from the unchanged context
        # provider in the publishing process. Reading it there reuses that
        # process's NAS lease instead of deadlocking on another nested lease.
        def _context(self, layout, recording):
            if self.key_context is None:
                raise ValueError("Report receipt key requires its observed context")
            return self.key_context

    runner = KeyReader(config, backend=object())
    while True:
        line = sys.stdin.buffer.readline(MAXIMUM_MESSAGE + 1)
        if not line:
            break
        if len(line) > MAXIMUM_MESSAGE or not line.endswith(b"\n"):
            raise ValueError("Receipt owner request exceeds its bounded message")
        try:
            value = json.loads(line)
            if set(value) != {
                "stage",
                "record",
                "inputs",
                "binding",
                "context",
            } or value["stage"] not in {
                "retention",
                "vision",
                "stt",
                "understanding",
                "report",
            }:
                raise ValueError("Unsupported receipt owner operation")
            if value["context"] is not None and value["stage"] != "report":
                raise ValueError("Only report keys accept an observed context")
            runner.key_context = value["context"]
            key = runner._key(
                value["stage"],
                value["record"],
                value["inputs"],
                _binding=value["binding"],
            )
            result = {"key": key}
        except Exception as exc:
            result = {"error_type": type(exc).__name__}
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
