"""Keep application-owned Python/FFmpeg subprocesses out of console windows."""
import os
import subprocess
import sys

if sys.platform == "win32" and os.environ.get("VISIONCORTEX_DESKTOP_MODE") == "1":
    class HiddenPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = HiddenPopen
