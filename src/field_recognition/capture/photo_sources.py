"""Upload path grammar and stable photo identities across the MetaData migration."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re


@dataclass(frozen=True)
class PhotoLocation:
    camera: str
    day: str
    moment: str
    filename: str

    @property
    def identity(self):
        return f"{self.camera}/{self.day}/{self.moment}/{self.filename}"

    @property
    def metadata_path(self):
        return f"{self.day}_{self.camera}/MetaData/{self.moment}/{self.filename}"


def photo_location(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) != 4 or any(p in {".", ".."} for p in parts):
        raise ValueError("Invalid photo directory")
    filename = parts[3]
    match = re.fullmatch(r"(\d{8})_(\d{6})(?:_\d+)?\.(?:jpg|jpeg|png)", filename, re.I)
    if not match:
        raise ValueError("Invalid photo filename")
    stamp = datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S")
    if parts[1] == "MetaData":
        day, separator, camera = parts[0].partition("_")
        if not separator:
            raise ValueError("Missing camera identity")
    else:
        camera, day = parts[:2]
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", camera)
        or day != stamp.strftime("%Y-%m-%d")
        or parts[2] != stamp.strftime("%H-%M-%S")
    ):
        raise ValueError("Photo directory and timestamp disagree")
    return PhotoLocation(camera, day, parts[2], filename)
