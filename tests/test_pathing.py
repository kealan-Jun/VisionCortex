from pathlib import Path

import pytest

from visioncortex.archive import _relative
from visioncortex.pathing import archive_contains, archive_relative_posix


NORMAL_ROOT = r"\\192.168.66.149\video_database\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\A\run-001"
EXTENDED_ROOT = r"\\?\UNC\192.168.66.149\video_database\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\A\run-001"
RELATIVE_ARTIFACT = r"Key-Materials\Key-Frames\001_Test\01-Hand-Object-Contact\event-001\Aligned_First+Third.jpg"


@pytest.mark.parametrize(
    ("artifact_root", "archive_root"),
    [
        (NORMAL_ROOT, NORMAL_ROOT),
        (EXTENDED_ROOT, EXTENDED_ROOT),
        (EXTENDED_ROOT, NORMAL_ROOT),
        (NORMAL_ROOT, EXTENDED_ROOT),
    ],
)
def test_archive_relative_treats_normal_and_extended_unc_as_same_root(
    artifact_root, archive_root
):
    artifact = Path(artifact_root + "\\" + RELATIVE_ARTIFACT)
    expected = RELATIVE_ARTIFACT.replace("\\", "/")

    assert archive_relative_posix(artifact, Path(archive_root)) == expected
    assert _relative(artifact, Path(archive_root)) == expected


def test_archive_relative_is_case_insensitive_for_windows_paths():
    artifact = Path(EXTENDED_ROOT.upper() + "\\Key-Materials\\Event.jpg")

    assert archive_relative_posix(artifact, Path(NORMAL_ROOT)) == (
        "Key-Materials/Event.jpg"
    )


@pytest.mark.parametrize(
    "outside",
    [
        r"\\192.168.66.149\video_database\VisionCortexExperimentArchive\sibling\file.jpg",
        r"\\?\UNC\192.168.66.149\video_database\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\A\run-001-escape\file.jpg",
    ],
)
def test_archive_relative_rejects_sibling_prefixes(outside):
    assert not archive_contains(Path(outside), Path(NORMAL_ROOT))
    with pytest.raises(ValueError, match="outside archive root"):
        archive_relative_posix(Path(outside), Path(NORMAL_ROOT))
