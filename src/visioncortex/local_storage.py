"""Prove local destinations without depending on a developer's mount names."""

from pathlib import Path
import sys

NETWORK_FILESYSTEMS = frozenset({"9p", "cifs", "fuse.sshfs", "nfs", "nfs4", "smb3", "sshfs"})


def mount_filesystem_type(path: Path) -> str:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("Cannot prove that the storage destination is local") from exc
    matches = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_value = fields[4]
            for encoded, character in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
                mount_value = mount_value.replace(encoded, character)
            mount_point = Path(mount_value)
            path.relative_to(mount_point)
            matches.append((len(mount_point.parts), fields[separator + 1].casefold()))
        except (ValueError, IndexError):
            continue
    if not matches:
        raise RuntimeError(f"Cannot determine destination filesystem type: {path}")
    return max(matches)[1]


def require_local_path(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    def reject_network_spelling(candidate):
        normalized = candidate.replace("\\", "/").casefold()
        parts = normalized.split("/")
        if normalized.startswith("//") or any(
            part == "nas" or part.endswith("-nas")
            or part in {"visioncortexexperimentarchive", "visioncortexexperimentcache"}
            for part in parts
        ):
            raise RuntimeError("A local non-NAS path is required")
    for candidate in (str(path), str(lexical)):
        reject_network_spelling(candidate)
    # Mount metadata is local. Reject a direct network destination before
    # resolving it, which can otherwise probe a Windows share or NAS directory.
    if sys.platform.startswith("linux") and mount_filesystem_type(lexical) in NETWORK_FILESYSTEMS:
        raise RuntimeError("A local non-NAS filesystem is required")
    resolved = path.expanduser().resolve()
    reject_network_spelling(str(resolved))
    if sys.platform.startswith("linux") and mount_filesystem_type(resolved) in NETWORK_FILESYSTEMS:
        raise RuntimeError(f"A local non-NAS filesystem is required: {resolved}")
    return resolved
