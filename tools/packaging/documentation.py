"""Customer documentation selection and fixed-commit source filtering."""
from __future__ import annotations

from pathlib import Path
import re
import posixpath
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from tools.packaging.primitives import export_git_commit, relative_path, safe_path


DOCUMENTATION_ENTRYPOINTS = {
    "customer_package": ("README.md", "SOURCE-README.md"),
    "desktop_update": ("deployment/rtx4050-windows/README.md",),
    "portable_desktop": (
        "README.md", "docs/FEATURES-AND-OPERATIONS.zh-CN.md",
        "deployment/rtx4050-windows/README.md",
    ),
    "prepared_service": ("deployment/rtx3090ti-ubuntu/README.md",),
    "frozen_execution": ("deployment/rtx4060/README.md",),
    "ubuntu_offline": (
        "README.md", "deployment/rtx3050-ubuntu20/README.md",
        "docs/VisionCortex-RTX3050-离线部署与使用交付手册.md",
    ),
    "windows_offline": ("README.md", "deployment/rtx4090/README.md"),
}
PUBLIC_RELOCATION_RECIPE = "public-document-relocation/1"
_PRIVATE_PREFIXES = ("docs/history/", "docs/superpowers/", "release/freeze/")
_PRIVATE_NAMES = {"agents.md", "claude.md", "baseline.md", "validation.md",
                  "dual-repository-release-policy.md", "repository-branches.md",
                  "camera-view-bindings.md"}
# Report categories and relative names only, never the matching private value.
_PRIVATE_DOCUMENT_PATTERNS = {
    "repository_roles": re.compile(r"开发仓|稳定仓|个人仓库|公司仓库|development-only|stable-only", re.I),
    "private_home": re.compile(r"(?:/home/(?![<$])[^\s/]+/|[A-Z]:[\\/]Users[\\/][^\s/\\]+[\\/])", re.I),
    "private_workspace": re.compile(r"/(?:srv|mnt|media)/[^\s/]+/(?:VisionCortex\w*|LocalWorkspaces)(?:/|\b)", re.I),
    "device_address": re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    "internal_run": re.compile(r"\bexp_\d{8}_\d{6}\b|\bDEV-\d{3}\b", re.I),
    "historical_remote": re.compile(r"https?://[^\s)]+/blob/[0-9a-f]{40}/(?:docs/(?:history|20\d\d-|DEV-|WORK-)|BASELINE\.md)", re.I),
    "credential_value": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b"),
}


def private_document(name: str) -> bool:
    """Private guidance and execution journals are outside customer navigation."""
    path = relative_path(name)
    normalized = path.as_posix().casefold()
    return (path.name.casefold() in _PRIVATE_NAMES
            or normalized.startswith(_PRIVATE_PREFIXES)
            or bool(re.match(r"docs/(?:20\d\d-\d\d-\d\d[_-]|dev-\d|work-(?:handoff|progress)-)", normalized)))


def validate_customer_documents(contents: dict[str, bytes]) -> None:
    """Reject private navigation/content without echoing secrets or site values."""
    for name, raw in sorted(contents.items()):
        if private_document(name):
            raise ValueError(f"Customer documentation rejected (private_entry): {name}")
        if Path(name).suffix.casefold() != ".md":
            continue
        text = raw.decode("utf-8")
        for category, pattern in _PRIVATE_DOCUMENT_PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"Customer documentation rejected ({category}): {name}")


def local_references(text: str) -> list[str]:
    text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.S)
    markdown = re.findall(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text)
    html = re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", text)
    return [unquote(urlsplit(value.strip("<>")).path) for value in markdown + html
            if not urlsplit(value.strip("<>")).scheme and not urlsplit(value.strip("<>")).netloc
            and urlsplit(value.strip("<>")).path]


def documentation_resources(root: Path, purpose: str) -> tuple[str, ...]:
    """Fail on private, missing or redirected references; select referenced files."""
    root = root.resolve()
    queue = list(DOCUMENTATION_ENTRYPOINTS[purpose])
    seen: set[str] = set()
    while queue:
        name = queue.pop()
        relative_path(name)
        if name in seen:
            continue
        if private_document(name):
            raise ValueError(f"Customer documentation rejected (private_entry): {name}")
        path = safe_path(root, name, reject_links=True)
        if path.is_symlink() or not path.exists() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or redirected documentation resource: {name}")
        if path.is_dir():
            # Directory references are navigation only, never recursive data sweeps.
            continue
        seen.add(name)
        if path.suffix.casefold() != ".md":
            continue
        raw = path.read_bytes()
        validate_customer_documents({name: raw})
        for link in local_references(raw.decode("utf-8")):
            target = path.parent / link
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Documentation link escapes source root: {name}")
            for component in (target, *target.parents):
                if component == root:
                    break
                if component.is_symlink():
                    raise ValueError(f"Redirected documentation resource: {name}")
            queue.append(target.resolve().relative_to(root).as_posix())
    return tuple(sorted(seen))


def project_documentation(root: Path, commit: str, contents: dict[str, bytes],
                          mappings: dict[str, str] | None = None) -> tuple[dict[str, bytes], dict[str, dict]]:
    """Preserve identities while relocating public local links; never query a remote."""
    import hashlib
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Customer documentation requires an immutable commit SHA")
    validate_customer_documents(contents)
    mappings = mappings or {}
    projected, receipts = dict(contents), {}
    pattern = re.compile(r"(?P<prefix>!?\[[^\]]*\]\(\s*|(?:href|src)=[\"'])(?P<value><[^>]+>|[^\s)\"']+)")
    for name, raw in sorted(contents.items()):
        if Path(name).suffix.casefold() != ".md":
            continue
        destination = mappings.get(name, name)
        relative_path(destination)
        def replace(match):
            value = match.group("value")
            url = urlsplit(value.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                return match.group(0)
            source_target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path)))
            relative_path(source_target)
            if private_document(source_target):
                raise ValueError(f"Customer documentation rejected (private_entry): {name}")
            target = mappings.get(source_target, source_target)
            relative_path(target)
            relocated = posixpath.relpath(target, posixpath.dirname(destination) or ".")
            # Keep unchanged spelling when neither endpoint moves.
            if destination == name and target == source_target:
                return match.group(0)
            link = urlunsplit(("", "", quote(relocated, safe="/@"), url.query, url.fragment))
            if value.startswith("<"):
                link = "<" + link + ">"
            return match.group("prefix") + link
        # Fenced command examples are not navigation and retain their exact bytes.
        parts = re.split(r"(```.*?```|~~~.*?~~~)", raw.decode("utf-8"), flags=re.S)
        rendered = "".join(part if index % 2 else pattern.sub(replace, part)
                           for index, part in enumerate(parts)).encode("utf-8")
        if rendered != raw:
            projected[name] = rendered
            receipts[name] = {"recipe": PUBLIC_RELOCATION_RECIPE, "source_path": name,
                              "package_path": destination, "base_commit_context": commit,
                              "source_sha256": hashlib.sha256(raw).hexdigest(),
                              "projection_sha256": hashlib.sha256(rendered).hexdigest()}
    return projected, receipts


def customer_source_name(name: str, documents: set[str]) -> bool:
    """Select runtime/source contracts, excluding private guidance and journals."""
    if private_document(name):
        return False
    if name in documents:
        return True
    if Path(name).suffix.casefold() == ".md":
        return False
    return name.startswith(("src/", "configs/", "examples/", "tools/", "deployment/", "docs/contracts/")) or name in {
        "pyproject.toml", "LICENSE", "LICENSE.txt", "start-visioncortex.sh", "Start-VisionCortex.ps1",
        "Start-VisionCortex.bat", "Start-VisionCortex.command",
    }


def validate_source_snapshot(root: Path, snapshot: dict) -> None:
    """Verify customer guidance carried by first-party source receipt entries."""
    if any(receipt.get("recipe") != PUBLIC_RELOCATION_RECIPE
           for receipt in snapshot.get("documentation_projections", {}).values()):
        raise ValueError("Customer source rejected (historical_projection)")
    documents = {}
    for record in snapshot.get("source_files", []):
        name = record["path"]
        if private_document(name):
            raise ValueError(f"Customer source rejected (private_entry): {name}")
        if Path(name).suffix.casefold() == ".md":
            documents[name] = safe_path(root, record.get("package_path", name), reject_links=True).read_bytes()
    validate_customer_documents(documents)


def validate_customer_tree(root: Path, names) -> None:
    """Inspect first-party guidance only, without reading model/runtime payloads."""
    contents = {}
    for name in names:
        relative_path(name)
        if "/" not in name or name.startswith(("docs/", "deployment/", "configs/", "examples/", "tools/")):
            if private_document(name):
                raise ValueError(f"Customer source rejected (private_entry): {name}")
            if Path(name).suffix.casefold() == ".md":
                contents[name] = safe_path(root, name, reject_links=True).read_bytes()
    validate_customer_documents(contents)


def export_customer_source(repo: Path, commit: str, destination: Path, purpose: str) -> None:
    """Filter an owned immutable export, preserving selected Git bytes unchanged."""
    export_git_commit(repo, commit, destination)
    documents = set(documentation_resources(destination, purpose))
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError("Customer source cannot contain redirected files")
        if path.is_file():
            name = path.relative_to(destination).as_posix()
            if not customer_source_name(name, documents):
                path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    validate_customer_documents({name: (destination / name).read_bytes() for name in documents})
