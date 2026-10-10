"""Assemble a Windows portable runtime without executing Windows binaries."""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import zipfile

from packaging.tags import compatible_tags, cpython_tags, parse_tag
from packaging.markers import default_environment
from packaging.requirements import Requirement

from tools.packaging.build_rtx3050_offline_package import _copy_runtime_models
from tools.desktop.rtx4050_portable import sha256, safe_path, verify_package, write_json
from tools.packaging.documentation import customer_source_name, documentation_resources, project_documentation, validate_customer_documents, validate_customer_tree, validate_source_snapshot


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "deployment/rtx4050-windows"


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Expected a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for item in bundle.infolist():
            path = safe_path(destination, item.filename.rstrip("/"))
            if item.filename.casefold() in seen or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError("Duplicate member or link in ZIP")
            seen.add(item.filename.casefold())
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output)


def check_windows_binaries(site_packages: Path) -> list[dict]:
    distributions = []
    compatible = set(cpython_tags((3, 12), platforms=["win_amd64"]))
    compatible.update(compatible_tags((3, 12), interpreter="cp312", platforms=["win_amd64"]))
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        wheel = distribution.read_text("WHEEL") or ""
        tags = [line.split(": ", 1)[1] for line in wheel.splitlines() if line.startswith("Tag: ")]
        if not tags or not any(parse_tag(tag) & compatible for tag in tags):
            raise RuntimeError(f"Non-Windows wheel: {distribution.metadata['Name']}")
        distributions.append({"name": distribution.metadata["Name"], "version": distribution.version, "tags": tags})
    required = {"torch", "torchvision", "sam-2", "clip", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs", "transformers", "fastapi"}
    actual = {item["name"].lower().replace("_", "-") for item in distributions}
    if not required <= actual:
        raise RuntimeError(f"Missing dependencies: {sorted(required - actual)}")
    versions = {item["name"].lower().replace("_", "-"): item["version"] for item in distributions}
    target = default_environment()
    target.update(python_version="3.12", python_full_version="3.12.10", sys_platform="win32",
                  os_name="nt", platform_system="Windows", platform_machine="AMD64", extra="")
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(target):
                continue
            installed = versions.get(requirement.name.lower().replace("_", "-"))
            if installed is None or (requirement.specifier and not requirement.specifier.contains(installed, prereleases=True)):
                raise RuntimeError(f"Unsatisfied Windows dependency: {distribution.metadata['Name']} -> {requirement}")
    for path in site_packages.rglob("*"):
        if path.suffix.lower() in {".so", ".dylib"}:
            raise RuntimeError(f"Non-Windows native library: {path}")
        if path.suffix.lower() in {".dll", ".pyd", ".exe"}:
            with path.open("rb") as handle:
                header = handle.read(64)
                if header[:2] != b"MZ":
                    raise RuntimeError(f"Not a Windows binary: {path}")
                handle.seek(struct.unpack_from("<I", header, 60)[0])
                signature = handle.read(6)
            # pip ships launcher templates for other architectures as data.
            template = path.suffix.lower() == ".exe" and (
                path.parent.name == "distlib" or
                (path.parent.name == "setuptools" and path.stem.startswith(("cli", "gui")))
            )
            if signature != b"PE\0\0\x64\x86" and not template:
                raise RuntimeError(f"Not an AMD64 binary: {path}")
    return sorted(distributions, key=lambda item: item["name"].lower())


def copy_source(destination: Path, allow_working_tree: bool) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status and not allow_working_tree:
        raise RuntimeError("Dirty checkout: explicitly select --allow-working-tree for an uncommitted candidate snapshot")
    allowed = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "-z", "src", "configs", "examples", "tools"], cwd=ROOT).decode().split("\0")
    documents = set(documentation_resources(ROOT, "portable_desktop"))
    allowed += list(documents)
    document_contents = {name: (ROOT / name).read_bytes() for name in documents if Path(name).suffix == ".md"}
    projected, projections = project_documentation(
        ROOT, commit, document_contents,
        {"README.md": "SOURCE-README.md", "deployment/rtx4050-windows/README.md": "README.md"},
    )
    records = []
    document_mappings = {"README.md": "SOURCE-README.md", "deployment/rtx4050-windows/README.md": "README.md"}
    for relative in sorted(set(allowed)):
        if not relative or not customer_source_name(relative, documents):
            continue
        if relative not in documents and Path(relative).suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".csv", ".html", ".css", ".js", ".md", ".toml"}:
            raise RuntimeError(f"Unexpected source artifact: {relative}")
        path = safe_path(ROOT, relative)
        if not path.exists():
            continue  # Tracked working-tree deletion.
        before = sha256(path)
        if relative in document_contents and before != hashlib.sha256(document_contents[relative]).hexdigest():
            raise RuntimeError(f"Source changed during documentation selection: {relative}")
        target = document_mappings.get(relative, relative)
        copy_file(path, destination / target)
        expected = before
        if relative in projections:
            (destination / target).write_bytes(projected[relative])
            expected = projections[relative]["projection_sha256"]
        if expected != sha256(destination / target) or before != sha256(path):
            raise RuntimeError(f"Source changed during snapshot: {relative}")
        record = {"path": relative, "package_path": target,
                  "sha256": sha256(destination / target)}
        if relative in projections:
            record["source_sha256"] = before
            record["documentation_projection"] = projections[relative]
        records.append(record)
    return {"base_commit": commit, "working_tree_snapshot": bool(status),
            "source_files": records, "documentation_projections": projections,
            "stable_release_readiness": "NOT_PROVEN"}


def install_desktop(build: Path, output: Path, source: dict) -> None:
    extract_zip(build / "assets/electron-v44.2.0-win32-x64.zip", output)
    (output / "electron.exe").rename(output / "VisionCortex.exe")
    (output / "resources/default_app.asar").unlink(missing_ok=True)
    for name in ("Start-VisionCortex.bat", "Start-VisionCortex.ps1"):
        (output / name).unlink(missing_ok=True)
    mappings = {
        "tools/rtx4050_portable.py": "tools/rtx4050_portable.py",
        "tools/desktop_storage.py": "tools/desktop_storage.py",
        "tools/windows_desktop_lifecycle.py": "tools/windows_desktop_lifecycle.py",
        "tools/verify_mllm_connection.py": "tools/verify_mllm_connection.py",
        "configs/mllm-providers.json": "configs/mllm-providers.json",
        "src/visioncortex/mllm.py": "src/visioncortex/mllm.py",
        "src/visioncortex/mllm_provider.py": "src/visioncortex/mllm_provider.py",
        "src/visioncortex/provider_connection.py": "src/visioncortex/provider_connection.py",
        "src/visioncortex/provider_credentials.py": "src/visioncortex/provider_credentials.py",
        "deployment/rtx4050-windows/README.md": "README.md",
        "deployment/rtx4050-windows/assets-lock.json": "receipts/assets-lock.json",
        "deployment/rtx4050-windows/desktop/sitecustomize.py": "python/Lib/sitecustomize.py",
    }
    # Thin entry points depend on the organized implementation packages.
    for subtree in (ROOT / "tools/desktop", ROOT / "tools/diagnostics", ROOT / "tools/packaging"):
        for path in subtree.glob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            mappings[relative] = relative
    mappings["tools/__init__.py"] = "tools/__init__.py"
    for name in ("package.json", "main.cjs", "controller.cjs", "connection.cjs", "storage.cjs", "preload.cjs", "setup.html",
                 "setup.css", "setup.js", "ensure-vc-runtime.ps1"):
        mappings[f"deployment/rtx4050-windows/desktop/{name}"] = f"resources/app/{name}"
    records = [item for item in source["source_files"] if item["path"] not in mappings]
    for relative, target in mappings.items():
        path = ROOT / relative
        before = sha256(path)
        copy_file(path, output / target)
        record = {"path": relative, "package_path": target, "sha256": before}
        if path.suffix.casefold() == ".md":
            projected, receipts = project_documentation(ROOT, source["base_commit"], {relative: path.read_bytes()},
                                                       {"README.md": "SOURCE-README.md", **mappings})
            source_digest = receipts.get(relative, {}).get("source_sha256", hashlib.sha256(projected[relative]).hexdigest())
            if source_digest != before:
                raise RuntimeError(f"Source changed during documentation selection: {relative}")
            (output / target).write_bytes(projected[relative])
            record["sha256"] = sha256(output / target)
            if relative in receipts:
                record.update(source_sha256=before, documentation_projection=receipts[relative])
                source.setdefault("documentation_projections", {})[relative] = receipts[relative]
        if record["sha256"] != sha256(output / target) or before != sha256(path):
            raise RuntimeError(f"Source changed during snapshot: {relative}")
        records.append(record)
    source["source_files"] = sorted(records, key=lambda item: item["path"])
    source["working_tree_snapshot"] = True
    (output / "python/python312._pth").write_text(
        "Lib\npython312.zip\n.\nLib/site-packages\n../src\n../tools\nimport site\n", encoding="utf-8")
    write_json(output / "receipts/source-snapshot.json", source)


def seal_package(output: Path) -> Path:
    validate_customer_tree(output, (p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()))
    documentation_resources(output, "customer_package")
    validate_source_snapshot(output, json.loads((output / "receipts/source-snapshot.json").read_bytes()))
    records = []
    seen = set()
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == output / "SHA256SUMS.json":
            continue
        name = path.relative_to(output).as_posix()
        if name.casefold() in seen:
            raise RuntimeError(f"Windows case collision: {name}")
        seen.add(name.casefold())
        records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(output / "SHA256SUMS.json", {"file_count": len(records), "total_bytes": sum(r["size_bytes"] for r in records), "files": records})
    verify_package(output)
    archive = output.with_suffix(".zip")
    partial = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    with zipfile.ZipFile(partial) as bundle:
        bad = bundle.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
    partial.rename(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def desktop_from_base(build: Path, output: Path, base: Path, refresh_source: bool = False) -> Path:
    """Preserve a verified CV source/runtime snapshot while replacing its entry point."""
    manifest = verify_package(base)
    validate_customer_tree(base, (item["path"] for item in manifest["files"]))
    source = json.loads((base / "receipts/source-snapshot.json").read_text())
    validate_source_snapshot(base, source)
    output.mkdir(parents=True)
    for item in manifest["files"]:
        copy_file(safe_path(base, item["path"]), safe_path(output, item["path"]))
    # Verify the copied base before changing any files; never copy mutable Runtime.
    copy_file(base / "SHA256SUMS.json", output / "SHA256SUMS.json")
    verify_package(output)
    if refresh_source:
        for item in source["source_files"]:
            if item["path"].startswith(("src/", "configs/", "examples/", "tools/", "deployment/", "docs/")) or Path(item["path"]).suffix.casefold() == ".md":
                safe_path(output, item.get("package_path", item["path"])).unlink()
        source = copy_source(output, allow_working_tree=True)
    source["desktop_base_manifest_sha256"] = sha256(base / "SHA256SUMS.json")
    install_desktop(build, output, source)
    metadata = json.loads((output / "BUNDLE-METADATA.json").read_text())
    metadata.update(entry_point="VisionCortex.exe", desktop_runtime="Electron 44.2.0 x64",
                    interface="embedded_existing_web_application", process_lifecycle="owned_windows_job",
                    source_base_commit=source["base_commit"], working_tree_snapshot=source["working_tree_snapshot"],
                    analysis_network="User-selected vision provider; verified with a live multi-image request in setup",
                    ai_providers=["volcengine", "aliyun", "zhipu", "custom"],
                    output_location="User-selected local folder or connected network share; writable probe required",
                    nas_discovery="Existing Windows mapped network drives only; no recursive media or subnet scan",
                    source_refreshed=refresh_source, desktop_base_manifest_sha256=source["desktop_base_manifest_sha256"])
    write_json(output / "BUNDLE-METADATA.json", metadata)
    return seal_package(output)


def connection_update(base: Path, output: Path) -> Path:
    """Build a small, version-bound connection repair without copying models or Runtime."""
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Refusing to overwrite an existing update")
    if output == ROOT or ROOT in output.parents:
        raise RuntimeError("Update artifacts must be built outside the checkout")
    original = json.loads((base / "SHA256SUMS.json").read_text(encoding="utf-8"))
    indexed = {item["path"]: item for item in original["files"]}
    validate_customer_tree(base, indexed)
    mappings = {"tools/rtx4050_portable.py": "tools/rtx4050_portable.py",
                "tools/verify_mllm_connection.py": "tools/verify_mllm_connection.py",
                "deployment/rtx4050-windows/README.md": "README.md"}
    # The verified repair carries the actual changed functions, not only shims.
    for relative in ("tools/__init__.py", "tools/desktop/__init__.py", "tools/diagnostics/__init__.py",
                     "tools/desktop/rtx4050_portable.py",
                     "tools/diagnostics/verify_mllm_connection.py", "tools/packaging/primitives.py"):
        mappings[relative] = relative
    for path in (ROOT / "tools/desktop").glob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        mappings[relative] = relative
    for name in ("connection.cjs", "main.cjs", "setup.js"):
        mappings[f"deployment/rtx4050-windows/desktop/{name}"] = f"resources/app/{name}"
    documents = set(documentation_resources(ROOT, "desktop_update"))
    for name in documents:
        mappings.setdefault(name, "SOURCE-README.md" if name == "README.md" else name)
    projected, projections = project_documentation(
        ROOT, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        {name: (ROOT / name).read_bytes() for name in documents}, mappings)
    # Only these files are read; unchanged runtime/model integrity remains a startup gate.
    for name in [*mappings.values(), "receipts/source-snapshot.json", "BUNDLE-METADATA.json", "python/python.exe"]:
        path = safe_path(base, name)
        if name not in indexed:
            if path.exists():
                raise RuntimeError(f"Untracked update target exists: {name}")
            continue
        if sha256(path) != indexed[name]["sha256"]:
            raise RuntimeError(f"Base file checksum mismatch: {name}")
    source = json.loads((base / "receipts/source-snapshot.json").read_text(encoding="utf-8"))
    validate_source_snapshot(base, source)
    metadata = json.loads((base / "BUNDLE-METADATA.json").read_text(encoding="utf-8"))
    payload = output / "payload"
    records = []
    for relative, target in mappings.items():
        before = sha256(ROOT / relative)
        copy_file(ROOT / relative, payload / target)
        if before != sha256(payload / target) or before != sha256(ROOT / relative):
            raise RuntimeError(f"Source changed during update snapshot: {relative}")
        record = {"path": relative, "package_path": target, "sha256": before}
        if relative in projected:
            source_digest = projections.get(relative, {}).get("source_sha256", hashlib.sha256(projected[relative]).hexdigest())
            if source_digest != before:
                raise RuntimeError(f"Source changed during documentation selection: {relative}")
            (payload / target).write_bytes(projected[relative])
            record["sha256"] = sha256(payload / target)
        if relative in projections:
            record.update(source_sha256=before, documentation_projection=projections[relative])
        records.append(record)
    validate_customer_documents({name: projected[name] for name in documents})
    source["documentation_projections"] = {name: receipt for name, receipt in source.get("documentation_projections", {}).items() if name not in mappings}
    source["documentation_projections"].update(projections)
    source["source_files"] = sorted([r for r in source["source_files"] if r["path"] not in mappings] + records,
                                     key=lambda item: item["path"])
    identity = {"kind": "desktop_connection_phase_timeouts", "base_manifest_sha256": sha256(base / "SHA256SUMS.json"),
                "working_tree_snapshot": True, "windows_runtime": "NOT_PROVEN", "stable_release_readiness": "NOT_PROVEN"}
    source["connection_update"] = metadata["connection_update"] = identity
    source["working_tree_snapshot"] = metadata["working_tree_snapshot"] = True
    write_json(payload / "receipts/source-snapshot.json", source)
    write_json(payload / "BUNDLE-METADATA.json", metadata)
    changed = [*mappings.values(), "receipts/source-snapshot.json", "BUNDLE-METADATA.json"]
    manifest = json.loads(json.dumps(original))
    for item in manifest["files"]:
        if item["path"] in changed:
            item.update(sha256=sha256(payload / item["path"]), size_bytes=(payload / item["path"]).stat().st_size)
    for name in changed:
        if name not in indexed:
            manifest["files"].append({"path": name, "sha256": sha256(payload / name),
                                      "size_bytes": (payload / name).stat().st_size})
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(item["size_bytes"] for item in manifest["files"])
    write_json(payload / "SHA256SUMS.json", manifest)
    spec = {**identity, "base_package_name": base.name,
            "updated_manifest_sha256": sha256(payload / "SHA256SUMS.json"),
            "python_sha256": indexed["python/python.exe"]["sha256"], "files": []}
    for name in [*changed, "SHA256SUMS.json"]:
        spec["files"].append({"path": name, "before_sha256": sha256(base / name) if (base / name).is_file() else None,
                               "after_sha256": sha256(payload / name)})
    write_json(output / "update.json", spec)
    for name in ("apply-update.py", "Apply-Update.ps1"):
        copy_file(DEPLOYMENT / name, output / name)
    (output / "Apply-Update.cmd").write_bytes(
        b'@echo off\r\n"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0Apply-Update.ps1"\r\npause\r\n')
    (output / "使用说明.txt").write_text(
        f"VisionCortex 连接验证修复候选\n适用原包：{base.name}\n\n"
        "1. 等待分析任务结束，关闭 VisionCortex。\n"
        "2. 将本修复包完整解压到独立目录，双击 Apply-Update.cmd。\n"
        "3. 选择包含 VisionCortex.exe 的原应用目录。版本不匹配时会拒绝更新。\n"
        "4. 提示 Update applied 后，重新打开原目录的 VisionCortex.exe 并验证连接。\n\n"
        "不需要重新下载模型，不修改 Runtime 中的数据、引擎和已加密密钥。\n"
        "旧文件备份位于原应用 Runtime/Updates/connection-*/before；更新异常会尝试自动恢复。\n"
        "本修复分开本地校验与云端计时，并显示阶段进度、保存脱敏诊断。未跳过文件完整性或真实多图调用校验。\n"
        "如果仍有问题，保留 Runtime/Logs/connection-verification.jsonl 和窗口报错，不发送 Key。\n"
        "本包仅修复连接流程，保留原包其余源码与模型。Windows 4050 实机首次启动、阿里云真实请求及完整实验分析尚未验收（NOT_PROVEN），不是稳定发布。\n",
        encoding="utf-8-sig")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                bundle.write(file, (Path(output.name) / file.relative_to(output)).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise RuntimeError("Update ZIP CRC failure")
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def assemble(build: Path, output: Path, models: Path | None, allow_working_tree: bool, base_package: Path | None = None, refresh_source: bool = False) -> Path:
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Refusing to overwrite an existing package")
    if ROOT == output or ROOT in output.parents:
        raise RuntimeError("Runtime packages must be built outside the Git checkout")
    assets = json.loads((DEPLOYMENT / "assets-lock.json").read_text())
    for item in assets["assets"]:
        path = safe_path(build / "assets", item["file"])
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Asset hash mismatch: {item['file']}")
    if base_package:
        if not allow_working_tree:
            raise RuntimeError("Desktop source changes require --allow-working-tree")
        try:
            return desktop_from_base(build, output, base_package, refresh_source)
        except Exception:
            if output.exists():
                (output / "PACKAGE-BUILD-FAILED.txt").write_text("Incomplete build; not a usable delivery.\n")
            raise
    if models is None:
        raise ValueError("A new offline runtime requires an explicit verified model root")
    packages = check_windows_binaries(build / "site-packages")
    output.mkdir(parents=True)
    try:
        source = copy_source(output, allow_working_tree)
        extract_zip(build / "assets/python-3.12.10-embed-amd64.zip", output / "python")
        # Torch inspects stdlib source. Include the matching official Lib tree,
        # retaining the embedded DLLs and its complete precompiled stdlib ZIP.
        with tarfile.open(build / "assets/Python-3.12.10.tgz") as archive:
            for item in archive.getmembers():
                prefix = "Python-3.12.10/Lib/"
                if not item.name.startswith(prefix) or not item.isfile():
                    continue
                relative = item.name[len(prefix):]
                if relative.startswith(("test/", "idlelib/", "tkinter/", "ensurepip/", "site-packages/")):
                    continue
                destination = safe_path(output / "python/Lib", relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as handle, destination.open("wb") as target:
                    shutil.copyfileobj(handle, target)
        shutil.copytree(build / "site-packages", output / "python/Lib/site-packages",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Cross-install console entry points are unused. Preserve nested bin
        # directories, especially NVIDIA's CUDA runtime DLL directory.
        entry_points = output / "python/Lib/site-packages/bin"
        if entry_points.is_dir():
            shutil.rmtree(entry_points)
        (output / "python/python312._pth").write_text(
            "Lib\npython312.zip\n.\nLib/site-packages\n../src\nimport site\n", encoding="utf-8")
        # Isolated embedded Python ignores host PYTHONPATH and user site packages.
        extract_zip(build / "assets/ffmpeg-7.0.2-essentials_build.zip", output / "vendor/ffmpeg-extracted")
        extracted = output / "vendor/ffmpeg-extracted"
        ffmpeg = next(extracted.glob("*/bin/ffmpeg.exe"))
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            copy_file(ffmpeg.parent / name, output / "vendor/ffmpeg/bin" / name)
        for name in ("LICENSE", "README.txt"):
            copy_file(ffmpeg.parent.parent / name, output / "vendor/ffmpeg" / name)
        shutil.rmtree(extracted)
        copy_file(build / "assets/vc_redist.x64.exe", output / "vendor/vc_redist.x64.exe")
        _copy_runtime_models(output, models)
        copy_file(build / "windows-resolved.txt", output / "receipts/windows-resolved.txt")
        copy_file(DEPLOYMENT / "assets-lock.json", output / "receipts/assets-lock.json")
        write_json(output / "receipts/python-distributions.json", {"packages": packages})
        install_desktop(build, output, source)
        write_json(output / "BUNDLE-METADATA.json", {
            "product": "VisionCortex", "target": "Windows 10 x64 / RTX 4050 Laptop 6GB",
            "source_base_commit": source["base_commit"],
            "working_tree_snapshot": source["working_tree_snapshot"],
            "installation": "offline_preassembled_embedded_python",
            "entry_point": "VisionCortex.exe", "desktop_runtime": "Electron 44.2.0 x64",
            "interface": "embedded_existing_web_application", "process_lifecycle": "owned_windows_job",
            "model_engine_policy": "build_and_measure_on_target_gpu",
            "analysis_network": "User-selected vision provider requires network and a user-provided key",
            "ai_providers": ["volcengine", "aliyun", "zhipu", "custom"],
            "windows_runtime": "NOT_PROVEN", "real_video_quality": "NOT_PROVEN",
            "stable_release_readiness": "NOT_PROVEN",
        })
        return seal_package(output)
    except Exception:
        (output / "PACKAGE-BUILD-FAILED.txt").write_text("Incomplete build; not a usable delivery.\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, help="Explicit verified runtime asset root; required for a new offline runtime")
    parser.add_argument("--allow-working-tree", action="store_true")
    parser.add_argument("--base-package", type=Path, help="Keep the verified source/runtime of an existing package")
    parser.add_argument("--refresh-source", action="store_true", help="Refresh application source while reusing verified models/runtime")
    parser.add_argument("--connection-update-from", type=Path, help="Build a small connection repair bound to this desktop package")
    args = parser.parse_args()
    if args.connection_update_from:
        if not args.allow_working_tree or args.base_package or args.refresh_source:
            parser.error("Connection updates require --allow-working-tree and cannot refresh the full source")
        archive = connection_update(args.connection_update_from.resolve(), args.output.resolve())
        print(json.dumps({"zip": str(archive), "bytes": archive.stat().st_size, "sha256": sha256(archive)}, indent=2))
        return
    if not args.build_root:
        parser.error("--build-root is required for a full package")
    if not args.base_package and not args.models_root:
        parser.error("--models-root is required for a new offline runtime")
    archive = assemble(args.build_root.resolve(), args.output.resolve(), args.models_root.resolve() if args.models_root else None,
                       args.allow_working_tree, args.base_package.resolve() if args.base_package else None, args.refresh_source)
    print(json.dumps({"zip": str(archive), "bytes": archive.stat().st_size, "sha256": sha256(archive)}, indent=2))


if __name__ == "__main__":
    main()
