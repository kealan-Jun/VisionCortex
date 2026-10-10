"""Explicit replaceable application dependencies; one owner per legacy port."""

from __future__ import annotations

import time
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Any

from ..pipeline import EvidencePipeline
from ..config import load_config
from ..collection_catalog import get_collection
from ..collection_state import record_collection_state
from ..archive_catalog import lightweight_release_integrity
from ..input_preflight import preflight_manifest_inputs
from ..nas_recordings import scan_recordings
from ..storage import (
    fixed_archive_staging_paths,
    prepare_from_nas_index,
    promote_fixed_archive,
    read_current_release_pointer,
)
from ..web_access import append_access_audit, validate_web_access_configuration


@dataclass
class DependencyPorts:
    EvidencePipeline: Callable = EvidencePipeline
    load_config: Callable = load_config
    get_collection: Callable = get_collection
    record_collection_state: Callable = record_collection_state
    lightweight_release_integrity: Callable = lightweight_release_integrity
    preflight_manifest_inputs: Callable = preflight_manifest_inputs
    scan_recordings: Callable = scan_recordings
    fixed_archive_staging_paths: Callable = fixed_archive_staging_paths
    prepare_from_nas_index: Callable = prepare_from_nas_index
    promote_fixed_archive: Callable = promote_fixed_archive
    read_current_release_pointer: Callable = read_current_release_pointer
    append_access_audit: Callable = append_access_audit
    validate_web_access_configuration: Callable = validate_web_access_configuration
    datetime: Any = datetime
    time: Any = time


ports = DependencyPorts()
