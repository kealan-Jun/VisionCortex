"""Compose shared camera, binding, measurement and archive components explicitly."""

from field_recognition.archive.archive_store import ArchiveStore
from field_recognition.bindings.instrument_ownership import install as install_ownership
from field_recognition.bindings.scene_binding import install as install_scenes
from field_recognition.capture.automation import install as install_automation
from field_recognition.capture.live_scan import install as install_live_scan
from field_recognition.capture.photo_watch import SavedPhotoWatcher
from field_recognition.capture.saved_photo import install as install_saved_photo
from field_recognition.capture.video_ocr import install as install_video_ocr
from field_recognition.measurements.evidence_timeline import install as install_evidence_timeline
from field_recognition.measurements.event_feed import install as install_event_feed
from field_recognition.measurements.photo_measurements import install as install_measurements
from field_recognition.measurements.result_reprocessing import install as install_reprocessing
from field_recognition.runtime.runtime_rpc import camera_component


def install_components(runtime):
    # The router collects compatibility handlers; workers never build an HTTP app.
    install_scenes(runtime)
    runtime["live_scanner"] = install_live_scan(runtime)
    install_saved_photo(runtime)
    runtime["saved_photo_watcher"] = camera_component(
        "watcher", runtime.config.camera_id, role=runtime.service_role, data_dir=runtime.data_dir
    ) or SavedPhotoWatcher(runtime)
    runtime["automatic_runner"] = install_automation(runtime)
    runtime["video_ocr"] = install_video_ocr(runtime)
    install_ownership(runtime)
    install_measurements(runtime)
    install_reprocessing(runtime)
    runtime["archive_store"] = ArchiveStore(runtime)
    install_evidence_timeline(runtime)
    install_event_feed(runtime)
