"""Reuse FR's panel localization, pixel refinement and conservative readout rules."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .components import enable_components


def make_runtime(model, bindings):
    enable_components()
    from field_recognition.recognition.ocr_runtime import create_model
    from field_recognition.recognition.panel_detector import PanelDetector
    from field_recognition.recognition.service import predict_panel, predict_readout
    root = Path(model['model_root'])
    os.environ['FIELD_OCR_DET_MODEL_DIR'] = str(root / model['det_directory'])
    os.environ['FIELD_OCR_REC_MODEL_DIR'] = str(root / model['rec_directory'])
    os.environ.pop('FIELD_OCR_SERVICE_URL', None)
    os.environ['FIELD_PANEL_DETECTOR_ENABLED'] = '1' if model.get('panel_manifest') else '0'
    if model.get('panel_manifest'):
        os.environ['FIELD_PANEL_MODEL_MANIFEST'] = str(root / model['panel_manifest'])
    ocr = create_model(model.get('device', 'cpu'), remote_config=None)
    detector = PanelDetector()
    registry = {b['instrument']['id']: b['instrument'] for b in bindings}
    runtime = dict(time=time, now=lambda: datetime.now(timezone.utc).isoformat(), ocr_lock=threading.RLock(),
                   load_ocr=lambda: ocr, ocr_state={'engine': 'PaddleOCR', 'status': 'ready'}, OCR_DEVICE=model.get('device', 'cpu'),
                   panel_detector=detector, get_instrument=lambda iid: registry.get(iid), receiver_camera=None)
    runtime['predict_panel'] = lambda *args, **kwargs: predict_panel(runtime, *args, **kwargs)
    runtime['predict_readout'] = lambda *args, **kwargs: predict_readout(runtime, *args, **kwargs)
    return runtime


def format_result(local, request):
    from field_recognition.measurements.reading_results import build_readings
    from field_recognition.measurements.measurement_records import display_fields
    if local.get('status') != 'completed':
        raise ValueError('Existing FR recognition pipeline did not complete')
    bindings = request.get('instrument_bindings', [])
    video = request.get('video_observation')
    document = dict(local, job_id=request['job_id'], capture_id=request['capture_id'], status='completed',
                    image_sha256=request.get('image_sha256') or hashlib.sha256(Path(request['media_path']).read_bytes()).hexdigest(),
                    camera_id=request.get('camera_id'), all_binding_snapshots=bindings,
                    request_trigger='video_stream' if video else 'voice_photo_directory')
    if video:
        document.update(video_observation=video, frame_metadata=request.get('frame_metadata', {}))
    else:
        document['external_photo'] = {'captured_at': request['occurred_at']}
    for region in document.get('panel_regions', []):
        match = next((b for b in bindings if b['instrument']['id'] == region['instrument_id']), None)
        if match:
            region.update(instrument=match['instrument'], binding_id=match['binding_id'], association_basis='registered_panel_mapping_at_capture' if video else 'same_photo_registered_equipment_tag')
    document['readings'] = build_readings(document)
    fields = []
    for field in display_fields(document):
        if field['value'] is not None and field.get('unit') and not field['issues']:
            fields.append(dict(field_key=field['name'], value=str(field['value']), unit=field['unit'], semantics='display_reading', valid=True, conflicts=[], association={'instrument_id': field['instrument_id']}))
    return dict(analysis_outcome='observations' if document['readings'] else 'no_numeric_readout', association_status='bound' if fields else 'unlocalized', normalized_fields=fields, field_document=document,
                component='fieldrecognition.service.predict_readout+measurement_records', raw_ocr=local, readings=document['readings'])


def read(request):
    enable_components()
    import cv2
    frame = cv2.imread(request['media_path'])
    if frame is None:
        raise ValueError('Source photograph cannot be decoded')
    runtime = make_runtime(request['model'], request.get('instrument_bindings', []))
    try:
        local = runtime['predict_readout'](frame, binding_snapshots=request.get('instrument_bindings', []))
    finally:
        runtime['panel_detector'].close()
    return format_result(local, request)
