"""Leaf-name compatibility only; original QR/bindings and unknown time survive."""
import re
from types import SimpleNamespace
from uuid import uuid4

from visioncortex.lab.timeline_native_artifacts import artifact_documents

from .test_timeline_native_artifacts import NoTaskDB


def test_new_fr_style_display_name_preserves_parent_clock_unknown_and_original_qr_history():
    aid,mid,capture,printed=str(uuid4()),str(uuid4()),str(uuid4()),str(uuid4())
    payload=dict(raw_tokens=[dict(text='550 990')],fields=[],qr_payload=printed,qr_hash='c'*64,
        historical_binding=dict(instrument_id=str(uuid4()),wearer_id=None,status='original_unverified'))
    activity=dict(id=aid,source_type='ocr',kind='reading',text_content='550 990',start_at=None,end_at=None,
        payload=payload,model_revision='original-pinned',evidence=[dict(media_id=mid,kind='photo',start_seconds='0',end_seconds='0')])
    parent='2026-10-29_fixture-camera/Comment/OCR/VoicePhotoReadings/'
    sources={mid:dict(capture_id=capture,source_sha256='a'*64,kind='photo',source_relative_path='MetaData/source.jpg',
        source_reported_at='2026-09-16T14:10:32.123+08:00',source_kind='customer_capture',
        time_basis=dict(basis='filename',clock_sync_valid=False,uncertainty_us=0))}
    docs,refs=artifact_documents(SimpleNamespace(db=NoTaskDB(),customer=uuid4()),archive=parent.split('/Comment/')[0],
        camera='fixture-camera',activities=[activity],media_sources=sources,rules=dict(revision='fixture',sha256='b'*64))
    path=refs[0]['evidence'];assert path.startswith(parent)
    leaf=path[len(parent):].split('/')[0]
    assert re.fullmatch(r'14-10-32\.123_[A-Za-z0-9_-]{1,100}',leaf)
    assert 'timeline-' not in leaf and aid in leaf and printed not in leaf
    evidence=docs[path]
    assert evidence['event_at'] is None and evidence['capture_time_status']=='unknown_or_uncalibrated'
    assert evidence['folder_time_basis']=='source_reported_display_only_uncalibrated'
    assert evidence['observations'][0]['activity']['payload']==payload
    assert evidence['source_refs'][mid]['capture_id']==capture
    assert refs[0]['naming']['legacy_candidate_is_not_an_installed_alias']
    assert '/UnknownTime_timeline-' in refs[0]['naming']['legacy_style_path_candidate']
