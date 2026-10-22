"""Public archive formats preserve old evidence without regenerating HTML."""
from pathlib import Path

import pytest

from test_frozen_archive_layout import fixture as content_fixture
from visioncortex.application import archive_documents
from visioncortex.device_day_contract import DeviceDayLayout, artifact, verify_artifact, read_json
from visioncortex.device_day_reports import publish_day


def test_word_guide_replaces_readme_once_without_touching_capture(tmp_path):
    archive = tmp_path/'Archive'
    archive.mkdir()
    raw = archive/'2026-10-22_camera/MetaData/110101/rgb.mp4'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'owned original unchanged')
    (archive/'Readme.html').write_bytes(b'old archived overview')
    config = {'storage': {'archive_root': str(archive), 'local_runtime_root': str(tmp_path/'Runtime')},
              'device_day': {'archive_html_enabled': False}}
    result = archive_documents.publish_guide(config)
    assert result['status'] == 'published' and not (archive/'Readme.html').exists()
    assert (archive/'DataRetentionGuide.docx').read_bytes() == (Path(archive_documents.__file__).parents[1]/'templates/DataRetentionGuide.docx').read_bytes()
    before = (archive/'DataRetentionGuide.docx').stat().st_mtime_ns
    assert archive_documents.publish_guide(config) == result
    assert (archive/'DataRetentionGuide.docx').stat().st_mtime_ns == before
    assert list((archive/'.Evidence/DocumentHistory').glob('*/Readme.html'))[0].read_bytes() == b'old archived overview'
    assert raw.read_bytes() == b'owned original unchanged'


def test_json_report_keeps_timeline_and_old_html_receipt_readable(tmp_path):
    config,index = content_fixture(tmp_path)
    config['device_day']['archive_html_enabled'] = False
    segment = index['segments'][0]
    segment['activity_label'] = '未检出活动'
    layout = DeviceDayLayout(tmp_path,index['archive'][11:],segment['start_us'])
    old = layout.reports/'LaboratoryDailyReport.html'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'completed older HTML bytes')
    reference = artifact(layout.root,old)
    publish_day(config,layout,index)
    assert not old.exists() and verify_artifact(layout.root,reference)
    current = read_json(layout.reports/'LaboratoryDailyReport.json')
    assert current['archive'] == index['archive'] and current['全天记录']['全天时间线']
    publish_day(config,layout,index)
    assert not old.exists() and verify_artifact(layout.root,reference)


def test_interrupted_html_retirement_restores_original_receipt_alias(tmp_path,monkeypatch):
    layout = DeviceDayLayout(tmp_path,'camera',1792376825000000)
    old = layout.reports/'LaboratoryDailyReport.html'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'actual older report')
    reference = artifact(layout.root,old)
    actual = archive_documents._publish_aliases
    monkeypatch.setattr(archive_documents,'_publish_aliases',lambda *_a: (_ for _ in ()).throw(OSError('interrupted alias publication')))
    with pytest.raises(OSError,match='interrupted alias'):
        archive_documents.retire_day_html(layout)
    assert not old.exists()
    monkeypatch.setattr(archive_documents,'_publish_aliases',actual)
    archive_documents.retire_day_html(layout)
    assert verify_artifact(layout.root,reference)


def test_document_retirement_refuses_raw_and_unowned_files(tmp_path):
    raw=tmp_path/'MetaData/capture/rgb.mp4'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'raw')
    with pytest.raises(ValueError,match='known owned'):
        archive_documents._preserve(tmp_path,'MetaData/capture/rgb.mp4','MetaData')
    assert raw.read_bytes()==b'raw'
