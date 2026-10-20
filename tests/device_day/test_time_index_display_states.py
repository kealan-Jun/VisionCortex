"""The browser's three separate dimensions come from owning interval rows."""
from test_retained_video_subintervals import exact_case

from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_file_index import build_file_index

__all__ = ['exact_case']


def test_time_index_projects_interval_category_and_processing_without_promoting_candidate(exact_case):
    case = exact_case
    publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    case.config['collection_ingest'] = {'source_root': str(case.layout.root)}
    time_index = build_file_index(case.config, case.index, {})
    rows = {row['segment_id']: row for row in time_index['streams']['video'] if row.get('record_scope') == 'retention_interval'}
    for interval in case.candidate['retention_intervals']:
        entry = rows[interval['segment_id']]
        assert entry['activity_category'] == interval['activity_category']
        assert entry['processing_status'] == interval['processing_status']
        assert entry['video_value'] == interval['video_value']
        if interval['video_value'] == 'selected':
            assert entry['processing_status'] == 'completed' and entry['file']
        else:
            assert entry['file'] is None
    parent = next(row for row in time_index['streams']['video'] if row.get('record_scope') == 'preprocessing_candidate')
    assert parent['activity_category'] is None and parent['video_value'] is None
    assert parent['processing_status'] is None
