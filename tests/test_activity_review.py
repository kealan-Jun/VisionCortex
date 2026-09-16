import copy
import json

import pytest

from visioncortex.activity_review import apply, assessment, binding, counts, save


def group():
    return {'group_id':'G1', 'group_uid':'stable-id', 'global_start_ms':0, 'global_end_ms':1000,
            'participating_views':['fp','tp'], 'archive_folder':'one', 'key_event_ids':['E1'],
            'model_understanding':{'steps':[{'current_step':'移动管架'}]}}


def test_equipment_organization_review_is_traceable_and_does_not_change_actions(tmp_path):
    original = group()
    previous = copy.deepcopy(original)
    path = save(tmp_path, original, binding(tmp_path, original),
                {'kind':'equipment_organization', 'workflow_relation':'standalone', 'reason':'用户回看确认仅整理仪器'})
    revised = apply(tmp_path, [original])
    assert original == previous
    assert revised[0]['model_understanding'] == previous['model_understanding']
    assert counts(revised) == {'experiments':0,'auxiliary_activities':1,'activity_records':1,'unclassified_activities':0}
    assert assessment(revised[0])['source'] == 'user_review'
    assert json.loads(path.read_text())['source_media_modified'] is False
    assert assessment(revised[0])['scope'].endswith('not_action_ground_truth')
    changed = copy.deepcopy(original)
    changed['global_end_ms'] = 2000
    assert assessment(apply(tmp_path, [changed])[0])['kind'] == 'uncertain'
    with pytest.raises(ValueError, match='已更新'):
        save(tmp_path, changed, binding(tmp_path, original), {'kind':'uncertain','workflow_relation':'unresolved','reason':'wait'})


def test_preparation_with_experiment_continuity_is_not_discarded():
    g = group()
    for relation, expected_auxiliary in [('part_of_experiment',False), ('unresolved',False), ('standalone',True)]:
        g['model_understanding']['activity_assessment'] = {
            'kind':'preparation', 'workflow_relation':relation, 'reason':'根据可见的前后承接判断'}
        assert assessment(g)['is_auxiliary'] is expected_auxiliary
        assert len(g['model_understanding']['steps']) == 1


def test_low_level_contact_alone_never_confirms_experiment_or_auxiliary():
    g = group()
    assert assessment(g)['kind'] == 'uncertain'
    assert assessment(g)['is_auxiliary'] is False
    assert counts([g])['unclassified_activities'] == 1


def test_model_contract_accepts_classification_without_claiming_action_quality():
    from visioncortex.mllm import GROUP_SYSTEM_PROMPT, _GroupResponse
    assert 'activity_assessment' in _GroupResponse.model_fields
    assert '不能确认实验成立' in GROUP_SYSTEM_PROMPT
