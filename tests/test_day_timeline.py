from copy import deepcopy
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from visioncortex.device_day_timeline import build_timeline, day_bounds, install_routes, link
from visioncortex.device_day_cross_view import associate
from visioncortex.device_day_schedule import scheduling_record


DAY = '2026-09-09'
BASE = day_bounds(DAY)[0]


def segment(name, start, end, activity='active'):
    return {'segment_id': name, 'recording_id': name, 'start_us': BASE+start*1000000,
            'end_us': BASE+end*1000000, 'start_ms': 0, 'end_ms': (end-start)*1000,
            'activity': activity, 'key_frames': [], 'scene_frames': [],
            'source_ref': {'path': f'MetaVideo/{name}.mp4', 'sha256': name},
            'video': None, 'json_path': f'ProcessedClips/Clips/{name}/Clip.json'}


def data():
    return [(DAY+'_fp', {'segments': [segment('a', 0, 10), segment('b', 10, 20), segment('c', 30, 40,'inactive')]}),
            (DAY+'_tp', {'segments': [segment('d', 5, 15, 'inactive'), segment('e',15,25)]})]


def test_day_union_spans_recordings_without_counting_views_twice():
    result = build_timeline(DAY, data())
    assert result['activity_seconds'] == 25
    assert result['covered_seconds'] == 35
    assert [(p['state'],(p['end_us']-p['start_us'])/1e6) for p in result['periods']] == [
        ('active',25), ('missing',5), ('inactive',10)]
    assert result['experiment_count'] is None
    assert result['person_identity_inferred'] is False
    assert len(result['entries']) == 5


def test_correspondence_includes_inactive_other_view_and_crosses_file_boundaries(default_config):
    entries=build_timeline(DAY,data())['entries']
    links=associate(entries, {'fp':'first_person','tp':'third_person'}, default_config)
    assert len(links)==3
    assert all(edge['shared_action_audit']['algorithm']=='visioncortex.actions.audit_candidates' for edge in links)
    assert all(not edge['clock_offset_verified'] and not edge['same_scene_verified'] for edge in links)
    assert all(edge['status']=='temporal_candidate' for edge in links)
    assert associate(entries, {'fp':'first_person','tp':'first_person'}, default_config)==[]
    assert associate(entries, {}, default_config)==[]
    # A revised source or audit input cannot reuse old association evidence.
    revised=deepcopy(entries)
    revised[0]['source_ref']['sha256']='changed'
    assert associate(revised,{'fp':'first_person','tp':'third_person'},default_config)[0]['link_id']!=links[0]['link_id']


def test_late_other_view_adds_links_without_mutating_original_intervals(default_config):
    initial=build_timeline(DAY,data()[:1])['entries']
    before=deepcopy(initial)
    assert associate(initial, {'fp':'first_person','tp':'third_person'}, default_config)==[]
    complete=build_timeline(DAY,data())['entries']
    assert associate(complete,{'fp':'first_person','tp':'third_person'},default_config)
    assert initial==before


def test_history_cannot_keep_live_priority_and_rollover_is_deterministic():
    r={'recording_start_us':BASE,'processing_priority':0}
    assert scheduling_record(r,'2026-09-11')['processing_priority']==1
    assert scheduling_record(r,DAY)['processing_priority']==0
    assert scheduling_record(r,'2026-09-10')['processing_priority']==1
    assert r['processing_priority']==0


def test_timeline_api_empty_date_is_not_no_experiment(default_config,tmp_path):
    c=deepcopy(default_config)
    c['storage']={'archive_root':str(tmp_path/'archive'), 'local_runtime_root':str(tmp_path/'runtime')}
    app=FastAPI()
    install_routes(app,lambda:c)
    client=TestClient(app)
    result=client.get('/api/day-timeline/'+DAY)
    assert result.status_code==200
    assert result.json()['periods']==[]
    assert result.json()['experiment_count'] is None
    assert client.get('/api/day-timeline/2026-02-31').status_code==400
    assert link(DAY+'_fp','MetaVideo/../../secret') is None
    assert link(DAY+'_fp','https://example.com/a') is None


def test_execution_identity_remains_unchanged_by_timeline_and_queue_updates():
    import ast
    import visioncortex.device_day as module
    from visioncortex.device_day_cache_identity import execution_identity, BASELINE_FILE
    source = Path(module.__file__).read_text()
    current = execution_identity(source)
    assert current != BASELINE_FILE
    tree = ast.parse(source)
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DeviceDayRunner')
    index = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == 'refresh_index')
    index.body = [ast.Pass()]
    assert execution_identity(ast.unparse(tree)) == current
    process = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == '_process')
    process.body.append(ast.Return(value=ast.Constant(value='changed execution')))
    assert execution_identity(ast.unparse(tree)) != current



def test_nonempty_action_evidence_uses_shared_auditor(default_config):
    from visioncortex.schemas import ActionCandidate, ActionType, ViewRole
    left, right = data()[0], data()[1]
    for archive, index in [left, right]:
        seg = index['segments'][0]
        candidate = ActionCandidate(candidate_id='contact', view_id=archive[11:],
            role=ViewRole.FIRST_PERSON if archive.endswith('_fp') else ViewRole.THIRD_PERSON,
            action_type=ActionType.HAND_OBJECT_CONTACT, local_start_ms=0, local_end_ms=1000,
            global_start_ms=0, global_end_ms=1000, key_global_ms=500,
            confidence=.9, objects=['hand', 'balance'], evidence=[{'frame_index': n} for n in range(5)])
        seg['activity_audit'] = {'events': [{'candidates': [candidate.model_dump(mode='json')]}]}
    entries=build_timeline(DAY,[left,right])['entries']
    edges=associate(entries,{'fp':'first_person','tp':'third_person'},default_config)
    assert edges
    assert all(e['shared_action_audit']['status']=='completed' for e in edges)
    assert all(not e['shared_action_audit']['formal_correspondence_promoted'] for e in edges)


def test_reprioritize_preserves_revision_and_completed_evidence(tmp_path):
    from visioncortex.device_day_queue import DeviceDayQueue
    from visioncortex.device_day_schedule import refresh_queue_priorities
    queue=DeviceDayQueue(tmp_path/'queue.sqlite3')
    queue.enqueue({'recording_id':'old','recording_start_us':1,'processing_priority':0,
                   'configured_role':'first_person'},'sealed-version')
    row=queue.claim('owner')
    queue.finish('owner',row['recording_id'],{'status':'completed','proof':'preserved'},1)
    with queue.connect() as db:
        before=dict(db.execute('select * from recordings').fetchone())
    refresh_queue_priorities(queue)
    with queue.connect() as db:
        after=dict(db.execute('select * from recordings').fetchone())
    assert before|{'payload':after['payload']}==after
    import json
    assert json.loads(after['payload'])['processing_priority']==1
