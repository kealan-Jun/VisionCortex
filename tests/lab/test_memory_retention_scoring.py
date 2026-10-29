"""Guard against inflated retention, future/changed questions and stale judgments."""
from ._support import research_helpers
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(research_helpers()))
from score_memory_retention import aggregate, score


def row(event='e1', delay=1, correct=False, model='m'):
    return dict(id=f'{event}_{delay}', event_id=event, case='c', model=model, run='r',
                history_interval_seconds=delay, overall_correct=correct,
                components={'action': 'correct' if correct else 'incorrect'},
                unsupported_claim=False)


def test_near_wrong_does_not_become_a_forgetting_denominator():
    result = aggregate([row(), row(delay=600)])['groups']['m/r']
    conditional = result['conditional_on_near_correct']['600']
    assert conditional['scorable'] == 0 and conditional['fraction'] is None
    assert conditional['interpretation'] == 'not estimable'
    assert result['by_delay']['600']['scorable'] == 1


def test_conditional_uses_same_event_and_excludes_near_wrong_events():
    result = aggregate([row(correct=True), row(delay=600),
                        row('e2'), row('e2', delay=600, correct=True)])['groups']['m/r']
    assert result['by_delay']['600']['fraction'] == .5
    conditional = result['conditional_on_near_correct']['600']
    assert conditional['eligible_events'] == ['e1'] and conditional['fraction'] == 0


def test_common_events_require_both_near_answers_correct():
    rows = [row(correct=True), row(delay=600, correct=True), row('e2'),
            row(model='other', correct=True), row(model='other', delay=600),
            row('e2', model='other', correct=True)]
    common = aggregate(rows)['common_near_correct_events']
    assert {r['event_id'] for r in common} == {'e1'}
    assert next(r for r in common if r['history_interval_seconds'] == 600)['right'] is False


def test_duplicate_judgments_cannot_inflate_denominators():
    with pytest.raises(ValueError, match='duplicate'):
        aggregate([row(), row()])


def fixture(tmp_path):
    question = dict(id='e1_near', event_id='e1', case='c', question_time=11,
                    evidence_end=10, question='Which way?')
    answer = dict(**question, text='A to B')
    raw = json.dumps([answer]).encode()
    (tmp_path / 'Answers.json').write_bytes(raw)
    mark = dict(id='e1_near', answer_file='Answers.json',
                answer_file_sha256=hashlib.sha256(raw).hexdigest(),
                answer_text_sha256=hashlib.sha256(b'A to B').hexdigest(),
                components={'direction': 'correct'}, unsupported_claim=False, notes='reviewed')
    annotation = dict(model='m', run='r', judgments=[mark])
    reference = dict(reference_type='ASSISTANT_VISUAL_REVIEW',
                     events=[dict(event_id='e1', required_components=['direction'])])
    return reference, annotation, dict(queries=[question])


def test_modified_raw_answer_is_rejected_before_scoring(tmp_path):
    args = fixture(tmp_path)
    (tmp_path / 'Answers.json').write_text('[]')
    with pytest.raises(ValueError, match='answer file changed'):
        score(*args, tmp_path)


def test_future_or_modified_question_and_missing_component_are_rejected(tmp_path):
    reference, annotation, questions = fixture(tmp_path)
    questions['queries'][0]['question_time'] += 600
    with pytest.raises(ValueError, match='question contract changed'):
        score(reference, annotation, questions, tmp_path)
    questions['queries'][0]['question_time'] -= 600
    annotation['judgments'][0]['components'] = {}
    with pytest.raises(ValueError, match='required component'):
        score(reference, annotation, questions, tmp_path)


def test_onestreamer_actual_prefix_is_not_its_future_materialized_cache(tmp_path):
    reference, annotation, questions = fixture(tmp_path)
    path = tmp_path / 'Answers.json'
    answer = json.loads(path.read_text())[0]
    answer.update(actual_memory_updates=11, actual_memory_sampled_frames=44,
                  actual_memory_latest_frame=10.733333, processed_video_time_at_query=11,
                  worker_latest_materialized_video_time=3600,
                  selection=dict(actual_generated_events=6, actual_selected_events=5,
                                 memory_tokens=200, truncated=True))
    path.write_text(json.dumps([answer]))
    annotation['judgments'][0]['answer_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    row = score(reference, annotation, questions, tmp_path)[0]
    assert row['actual_memory_frames'] == 44 and row['actual_memory_updates'] == 11
    assert row['processed_video_time_at_query'] == 11
    assert row['worker_latest_materialized_video_time'] == 3600
    assert row['generated_memory_events'] == 6 and row['selected_memory_events'] == 5
