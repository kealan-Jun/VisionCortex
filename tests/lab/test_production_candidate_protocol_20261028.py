"""CPU-only full-frame protocol checks; no GPU, customer identities or NAS writes."""
from ._support import research_helpers
from fractions import Fraction
import importlib.util
import hashlib
import json
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location(
    'production_candidate_protocol',
    research_helpers() / 'production_candidate_protocol_20261028.py')
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)
GATE_SPEC = importlib.util.spec_from_file_location(
    'production_candidate_gate',
    research_helpers() / 'production_candidate_gate_20261028.py')
gate = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(gate)


def case():
    return dict(id='fixture-stream', duration_seconds=4, sources=[
        dict(relative='packet1.mp4', sha256='a' * 64, offset_seconds=0, duration_seconds=2),
        dict(relative='packet2.mp4', sha256='b' * 64, offset_seconds=2, duration_seconds=2)])


def test_contiguous_source_range_crosses_packet_boundary_without_dropping_intervening_time():
    spans = protocol.source_spans(case(), Fraction(1), Fraction(3))
    assert [(s['source_relative'], s['source_start_seconds'], s['source_end_seconds']) for s in spans] == [
        ('packet1.mp4', '1', '2'), ('packet2.mp4', '0', '1')]
    ref = dict(events=[dict(event_id='fixture-event', case='fixture-stream',
        required_evidence_intervals=[[1, 1.2], [2.8, 3]], question='按顺序描述两个指定时段',
        verified_answer='私有答案绝不能进入公开源计划', required_components=['order'])])
    policy, tasks, questions, private, full = protocol.build_payloads(
        dict(cases=[case()]), ref, context_seconds=0)
    assert tasks[0]['source_spans'] == spans
    assert tasks[0]['source_start_seconds'] == '1' and tasks[0]['source_end_seconds'] == '3'
    assert '私有答案' not in json.dumps([policy, tasks, questions, full], ensure_ascii=False)
    assert '私有答案' in private[0]['verified_answer']
    assert tasks[0]['binding_history'] == []
    assert tasks[0]['physical_clock_uncertainty_seconds'] is None
    assert questions[0]['release_after_all_task_input'] and not questions[0]['write_answers_to_memory']


@pytest.mark.parametrize('second_offset', [1.9, 2.1])
def test_gap_or_overlap_cannot_be_silently_concatenated(second_offset):
    value = case()
    value['sources'][1]['offset_seconds'] = second_offset
    with pytest.raises(ValueError, match='gap or overlap'):
        protocol.source_spans(value, Fraction(1), Fraction(3))


def test_real_lossless_codec_retains_every_frame_native_dimensions_and_cross_packet_time(tmp_path):
    np = pytest.importorskip('numpy')
    import cv2
    probes = {}
    decoded = []
    for packet_id, shade in [('a', 20), ('b', 170)]:
        path = tmp_path / f'{packet_id}.mkv'
        # Lossless fixture at 5 fps; these are actual codec frames, not counted mocks.
        frames = np.zeros((10, 32, 48, 3), dtype=np.uint8)
        for i in range(10):
            frames[i, :, :, :] = (shade + i, i * 10, i * 20)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', '48x32', '-framerate', '5',
            '-i', '-', '-c:v', 'ffv1', str(path)], input=frames.tobytes(), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        probes[packet_id * 64] = protocol.read_probe(path)
        cap = cv2.VideoCapture(str(path))
        restored = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            restored.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        assert len(restored) == 10 and np.array_equal(np.array(restored), frames)
        decoded.extend(restored)
    task = dict(task_id='fixture-01', source_spans=protocol.source_spans(case(), Fraction(1), Fraction(3)))
    rows = protocol.select_frame_rows(task, probes)
    assert len(rows) == 10
    assert [r['source_frame_index'] for r in rows] == list(range(5, 10)) + list(range(5))
    assert [r['case_pts_seconds'] for r in rows] == [str(Fraction(i, 5)) for i in range(5, 15)]
    assert {(r['width'], r['height']) for r in rows} == {(48, 32)}
    assert all(r['source_sha256'] == ('a' if i < 5 else 'b') * 64 for i, r in enumerate(rows))
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 64 * 1024**2


def test_frame_plan_rejects_out_of_order_actual_pts():
    task = dict(task_id='fixture-01', source_spans=protocol.source_spans(case(), Fraction(0), Fraction(2)))
    probe = ([dict(width=48, height=32)] * 2, [Fraction(1), Fraction(0)], {})
    with pytest.raises(ValueError, match='chronological'):
        protocol.select_frame_rows(task, {'a' * 64: probe})


def test_diagnostic_correct_answer_cannot_fill_missing_full_frame_or_day_evidence():
    manifest = dict(tasks=[dict(task_id='quality-01'), dict(task_id='quality-02')])
    result = gate.assess(manifest, dict(tasks=[dict(task_id='quality-01', status='completed',
        all_required_facts_correct=True, unsupported_claims=0, identity_batch_fabrications=0)],
        whole_day=dict(whole_wall_seconds=29 * 60)))
    assert result['finite_known_event_gate'] == 'NOT_PASSED'
    assert result['missing_tasks'] == ['quality-02']
    assert 'original_frame_count' in result['failed_tasks'][0]['failed_checks']
    assert all(w['status'] == 'NOT_DEMONSTRATED' for w in result['nightly_windows'])
    assert result['whole_day_eta_seconds'] is None


def test_explicit_capacity_refusal_stays_in_quality_denominator_and_no_action_claim_is_not_success():
    result = gate.assess(dict(tasks=[dict(task_id='quality-01')]), dict(tasks=[dict(
        task_id='quality-01', status='capacity_refused', text='未见新动作')]))
    assert result['fixed_event_count'] == 1 and result['passed_event_count'] == 0
    assert result['finite_known_event_gate'] == 'NOT_PASSED'


@pytest.mark.parametrize('invalid', [True, float('nan'), float('inf'), -1, None])
def test_invalid_or_missing_wall_and_stage_measurements_never_become_nightly_pass(invalid):
    result = gate.assess(dict(tasks=[dict(task_id='quality-01')]), dict(whole_day=dict(whole_wall_seconds=invalid)))
    assert all(w['status'] == 'NOT_DEMONSTRATED' for w in result['nightly_windows'])


def test_empty_quality_set_cannot_pass_vacuously():
    with pytest.raises(ValueError, match='Nonempty'):
        gate.assess(dict(tasks=[]), {})


def padding_fixture():
    expected = [dict(source_sha256='a' * 64, source_frame_index=i,
                     source_pts_seconds=str(Fraction(i, 30)), width=48, height=32) for i in range(5)]
    delivered = [dict(expected[min(i, 4)], model_frame_index=i,
        padding_kind='original' if i < 5 else 'repeat_last_for_temporal_patch',
        padded_width=56, padded_height=56, source_rect_xywh=[0, 0, 48, 32],
        original_rgb_sha256=f'{min(i, 4):064x}', unpadded_processor_rgb_sha256=f'{min(i, 4):064x}')
        for i in range(8)]
    return expected, delivered


def test_native_spatial_border_and_temporal_tail_padding_never_expand_source_count_or_time():
    expected, delivered = padding_fixture()
    result = gate.validate_native_delivery(expected, delivered, temporal_patch_size=4, spatial_multiple=28)
    assert result['unique_original_frames'] == 5
    assert result['native_tail_padding_frames'] == 3 and result['total_tensor_frames'] == 8
    assert result['added_source_time_seconds'] == 0
    assert result['consumption_scope'] == 'processor_input_only_not_completed_vision_encoding'


@pytest.mark.parametrize('field,value', [('source_frame_index', 2), ('source_pts_seconds', '9'),
    ('unpadded_processor_rgb_sha256', 'f' * 64), ('source_rect_xywh', [0, 0, 47, 32]),
    ('padding_kind', 'original'), ('padded_width', 84)])
def test_padding_cannot_hide_omissions_resize_new_frames_or_invented_time(field, value):
    expected, delivered = padding_fixture()
    delivered[-1][field] = value
    with pytest.raises(ValueError):
        gate.validate_native_delivery(expected, delivered, temporal_patch_size=4, spatial_multiple=28)


@pytest.fixture
def encoder_evidence_fixture(tmp_path):
    """Explicit synthetic receipts test the gate, not an actual GPU completion."""
    row = dict(task_id='quality-01', model_run_id='synthetic-test-run', text='仅合成回执测试',
        status='completed', all_original_frames_verified=True, native_pixels_verified=True,
        actual_processor_consumption_verified=True, source_sha_pts_and_pixel_receipts_verified=True,
        query_history_excluded_from_memory_verified=True, all_required_facts_correct=True,
        unsupported_claims=0, identity_batch_fabrications=0, unknowns_and_unprocessed_explicit=True)
    actual = dict(tasks=[dict(task_id='quality-01', actual_planned_frames=3,
                              ordered_frame_manifest_sha256='a' * 64)])
    encoded = dict(schema='vc-lab-native-encoder-completion-receipt/1', model_run_id='synthetic-test-run',
        task_id='quality-01', frame_manifest_sha256='a' * 64, status='completed',
        unique_original_frames=3, completed_original_frames=3, encoder_implementation='official_unmodified',
        vision_calls=[dict(forward_entered=True, forward_returned=True, cuda_synchronized_return=True,
            error=None, input_tensor_sha256='b' * 64, output_tensor_sha256='c' * 64,
            output_shape=[3, 2], completed_source_frame_ordinals=[0, 1, 2])])
    review = dict(review_type='INDEPENDENT_ENCODER_EVIDENCE_AUDIT', candidate_supplied=False,
        reviewer='synthetic-unit-test-auditor-not-real-run', task_reviews=[dict(
            task_id='quality-01', model_run_id='synthetic-test-run', audit_status='VERIFIED',
            frame_manifest_sha256='a' * 64, encoder_implementation_accepted=True,
            encoder_receipt_file='Encoder.json', answer_file='Answer.json',
            answer_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest())])

    def persist():
        for name, data, key in [('Encoder.json', encoded, 'encoder_receipt_sha256'),
                                ('Answer.json', dict(text=row['text']), 'answer_file_sha256')]:
            raw = json.dumps(data).encode()
            (tmp_path / name).write_bytes(raw)
            review['task_reviews'][0][key] = hashlib.sha256(raw).hexdigest()

    persist()
    return row, actual, encoded, review, tmp_path, persist


def test_self_reported_complete_and_processor_input_cannot_pass_without_independent_encoder_receipt(encoder_evidence_fixture):
    row, _, _, review, _, _ = encoder_evidence_fixture
    row['actual_encoder_completion_verified'] = True
    row['independent_encoder_reviews'] = review  # A candidate cannot inject the separate auditor input.
    result = gate.assess(dict(tasks=[dict(task_id='quality-01')]), dict(tasks=[row]))
    assert result['schema'] == 'vc-lab-production-candidate-gate/2'
    assert result['finite_known_event_gate'] == 'NOT_PASSED'
    assert 'independent_actual_encoder_completion' in result['failed_tasks'][0]['failed_checks']


def test_separate_hash_bound_return_artifacts_can_pass_finite_gate_but_cannot_fill_nightly_evidence(encoder_evidence_fixture):
    row, actual, _, review, root, _ = encoder_evidence_fixture
    result = gate.assess(dict(tasks=[dict(task_id='quality-01')]), dict(tasks=[row]),
        actual_input_audit=actual, independent_encoder_reviews=review, evidence_root=root)
    assert result['finite_known_event_gate'] == 'PASS_DIAGNOSTIC_ONLY'
    assert all(w['status'] == 'NOT_DEMONSTRATED' for w in result['nightly_windows'])


@pytest.mark.parametrize('mutation', ['entry_only', 'missing_frame', 'wrong_run', 'changed_raw', 'unaccepted_implementation'])
def test_encoder_entry_partial_coverage_transplant_and_changed_artifacts_are_rejected(encoder_evidence_fixture, mutation):
    row, actual, encoded, review, root, persist = encoder_evidence_fixture
    if mutation == 'entry_only':
        encoded['vision_calls'][0]['forward_returned'] = False
    elif mutation == 'missing_frame':
        encoded['vision_calls'][0]['completed_source_frame_ordinals'] = [0, 1]
    elif mutation == 'wrong_run':
        encoded['model_run_id'] = 'different-run'
    elif mutation == 'unaccepted_implementation':
        review['task_reviews'][0]['encoder_implementation_accepted'] = False
    if mutation == 'changed_raw':
        (root / 'Encoder.json').write_text('{}')
    else:
        persist()
    result = gate.assess(dict(tasks=[dict(task_id='quality-01')]), dict(tasks=[row]),
        actual_input_audit=actual, independent_encoder_reviews=review, evidence_root=root)
    assert result['finite_known_event_gate'] == 'NOT_PASSED'
    assert result['failed_tasks'][0]['encoder_evidence_errors']
