"""Actual execution messages require their specific verified repair grant."""
import pytest

from test_retry_repair import evidence, exhausted, row
from visioncortex.device_day_retry import apply_retry_plan, failure_kind, plan_retry


@pytest.mark.parametrize('error,message,kind,checks', [
    ('RuntimeError', 'NMS postprocessing repeatedly timed out; refusing incomplete frame evidence',
     'vision_nms', ('vision_model_verified', 'nms_repair_verified')),
    ('ValueError', 'Required source frame could not be decoded',
     'media_decode', ('source_decode_verified',)),
])
def test_installed_execution_failure_receives_only_its_verified_bounded_retry(tmp_path, error, message, kind, checks):
    result = {'status': 'failed', 'error_type': error, 'message': message}
    assert failure_kind(result) == kind
    queue = exhausted(tmp_path, result)
    before = row(queue)
    repair = evidence() | {'repair_kind': kind, 'checks': {
        'current_source_verified': 'PROVEN', 'successful_requests_preserved': 'PROVEN',
        **{check: 'PROVEN' for check in checks}}}
    incomplete = repair | {'checks': {key: value for key, value in repair['checks'].items() if key != checks[0]}}
    with pytest.raises(ValueError, match='missing a required verification'):
        plan_retry(queue.path, 'record', incomplete)
    plan = plan_retry(queue.path, 'record', repair)
    assert row(queue) == before
    assert apply_retry_plan(queue, plan)
    current = row(queue)
    assert current['retry_limit'] == 6
    assert current['attempts'] == 3 and current['status'] == 'failed'
    assert current['result'] == before['result']


@pytest.mark.parametrize('message', [
    'NMS postprocessing failed for an unknown reason',
    'NMS is unsupported by the current plugin',
    'Required source frame metadata could not be read',
    'The decoder stopped for an unknown reason',
])
def test_unknown_nms_or_decoder_error_is_not_a_recognized_repair(message):
    assert failure_kind({'error_type': 'RuntimeError', 'message': message}) is None
