"""CPU-only causal branches and native cache-selection/floor-rule checks."""
from ._support import research_helpers
import sys

import pytest

SCRIPTS = research_helpers()
sys.path.insert(0, str(SCRIPTS))
from onestreamer_retention_cache import audited_render, causal_rows, floor_frame_index, native_citations, recent_requests, replay, stable_sha, target_overlap, validate_cached_sources


def row(start, raw='</Observe> transfer from A to B'):
    return dict(start=start, end=start+1, sampled_timestamps=[start+i/4 for i in range(4)], raw=raw, kind='response')


def test_causal_filter_uses_only_completed_updates_and_never_false_stream_end():
    rows = [row(0), dict(round='stream_end', start=0, end=1, raw='future-summary-leak'), row(1), row(2)]
    assert causal_rows(rows, 2) == [rows[0], rows[2]]


@pytest.mark.parametrize('bad', [[row(1), row(0)], [row(0), row(2)], [dict(row(0), sampled_timestamps=[1.])]])
def test_invalid_order_gap_and_future_boundary_frames_are_rejected(bad):
    with pytest.raises(ValueError):
        causal_rows(bad, 10)


def test_caption_sampling_uses_official_floor_instead_of_round():
    assert floor_frame_index(.25, 0, '30/1') == 7
    assert floor_frame_index(.75, 0, '30/1') == 22
    assert floor_frame_index(900.25, 900, '30/1') == 7
    assert round(.25*30) == 8


def test_caption_cache_must_match_new_inventory_source_hash_and_native_input_policy():
    case = dict(sources=[dict(offset_seconds=0, duration_seconds=900, relative='source', sha256='a'*64)])
    good = dict(row(0), source='source', source_sha256='a'*64)
    validate_cached_sources([good], case)
    with pytest.raises(ValueError, match='source'):
        validate_cached_sources([dict(good, source_sha256='b'*64)], case)
    with pytest.raises(ValueError, match='policy'):
        validate_cached_sources([dict(good, sampled_timestamps=[0,.5])], case)


def test_recent_native_16_frames_are_strictly_past_and_cross_packets_with_source_ids():
    case = dict(duration_seconds=1800, sources=[dict(relative='p'+str(i), offset_seconds=i*900,
        duration_seconds=900, stream=dict(avg_frame_rate='30/1')) for i in range(2)])
    frames = recent_requests(case, 905)
    assert len(frames) == 16 and max(r['requested_time'] for r in frames) < 905
    assert {f['source']['relative'] for f in frames} == {'p0', 'p1'}
    assert frames[-1]['source_frame_index'] == 149
    assert recent_requests(case, 2)[0]['source_frame_index'] == 29


@pytest.fixture
def native():
    import ast
    from dataclasses import dataclass
    source = research_helpers() / 'onestreamer_source'
    if not source.exists():
        pytest.skip('Pinned native OneStreamer research checkout not present on this host')
    sys.path.insert(0, str(source))
    from onestreamer.memory import MemoryBank, MemoryEvent, parse_caption_output
    # Execute the untouched native tracker class without importing its unrelated
    # model-serving module (the local CPU test environment need not contain torch).
    path = source/'onestreamer/caption_optimized.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SegmentScopeTracker')
    context = dict(dataclass=dataclass, MemoryEvent=MemoryEvent, parse_caption_output=parse_caption_output)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), context)
    return context['SegmentScopeTracker'], MemoryBank, MemoryEvent


def test_native_scope_tracker_replay_does_not_use_future_summary_or_mutate_source_cache(native):
    tracker_class, _, _ = native
    rows = [row(0), row(1, '</Summary> old transfer segment'), row(2, '</Observe> a later action'), row(3, '</Summary> FUTURE')]
    original = stable_sha(rows)
    tracker, events, prefix = replay(rows, 3, tracker_class, 'silence_observe_summary_v50_training_user')
    assert len(prefix) == 3 and len(events) == 3
    assert events[1].start == 0 and events[1].end == 2 and tracker.segment_start == 2
    assert all('FUTURE' not in e.text for e in events)
    assert stable_sha(rows) == original


def test_actual_native_memory_event_selection_is_audited_with_ids_and_token_cap(native):
    _, bank_class, event_class = native
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())
    tokenizer = Tokenizer()
    events = [event_class('response', i, i+1, 'transfer A to B at '+str(i), '</Observe> '+str(i)) for i in range(8)]
    bank = bank_class(events)
    full, fullmeta = audited_render(bank, 'A to B', tokenizer, 12000)
    base = len(bank_class([]).render('hierarchical').encode())
    short, meta = audited_render(bank, 'A to B', tokenizer, base+100)
    assert not fullmeta['truncated'] and fullmeta['actual_selected_events'] == 8
    assert meta['truncated'] and 0 < meta['actual_selected_events'] < 8
    assert meta['memory_tokens'] <= base+100
    assert set(meta['selected_event_ids']) <= set(meta['generated_event_ids'])
    assert short != full and len({stable_sha(e.to_dict()) for e in events}) == 8


def test_target_overlap_and_native_citations_keep_evaluation_mapping_distinct(native):
    _, _, event_class = native
    case = dict(sources=[dict(relative='source',sha256='a'*64,offset_seconds=0,duration_seconds=900)])
    event = event_class('response',165,172,'moves container','raw')
    identity = stable_sha(event.to_dict())
    record = target_overlap([event],{identity},'165—172秒做了什么？',case)
    assert record['events'][0]['selected']
    assert record['events'][0]['evaluation_source_mapping'][0]['start_seconds'] == 165
    citations = native_citations('165—172秒发生动作；175—199秒的情况未知。',180,case)
    assert citations[0]['within_causal_history'] and not citations[1]['within_causal_history']
    assert citations[0]['timestamp_accuracy'] == 'NOT_VERIFIED'
