"""Static detector cues must not become experiment material."""
from pathlib import Path

import pytest

from visioncortex.actions import audit_candidates, build_activity_intervals, build_experiment_segments
from visioncortex.analysis.observations import interaction_temporal_support
from visioncortex.schemas import ActionCandidate, ActionType, AlignmentTransform, ViewInput, ViewRole


def candidate(action, *, approach=False, changed=False, motion=0):
    return ActionCandidate(
        candidate_id=action.value, action_type=action, view_id="fixed", role=ViewRole.THIRD_PERSON,
        local_start_ms=1000, local_end_ms=6000, global_start_ms=1000, global_end_ms=6000,
        key_global_ms=3000, objects=["gloved_hand", "bottle_cap"], confidence=.95,
        evidence=[{"frame_index": n, "interaction_state": {"approach_confirmed": approach},
                   "state_cue": "bottle_cap", "state_before": "cap_visible",
                   "state_after": "open_visible" if changed else "cap_visible",
                   "hand_track_id": 1, "object_track_id": 2,
                   "object_instance": {"center_norm": [.5, .5]},
                   "hand_instance": {"center_norm": [.5 + motion * n / 19, .5]}}
                  for n in range(20)])


def classify(items, config):
    config["segmentation"]["allow_strong_single_role_actions"] = True
    config["segmentation"]["require_both_roles"] = False
    t = AlignmentTransform(view_id="fixed", reference_view_id="fixed", state="aligned", confidence=1)
    events, _ = audit_candidates(items, {"fixed": t}, config)
    views = [ViewInput(view_id="fixed", role=ViewRole.THIRD_PERSON, video=Path("fixture.mp4"))]
    return events, build_experiment_segments(events, views, config)


@pytest.mark.parametrize("action", [ActionType.HAND_OBJECT_CONTACT, ActionType.CONTAINER_STATE_CHANGE,
                                    ActionType.DEVICE_PANEL_OPERATION])
def test_persistent_high_confidence_static_cue_cannot_publish_activity(default_config, action):
    item = candidate(action)
    events, segments = classify([item], default_config)
    assert len(events) == 1 and events[0].formal_admission_status == "provisional"
    assert segments == []
    intervals, decisions = build_activity_intervals(events, [item], [(0, 10000)], default_config)
    assert intervals == [] and decisions[0]["activity_seed_eligible"] is False
    assert events[0].candidates[0].evidence == item.evidence


@pytest.mark.parametrize("kind", ["approach", "motion", "state_change"])
def test_observed_partial_operation_retains_activity(default_config, kind):
    item = (candidate(ActionType.CONTAINER_STATE_CHANGE, changed=True) if kind == "state_change"
            else candidate(ActionType.HAND_OBJECT_CONTACT, approach=kind == "approach",
                           motion=.04 if kind == "motion" else 0))
    events, _ = classify([item], default_config)
    assert events[0].formal_admission_status == "formal"
    assert build_activity_intervals(events, [item], [(0, 10000)], default_config)[0]


def test_unknown_initial_state_is_not_a_state_change(default_config):
    item = candidate(ActionType.CONTAINER_STATE_CHANGE)
    item.evidence[0]["state_before"] = "unknown"
    assert interaction_temporal_support(item, default_config)["supported"] is False


def test_gap_inferred_release_and_small_jitter_cannot_confirm_contact(default_config):
    item = candidate(ActionType.HAND_OBJECT_CONTACT, motion=.001)
    item.provenance["interaction_state"] = {"release_observed": True,
                                            "release_inferred_from_next_observation_gap": True}
    assert interaction_temporal_support(item, default_config)["supported"] is False


def test_static_rejected_context_cannot_extend_real_operation(default_config):
    anchor = candidate(ActionType.HAND_OBJECT_CONTACT, approach=True)
    stale = candidate(ActionType.DEVICE_PANEL_OPERATION)
    stale.local_start_ms = stale.global_start_ms = 15000
    stale.local_end_ms = stale.global_end_ms = 20000
    stale.key_global_ms = 17000
    events, _ = classify([anchor, stale], default_config)
    intervals, _ = build_activity_intervals(events, [anchor, stale], [(0, 30000)], default_config)
    assert max(end for _, end in intervals) <= 9000


def test_reviewed_scan_reuse_binds_detector_inputs_and_source_changes(default_config, tmp_path):
    from visioncortex.device_day_contract import artifact, atomic_json, digest, file_hash
    from visioncortex.device_day_models import DeviceDayModels
    root = tmp_path / 'cache'
    old = {'source': 'sealed-media', 'sample_fps': 20, 'windows': [[0, 1000]], 'code': {'detection': 'detector-code'},
           'recursive_sources': {'analysis/observations.py': 'before', 'detection.py': 'unchanged'}}
    new = old | {'recursive_sources': old['recursive_sources'] | {'analysis/observations.py': 'after'}}
    directory = root / 'device-day-scans' / digest(old)
    directory.mkdir(parents=True)
    ledger = directory / 'detections.jsonl'
    ledger.write_text('actual recorded detector evidence')
    manifest = directory / 'scan-receipt.json'
    atomic_json(manifest, {'status': 'completed', 'identity': old, 'ledger': ledger.name,
                           'artifacts': [artifact(directory, ledger)], 'wall_seconds': 10})
    proof = tmp_path / 'Binding.json'
    entry = {'directory': directory.relative_to(root).as_posix(), 'manifest_sha256': file_hash(manifest)}
    value = {'schema_version': 'visioncortex-verified-scan-reuse/1',
             'reviewed_consumer_changes': {'analysis/observations.py': {'from': 'before', 'to': 'after'}},
             'entries': {digest(new): entry}}
    atomic_json(proof, value)
    default_config['storage']['local_cache_root'] = str(root)
    default_config.setdefault('device_day', {})['verified_scan_reuse'] = {'path': str(proof), 'sha256': file_hash(proof)}
    model = DeviceDayModels(default_config)
    assert model._reviewed_scan(new)['_ledger'] == ledger
    assert model._reviewed_scan(new | {'windows': [(0, 1000)]})['_ledger'] == ledger
    assert model._reviewed_scan(new | {'sample_fps': 10}) is None
    # Even an explicitly inserted target cannot change detector inputs under
    # an operation-consumer-only review.
    changed = new | {'sample_fps': 10}
    value['entries'][digest(changed)] = entry
    atomic_json(proof, value)
    default_config['device_day']['verified_scan_reuse']['sha256'] = file_hash(proof)
    model = DeviceDayModels(default_config)
    with pytest.raises(ValueError, match='inputs or detector'):
        model._reviewed_scan(changed)
    ledger.write_text('modified bytes')
    with pytest.raises(ValueError, match='artifacts changed'):
        model._reviewed_scan(new)
