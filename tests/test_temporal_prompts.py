import pytest

from visioncortex.temporal_prompts import select_temporal_prompts


def proposal(identity, label, score, box):
    return {'id': identity, 'label': label, 'confidence': score, 'box': box}


def test_same_geometry_keeps_conflicting_evidence_and_covers_other_classes():
    rows = [proposal('cap1', 'cap', .99, [0, 0, 10, 10]),
            proposal('cap2', 'cap', .98, [20, 0, 30, 10]),
            proposal('box', 'pipette_tip_box', .9, [40, 0, 50, 10]),
            proposal('rack', 'pipette_rack', .8, [40, 0, 50, 10]),
            proposal('bottle', 'bottle', .7, [60, 0, 70, 10])]
    selected, audit = select_temporal_prompts(rows, max_objects=3)
    assert [p['id'] for p in selected] == ['cap1', 'box', 'bottle']
    group = selected[1]['proposal_group']
    assert {m['label'] for m in group['members']} == {'pipette_tip_box', 'pipette_rack'}
    assert group['label_status'] == 'ambiguous_model_proposals'
    assert group['physical_identity_confirmed'] is False
    assert sum(len(g['members']) for g in audit['groups']) == len(rows)


def test_no_transitive_merging_or_nested_object_removal():
    # Adjacent boxes overlap >= .85, but the first and third do not.
    rows = [proposal(str(i), str(i), .9 - i*.1, [i*7, 0, 100+i*7, 100]) for i in range(3)]
    rows.append(proposal('inside', 'tube', .5, [40, 40, 50, 50]))
    selected, _ = select_temporal_prompts(rows)
    assert len(selected) == 3
    assert len(selected[0]['proposal_group']['members']) == 2
    assert selected[-1]['id'] == 'inside'


def test_bounded_eight_and_invalid_values():
    rows = [proposal(str(i), str(i), .8, [i*20, 0, i*20+10, 10]) for i in range(12)]
    selected, audit = select_temporal_prompts(rows, max_objects=8)
    assert len(selected) == 8 and len(audit['groups']) == 12
    for kwargs in [{'max_objects': 9}, {'overlap_threshold': float('nan')}]:
        with pytest.raises(ValueError):
            select_temporal_prompts(rows, **kwargs)
    with pytest.raises(ValueError):
        select_temporal_prompts([rows[0], rows[0]])


def test_hands_excluded_and_input_unchanged():
    hand = proposal('h', 'hand', .99, [0, 0, 10, 10])
    selected, _ = select_temporal_prompts([hand])
    assert selected == [] and 'proposal_group' not in hand
