import copy
import json

import cv2
import numpy as np
import pytest

from visioncortex.lab.temporal_display import (SEGMENTS, SEGMENT_BOXES, TemporalDisplay,
    TemporalConfig, local_quality)
from visioncortex.lab.temporal_registration import normalized_box


IDENTITY = dict(device_id='fixture-device', screen_id='panel', field_id='PV', track_session_id='track-1')
LAYOUT = dict(revision='synthetic-reviewed/1', display_type='seven_segment', color='red',
    digits=[[.33, .15, .57, .85]], dynamic_boxes=[[.30, .12, .60, .88]],
    format_evidence_ref='synthetic-layout-no-sign-no-decimal', decimal_mode='none',
    supports_negative=False, opportunity_model_revision='synthetic-controlled-opportunity/1')


def image_with(segments, *, seed=91):
    rng = np.random.default_rng(seed)
    gray = rng.integers(20, 190, (100, 240, 1), dtype=np.uint8)
    image = np.repeat(gray, 3, axis=2)
    image[10:90, 65:155] = 10
    x1, y1, x2, y2 = normalized_box(LAYOUT['digits'][0], image.shape)
    for name in segments:
        a, b, c, d = normalized_box(SEGMENT_BOXES[name], (y2-y1, x2-x1))
        image[y1+b:y1+d, x1+a:x1+c] = (0, 0, 220)
    return image


def visibility(image, observable='abcdefg', layout=None):
    layout = layout or LAYOUT
    mask = np.zeros(image.shape[:2], dtype=bool)
    cells = {}
    x1, y1, x2, y2 = normalized_box(layout['digits'][0], image.shape)
    for name in observable:
        a, b, c, d = normalized_box(SEGMENT_BOXES[name], (y2-y1, x2-x1))
        mask[y1+b:y1+d, x1+a:x1+c] = True
        cells[f'0:{name}'] = dict(observable=True, basis='controlled fixture local scan opportunity, not customer calibration')
    return dict(model_revision=layout['opportunity_model_revision'], evidence_ref='synthetic-calibration-reference',
                method='local_segment_visibility/1', valid_mask=mask, opportunities=cells)


def ref(index, **extra):
    return dict(source_sha256='a'*64, pts=index*33333, time_base='1/1000000', frame_index=index,
                source_to_roi=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], **extra)


def with_latched_fixture(layout=None):
    """Controlled visible mode indicator; contains no numeric truth value."""
    value = copy.deepcopy(layout or LAYOUT)
    box = [.03, .2, .16, .45]
    x1, y1, x2, y2 = normalized_box(box, image_with('ad').shape)
    template = cv2.cvtColor(image_with('ad')[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).tolist()
    value['state_witness'] = dict(kind='latched_display_indicator/1', box=box, template=template,
        calibration_ref='controlled-fixture-latched-mode-calibration', model_revision='fixture-hold/1',
        semantics='this_field_value_latched_while_indicator_visible')
    return value


def observe(engine, image, index, **kwargs):
    return engine.observe(image, identity=kwargs.pop('identity', IDENTITY), source_ref=kwargs.pop('source_ref', ref(index)),
                          layout=kwargs.pop('layout', LAYOUT), **kwargs)


def test_complete_supported_single_frame_does_not_wait_for_thirty():
    image = image_with(SEGMENTS['2'])
    result = observe(TemporalDisplay(), image, 0, visibility=visibility(image), ocr=lambda _: dict(raw_text='2'))
    assert result['status'] == 'confirmed'
    assert result['value_text'] == '2'
    assert result['selected_frame_count'] == 1
    assert result['method'] == 'reliable_single_frame_segments'
    assert result['gates']['confirmed']
    json.dumps(result, allow_nan=False)


def test_missing_light_is_unknown_without_a_calibrated_opportunity():
    image = image_with(SEGMENTS['9'])
    result = observe(TemporalDisplay(), image, 0, ocr=lambda _: dict(raw_text='9', score=.999))
    assert result['status'] == 'uncertain'
    assert result['value_text'] is None
    assert result['segment_observations']['0:e']['state'] == 'UNKNOWN'
    assert '8' in result['candidates'] and '9' in result['candidates']


def test_all_one_mask_cannot_authorize_off():
    image = image_with(SEGMENTS['9'])
    supplied = visibility(image)
    supplied['valid_mask'][:] = True
    result = observe(TemporalDisplay(), image, 0, visibility=supplied)
    assert result['status'] == 'uncertain'
    assert result['quality']['opportunity_model']['reason'] == 'nonlocal_or_all_one_opportunity_mask'
    assert result['segment_observations']['0:e']['state'] == 'UNKNOWN'


def test_complementary_supported_regions_recover_known_stable_fixture():
    engine = TemporalDisplay()
    frames = [image_with('ad'), image_with('e'), image_with('bg')]
    results = [observe(engine, frame, i, layout=with_latched_fixture(),
                       visibility=visibility(frame, 'f' if i == 0 else ''))
               for i, frame in enumerate(frames)]
    assert results[0]['status'] != 'confirmed' and results[1]['status'] != 'confirmed'
    result = results[2]
    assert result['status'] == 'confirmed', result
    assert result['value_text'] == '2'
    assert result['selected_frame_count'] == 3
    assert result['independent_information_patterns'] == 3
    assert len(result['source_refs']) == 3
    assert result['observation_time_end']['pts'] == ref(2)['pts']
    assert result['fusion']['derived_pixels_are_new_observations'] is False


def test_compatible_partial_digits_without_state_witness_stay_uncertain():
    engine = TemporalDisplay()
    for i, segments in enumerate(['ad', 'e', 'bg']):
        frame = image_with(segments)
        result = observe(engine, frame, i, visibility=visibility(frame, 'f' if i == 0 else ''))
    assert result['value_text'] is None
    assert '2' in result['candidates']
    assert 'state_change_vs_degradation_not_excluded' in result['rejection_reasons']


def test_claiming_stable_true_without_indicator_pixels_does_not_prove_state():
    engine = TemporalDisplay()
    layout = with_latched_fixture()
    layout['state_witness']['template'] = np.ones((25, 32)).tolist()
    for i, segments in enumerate(['ad', 'e', 'bg']):
        frame = image_with(segments)
        result = observe(engine, frame, i, layout=layout, visibility=visibility(frame, 'f' if i == 0 else ''))
    assert result['status'] != 'confirmed'
    assert not result['quality']['state_witness']['accepted']


def test_two_to_five_positive_union_cannot_become_confirmed_eight():
    engine = TemporalDisplay()
    for i, digits in enumerate(['2', '5', '2']):
        result = observe(engine, image_with(SEGMENTS[digits]), i)
        assert result['status'] != 'confirmed'
        assert result['value_text'] is None
        if i == 1:
            assert 'state_change_vs_complementary_missing_segments_unresolved' in result['rejection_reasons']


def test_opposite_on_off_resets_before_cross_state_fusion():
    engine = TemporalDisplay()
    first = image_with(SEGMENTS['2'])
    observe(engine, first, 0, visibility=visibility(first))
    next_image = image_with(SEGMENTS['5'])
    result = observe(engine, next_image, 1, visibility=visibility(next_image))
    assert result['status'] != 'confirmed'
    assert result['selected_frame_count'] == 1
    assert result['conflicting_evidence']
    assert 'content_change_or_degradation_conflict' in result['reset_reasons']


def test_repeated_missing_segment_never_accumulates_confidence():
    engine = TemporalDisplay()
    image = image_with(SEGMENTS['9'])
    for index in range(30):
        result = observe(engine, image, index, ocr=lambda _: dict(raw_text='9', score=.999))
        assert result['status'] != 'confirmed'
    assert len(next(iter(engine.states.values())).frames) == 1
    assert 'duplicate_or_near_identical_pixels_do_not_add_support' in result['rejection_reasons']


@pytest.mark.parametrize('kind', ['duplicate', 'reverse', 'gap'])
def test_original_pts_duplicate_reverse_or_gap_does_not_join_old_evidence(kind):
    engine = TemporalDisplay()
    observe(engine, image_with('ad'), 1)
    index = 1 if kind == 'duplicate' else (0 if kind == 'reverse' else 20)
    result = observe(engine, image_with('e'), index)
    assert result['status'] != 'confirmed'
    if kind == 'gap':
        assert 'source_pts_gap' in result['reset_reasons']
        assert result['selected_frame_count'] == 1
    else:
        assert kind+'_pts' in result['rejection_reasons']
        assert engine.buffer_bytes == 0


def test_identity_and_source_revision_isolate_state_and_preserve_refs():
    engine = TemporalDisplay()
    a = image_with(SEGMENTS['2'])
    first = observe(engine, a, 0, visibility=visibility(a), source_ref=ref(0, media_offset_seconds=0., source_frame_pixels_sha256='b'*64))
    other_identity = dict(IDENTITY, field_id='SV')
    b = image_with(SEGMENTS['5'])
    second = observe(engine, b, 0, visibility=visibility(b), identity=other_identity)
    assert first['value_text'] == '2' and second['value_text'] == '5'
    assert len(engine.states) == 2
    assert first['source_refs'][0]['source_frame_pixels_sha256'] == 'b'*64
    assert first['source_refs'][0]['media_offset_seconds'] == 0.
    changed = dict(ref(1), source_sha256='c'*64)
    result = observe(engine, image_with('ad'), 1, source_ref=changed)
    assert 'source_or_layout_revision_changed' in result['reset_reasons']
    assert result['selected_frame_count'] == 1


def test_lcd_text_and_high_score_without_glyph_proof_remains_candidate():
    engine = TemporalDisplay()
    image = image_with('ad')
    calls = []
    layout = dict(revision='fixture-lcd/1', display_type='lcd', dynamic_boxes=[[.3, .12, .6, .88]])
    result = observe(engine, image, 0, layout=layout,
        ocr=lambda patch: calls.append(patch.shape) or dict(raw_text='-01.20', candidates=['-01.20'], score=.999))
    assert calls
    assert result['raw_text'] == '-01.20'
    assert result['value_text'] is None and result['status'] == 'uncertain'
    assert 'lcd_glyph_completeness_unproven' in result['rejection_reasons']


def test_format_configuration_does_not_insert_unseen_decimal_or_sign():
    image = image_with(SEGMENTS['8'])
    layout = copy.deepcopy(LAYOUT)
    layout.pop('format_evidence_ref')
    layout['decimal_after'] = 0
    result = observe(TemporalDisplay(), image, 0, layout=layout)
    assert result['value_text'] is None
    assert 'decimal_observation_missing' in result['rejection_reasons']
    assert 'sign_observation_missing' in result['rejection_reasons']
    assert all('.' not in value for value in result['candidates'])


def test_roi_budget_failure_clears_retained_arrays():
    engine = TemporalDisplay(TemporalConfig(max_buffer_bytes=500))
    result = observe(engine, image_with('ad'), 0)
    assert 'roi_byte_budget_exceeded' in result['rejection_reasons']
    assert engine.buffer_bytes == 0


def test_actual_local_quality_mask_contains_unknown_glare_and_contrast_regions():
    image = image_with('ad')
    image[30:45, 80:120] = 255
    signal, mask, quality = local_quality(image, 'red')
    assert signal.shape == mask.shape == image.shape[:2]
    assert 0 < mask.sum() < mask.size
    assert not mask[35, 90]
    assert quality['glare_fraction'] > 0
    assert not quality['calibrated_probability']


def test_explicit_baseline_controls_are_not_extra_independent_evidence():
    engine = TemporalDisplay(TemporalConfig(compare_baselines=True))
    for i, segments in enumerate(['ad', 'e', 'bg']):
        result = observe(engine, image_with(segments), i, ocr=lambda _: dict(raw_text='candidate'))
    assert set(result['baseline_controls']) == {'best', 'mean', 'median', 'maximum'}
    assert result['baseline_controls_are_independent_evidence'] is False


def test_layout_input_is_not_mutated_by_digit_symbol_exclusion():
    image = image_with('ad')
    layout = copy.deepcopy(LAYOUT)
    layout['symbols'] = [dict(kind='decimal', box=[.60, .75, .64, .85], after_digit=0)]
    before = copy.deepcopy(layout)
    observe(TemporalDisplay(), image, 0, layout=layout)
    assert layout == before


def test_symbol_reference_and_callback_metadata_are_bounded():
    image = image_with('ad')
    layout = copy.deepcopy(LAYOUT)
    layout['symbols'] = [dict(kind='decimal', box=[.6, .7, .65, .8], after_digit=0)]*17
    with pytest.raises(ValueError, match='Bounded display layout'):
        observe(TemporalDisplay(), image, 0, layout=layout)
    with pytest.raises(ValueError, match='Source reference exceeds'):
        observe(TemporalDisplay(), image, 0, source_ref=ref(0, extraneous='x'*20000))
    with pytest.raises(ValueError, match='Bounded original OCR'):
        observe(TemporalDisplay(), image, 0, ocr=lambda _: dict(raw_text='x'*5000))


def test_dropped_frame_spacing_cannot_expand_thirty_frames_past_one_second():
    engine = TemporalDisplay()
    for i in range(31):
        image = image_with('abcdefg'[i % 7])
        image[:, :, 2][image[:, :, 2] == 220] -= i
        result = observe(engine, image, i, source_ref=dict(ref(i), pts=i*40000))
        assert result['selected_span_seconds'] <= 1.
        assert result['selected_frame_count'] <= 30
    state = next(iter(engine.states.values()))
    assert float(state.frames[-1].timestamp-state.frames[0].timestamp) <= 1.


def test_lcd_white_background_is_not_called_emission_glare():
    image = np.full((40, 100, 3), 255, np.uint8)
    cv2.putText(image, '12', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (10, 10, 10), 2)
    _, mask, quality = local_quality(image, 'dark')
    assert quality['glare_fraction'] == 0
    assert quality['possible_bright_clipping_fraction'] > .5
    assert not mask.all()


def red_lens_image(segments):
    """Synthetic degradation matching a real red-lens background, not truth."""
    image = image_with('')
    x1, y1, x2, y2 = normalized_box(LAYOUT['digits'][0], image.shape)
    image[y1:y2, x1:x2] = (20, 20, 160)
    for segment in segments:
        a, b, c, d = normalized_box(SEGMENT_BOXES[segment], (y2-y1, x2-x1))
        image[y1+b:y1+d, x1+a:x1+c] = (90, 110, 255)
    return image


def test_colored_lens_chroma_alone_cannot_activate_seven_segments():
    image = red_lens_image('')
    result = observe(TemporalDisplay(), image, 0)
    assert all(v['state'] == 'UNKNOWN' for v in result['segment_observations'].values())
    assert result['value_text'] is None
    gate = result['quality']['positive_emission_model']['digit_slots']['0']
    assert not gate['accepted']
    assert gate['channel_contrast_span'] == 0
    assert result['quality']['opportunity_model']['reason'] == 'no_calibrated_local_opportunity_model'


def test_red_lens_retains_positive_strokes_but_dark_regions_remain_unknown():
    image = red_lens_image(SEGMENTS['2'])
    result = observe(TemporalDisplay(), image, 0)
    for segment in 'abcdefg':
        expected = 'ON' if segment in SEGMENTS['2'] else 'UNKNOWN'
        assert result['segment_observations'][f'0:{segment}']['state'] == expected
    assert result['status'] == 'uncertain'
    assert '2' in result['candidates']
    assert result['quality']['positive_emission_model']['digit_slots']['0']['accepted']


def test_red_lens_glare_does_not_become_positive_emission_or_off():
    image = red_lens_image('')
    x1, y1, x2, y2 = normalized_box(LAYOUT['digits'][0], image.shape)
    a, b, c, d = normalized_box(SEGMENT_BOXES['a'], (y2-y1, x2-x1))
    image[y1+b:y1+d, x1+a:x1+c] = 255
    result = observe(TemporalDisplay(), image, 0)
    assert result['segment_observations']['0:a']['state'] == 'UNKNOWN'
    assert not any(v['state'] == 'OFF' for v in result['segment_observations'].values())
    assert result['status'] != 'confirmed'
