"""Native codec/sample-clock boundaries, using generated fixtures without models."""
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import wave

import pytest

from visioncortex.lab.audio_source import sample_coverage
from visioncortex.lab.local_provider import PreparedMedia
from visioncortex.lab.timeline_runner import TimelineRunner


def native_document():
    return dict(streams=[dict(codec_name='opus', sample_rate='48000', time_base='1/48000',
                             start_pts=312, duration_ts=5112,
                             extradata='00000000: 4f70 7573 4865 6164 0101 3801 80bb 0000  OpusHead\n00000010: 0000 00  ...')],
                format=dict(duration='0.106500'),
                frames=[dict(pkt_pts=312, nb_samples=648)] + [dict(pkt_pts=n, nb_samples=960) for n in range(960, 4800, 960)])


def test_verified_opus_edges_are_not_blanket_duration_tolerance():
    proof = sample_coverage(native_document(), '0', '.1065')
    assert proof['actual_sample_range'] == dict(start_seconds='13/2000', end_seconds='1/10')
    assert proof['prepared_zero_to_source_seconds'] == '13/2000'
    assert proof['prepared_duration_seconds'] == '187/2000'
    assert len(proof['boundary_without_samples']) == 2
    assert proof['decoded_native_sample_count'] == 4488
    assert proof['continuous_sample_clocks_verified']


@pytest.mark.parametrize('change', ['gap', 'overlap', 'reorder', 'missing_pts', 'unknown_start', 'unknown_tail', 'no_codec_proof', 'lying_duration'])
def test_unknown_sample_holes_and_unproved_edges_reject(change):
    doc = native_document()
    if change == 'gap':
        doc['frames'][2]['pkt_pts'] += 1
    elif change == 'overlap':
        doc['frames'][2]['pkt_pts'] -= 1
    elif change == 'reorder':
        doc['frames'][1:3] = reversed(doc['frames'][1:3])
    elif change == 'missing_pts':
        doc['frames'][2].pop('pkt_pts')
    elif change == 'unknown_start':
        doc['streams'][0]['start_pts'] += 1
    elif change == 'unknown_tail':
        doc['frames'][-1]['nb_samples'] -= 1
    elif change == 'no_codec_proof':
        doc['streams'][0]['extradata'] = ''
    else:
        doc['format']['duration'] = '30'
    with pytest.raises(ValueError):
        sample_coverage(doc, '0', '30' if change == 'lying_duration' else '.1065')


def preparer(path, duration):
    ref = dict(media_id='fixture-audio', media_version=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               source_kind='fixture', start_seconds='0', end_seconds=str(duration))
    prepared = object.__new__(PreparedMedia)
    prepared.settings = SimpleNamespace(max_native_bytes=64*1024**2)
    prepared.media = SimpleNamespace(locate=lambda _: dict(status='available', version=1, path=path))
    return prepared, ref


def test_native_opus_preparation_proves_trim_and_prepared_origin(tmp_path):
    path = tmp_path/'generated-fixture.opus'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=0.5',
                    '-c:a', 'libopus', str(path)], check=True)
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)]))
    prepared, ref = preparer(path, metadata['format']['duration'])
    raw, mime, receipt = prepared.prepare(ref, 'transcription', {}, tmp_path)
    proof = receipt['transform']['source_sample_coverage']
    assert mime == 'audio/wav' and Fraction(proof['prepared_zero_to_source_seconds']) == 0
    with wave.open(io.BytesIO(raw)) as decoded:
        assert decoded.getnframes() == 8000
    assert Fraction(proof['prepared_duration_seconds']) == Fraction(1, 2)
    assert len(proof['boundary_without_samples']) == 1
    assert receipt['transform']['duration_tolerance_seconds'] == '0.000062500000'
    assert not list(tmp_path.glob('input-*'))


def test_nonzero_native_audio_seek_preserves_original_sample_content(tmp_path):
    path = tmp_path/'generated-distinct-samples.wav'
    # Different plateaus detect an origin/seek error without an ASR model.
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\x10\x00'*16000 + b'\x20\x00'*16000 + b'\x30\x00'*16000 + b'\x40\x00'*16000)
    prepared, ref = preparer(path, 4)
    ref.update(start_seconds='2.25', end_seconds='3.25')
    raw, _, receipt = prepared.prepare(ref, 'transcription', {}, tmp_path)
    with wave.open(io.BytesIO(raw)) as decoded:
        samples = decoded.readframes(decoded.getnframes())
    assert samples == b'\x30\x00'*12000 + b'\x40\x00'*4000
    assert Fraction(receipt['transform']['source_sample_coverage']['prepared_zero_to_source_seconds']) == Fraction(9, 4)
    ref.update(start_seconds='0', end_seconds='30')
    with pytest.raises(ValueError, match='cover'):
        prepared.prepare(ref, 'transcription', {}, tmp_path)


@pytest.mark.parametrize('aligned', [True, False])
def test_ASR_maps_prepared_zero_and_limits_unaligned_text_to_actual_samples(aligned):
    ref = dict(media_id='fixture-audio', media_version=1, sha256='a'*64, source_kind='fixture', start_seconds='0', end_seconds='.1065')
    proof = sample_coverage(native_document(), '0', '.1065')
    runner = object.__new__(TimelineRunner)
    runner.media = SimpleNamespace(prepare=lambda *args: (b'generated-wave', 'audio/wav', dict(transform=dict(source_sample_coverage=proof))))
    result = dict(text='fixture transcript')
    if aligned:
        result['segments'] = [dict(text=result['text'], start_seconds='.01', end_seconds='.03')]
    client = SimpleNamespace(infer=lambda *args: (result, dict(response_sha256='b'*64)))
    events, receipts = runner._asr(client, {}, dict(core=ref, input_spans=[ref]), Path('.'))
    span = events[0]['spans'][0]
    assert (span['start_seconds'], span['end_seconds']) == (('0.016500', '0.036500') if aligned else ('0.006500', '0.100000'))
    assert events[0]['payload']['comment_candidate']['status'] == 'pending_interpretation'
    assert events[0]['payload']['source_sample_coverage'] == proof
    assert receipts[0]['derivation']['transform']['source_sample_coverage'] == proof
    if aligned:
        result['segments'][0]['end_seconds'] = '.1'
        with pytest.raises(ValueError, match='escaped'):
            runner._asr(client, {}, dict(core=ref, input_spans=[ref]), Path('.'))


@pytest.mark.parametrize('segments', [True, False])
def test_ASR_point_words_retain_original_observations_without_invented_duration(segments):
    ref = dict(media_id='fixture-audio', media_version=1, sha256='a'*64, source_kind='fixture', start_seconds='0', end_seconds='.1065')
    proof = sample_coverage(native_document(), '0', '.1065')
    runner = object.__new__(TimelineRunner)
    runner.media = SimpleNamespace(prepare=lambda *args: (b'fixture', 'audio/wav', dict(transform=dict(source_sample_coverage=proof))))
    words = [dict(text='point', start_seconds='.01', end_seconds='.01'), dict(text='word', start_seconds='.01', end_seconds='.03')]
    result = dict(text='point word', words=words)
    if segments:
        result['segments'] = [dict(text=result['text'], start_seconds='.005', end_seconds='.035')]
    client = SimpleNamespace(infer=lambda *args: (result, dict(response_sha256='b'*64)))
    events, _ = runner._asr(client, {}, dict(core=ref, input_spans=[ref]), Path('.'))
    event = events[0]
    alignment = event['payload']['asr_alignment']
    assert alignment['original_word_observations'] == words
    assert alignment['zero_duration_word_indices'] == [0]
    point = alignment['source_word_observations'][0]
    assert point['start_seconds'] == point['end_seconds'] == '0.016500000000'
    assert alignment['selected_unit'] == ('segments' if segments else 'whole_input_alignment_unavailable')
    assert event['payload']['comment_candidate']['alignment_available'] is segments
    assert event['payload']['comment_candidate']['status'] == 'pending_interpretation'
    span = event['spans'][0]
    assert (span['start_seconds'], span['end_seconds']) == (('0.011500', '0.041500') if segments else ('0.006500', '0.100000'))
    words[0]['end_seconds'] = '.2'
    with pytest.raises(ValueError, match='escaped'):
        runner._asr(client, {}, dict(core=ref, input_spans=[ref]), Path('.'))
