"""Disposition of derived video only; originals and audio are never removed."""
from ..device_day_contract import atomic_bytes, atomic_json
from ..device_day_narrative import segment_record


def disposition(segment, meaning, recording, context=None, *, segments=None):
    readable = segment_record(segment, meaning, recording, context, segments=segments)
    windows = (meaning or {}).get('windows', [])
    # Do not mistake a failed/pending/unobservable interval for an idle one.
    clear_static = (bool(windows) and meaning.get('status') == 'completed'
                    and all(w.get('activity_kind') == 'static' and w.get('observability') == 'clear'
                            and not w.get('experiment_steps') and not w.get('work_activities')
                            for w in windows))
    covered = sorted((w['start_ms'], w['end_ms']) for w in windows if 'start_ms' in w and 'end_ms' in w)
    clear_static &= len(covered) == len(windows)
    cursor = segment['start_ms']
    for left, right in covered:
        if left > cursor + 1:
            clear_static = False
        cursor = max(cursor, right)
    clear_static &= cursor >= segment['end_ms'] - 1
    speech = recording.get('transcription') or {}
    audio = recording.get('audio') or {}
    audio_pending = audio.get('status') == 'provided' and speech.get('status') != 'completed'
    protected = readable['留存判断']['保护原因']
    omit = (clear_static and segment['end_ms'] - segment['start_ms'] >= 60000
            and not protected and not audio_pending)
    return {'schema_version': 'visioncortex-derived-video-disposition/1',
        'decision': 'omit_static_derivative' if omit else 'retain_derivative',
        'reason': ('完成的抽样窗口均为可见静止场景，没有已关联的活动、实验等待、读数或语音保护信号。'
                   if omit else '存在活动、保护信号、未完成理解或可见性/录音缺口，保留处理视频。'),
        'scope': 'derived_video_only', 'original_video_unchanged': True, 'audio_unchanged': True,
        'source_ref': segment['source_ref'], 'source_start_ms': segment['start_ms'],
        'source_end_ms': segment['end_ms'], 'start_us': segment['start_us'], 'end_us': segment['end_us'],
        'protection_signals': protected, 'model_evidence': readable['溯源']['执行记录'],
        'continuous_inactivity_proven': False,
        'limitation': '仅省略处理后的视频副本；原片仍可回看。抽样静止不证明帧间绝无短动作。'}


def publish_disposition(folder, decision, *, video_present=False):
    """Publish an explanation only when this output truly has no derived video.

    Never unlink a pre-existing video here: its owning materializer must publish
    a separate generation, preserving historical artifacts and source mappings.
    """
    value = decision | {'derived_video_present': video_present}
    atomic_json(folder / 'Retention.json', value)
    if decision['decision'] == 'omit_static_derivative' and not video_present:
        note = ('此时间区间未另行保存处理后视频。\n' + decision['reason'] + '\n'
                '原视频保持不动，录音及转写保持不动。\n'
                f'原视频：{decision["source_ref"]["path"]}\n'
                f'原片位置：{decision["source_start_ms"]}—{decision["source_end_ms"]} ms\n'
                + decision['limitation'] + '\n详见 Retention.json 中的输入、输出和时间证据。\n')
        atomic_bytes(folder / 'VideoNotRetained.txt', note.encode())
    return value
