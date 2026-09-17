"""Read time-index JSON files and return separate, traceable media streams."""
from copy import deepcopy
from datetime import datetime, timedelta
import math

from .device_day_time_lookup import interval
from .media_time import media_ms, valid_points


def query_file_indexes(indexes, at_us, duration_seconds=1):
    if (isinstance(at_us, bool) or not isinstance(at_us, int) or at_us <= 0
            or not isinstance(duration_seconds, (int, float)) or isinstance(duration_seconds, bool)
            or not math.isfinite(duration_seconds) or not 0 < duration_seconds <= 3600):
        raise ValueError('Use a positive Unix microsecond timestamp and a window of up to one hour')
    end_us = at_us + max(1, round(duration_seconds*1000000))
    def overlaps(item):
        return interval(item.get('start_us'), item.get('end_us')) and item['start_us'] < end_us and item['end_us'] > at_us
    matches = []
    for index in indexes:
        if index.get('schema_version') != 'visioncortex-file-time-index/2':
            raise ValueError('File lookup requires a version 2 TimeIndex.json')
        streams = index['streams']
        originals = {e['recording_id']: e for e in streams['video'] if e['kind'] == 'original_video'}
        selected = {key: [] for key in ('video', 'audio', 'images', 'multimodal')}
        for stream in selected:
            for item in streams[stream]:
                if not overlaps(item):
                    continue
                result = deepcopy({k: v for k, v in item.items() if k != 'clock_mapping'})
                seek_us = max(at_us, item['start_us'])
                if stream == 'video':
                    original = originals.get(item['recording_id'], {})
                    clock = original.get('clock_mapping') or {}
                    points = valid_points(clock)
                    if clock.get('points') and (not points or not points[0][1] <= seek_us <= points[-1][1]):
                        result.update(file_offset_seconds=None, seek_status='outside_verified_clock_samples')
                    elif original.get('start_us'):
                        offset, basis = media_ms(clock, seek_us, original['start_us'])
                        clip = item['kind'] == 'video_segment' and item.get('file')
                        start_ms = (item.get('source_start_ms') or 0) if clip else 0
                        result.update(source_video_offset_seconds=offset/1000,
                            file_offset_seconds=max(0, (offset-start_ms)/1000), time_basis=basis,
                            seek_status='mapped_native_pts_unverified',
                            playback_file=item.get('file') or item.get('source_video'))
                elif stream == 'audio':
                    result['file_offset_seconds'] = (seek_us-item['start_us'])/1000000
                    result['transcription']['sentences'] = [s for s in result['transcription']['sentences'] if overlaps(s)]
                selected[stream].append(result)
        if any(selected.values()):
            ids = {e.get('recording_id') for values in selected.values() for e in values}
            matches.append({'archive': index['archive'], 'path_bases': index['path_bases'],
                'source_index_updated_at': index.get('source_index_updated_at'),
                'recordings': [r for r in index.get('recordings', []) if r['recording_id'] in ids],
                'streams': selected})
    return {'schema_version': 'visioncortex-file-time-query/1', 'start_us': at_us, 'end_us': end_us,
            'timezone': 'Asia/Shanghai', 'interval_semantics': 'start_inclusive_end_exclusive',
            'cross_camera_alignment_verified': False, 'matches': matches}


def main():
    import argparse
    import json
    from pathlib import Path
    from .device_day_contract import atomic_json, read_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, action='append', default=[])
    parser.add_argument('--archive-root', type=Path, help='Discover this date\'s device TimeIndex.json files')
    parser.add_argument('--at', required=True, help='ISO timestamp including timezone, e.g. 2026-09-17T17:50:52+08:00')
    parser.add_argument('--duration-seconds', type=float, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    moment = datetime.fromisoformat(args.at)
    if moment.utcoffset() is None:
        parser.error('--at must include an explicit timezone')
    at_us = round(moment.timestamp()*1000000)
    query_file_indexes([], at_us, args.duration_seconds)  # Validate before filesystem discovery.
    paths = list(args.index)
    if args.archive_root:
        from zoneinfo import ZoneInfo
        first = moment.astimezone(ZoneInfo('Asia/Shanghai'))
        last = first + timedelta(seconds=args.duration_seconds) - timedelta(microseconds=1)
        for day in sorted({first.date().isoformat(), last.date().isoformat()}):
            paths.extend(sorted(args.archive_root.glob(f'{day}_*/Comment/TimeIndex.json')))
    if not paths:
        parser.error('No TimeIndex.json found; supply --index or --archive-root')
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    result = query_file_indexes([read_json(p) for p in paths], at_us, args.duration_seconds)
    if args.output:
        if args.output.resolve() in set(paths):
            parser.error('--output cannot overwrite an input index')
        atomic_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
