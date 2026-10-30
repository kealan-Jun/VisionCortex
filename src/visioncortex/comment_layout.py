"""Comment index compatibility; source recordings and observations stay intact."""
from copy import deepcopy
import re

_TIME = re.compile(r'^\d{2}-\d{2}-\d{2}(?:\.\d+)?_\d{2}-\d{2}-\d{2}(?:\.\d+)?$')
_PATH_KEYS = {'path', 'content', 'text_file', 'content_file', 'transcript_file',
              'transcript_path', 'source_index', 'field_events_file'}


def current_audio_references(document, groups, references):
    """Rewrite only registered transcript links in a mutable current index.

    Original receipts use the existing aliases and retain their old hashes.
    Current indexes instead refer to the readable current member and its hash.
    Spoken text, recording identities and media time are never normalized.
    """
    groups = set(groups)
    if any(not _TIME.fullmatch(group) for group in groups):
        raise ValueError('Invalid Comment time directory')

    def target(value):
        if not isinstance(value, str):
            return value
        parts = value.split('/')
        if (len(parts) == 3 and parts[0] == 'Comment' and parts[1] in groups
                and parts[2] in {'Transcript.json', 'Transcript.txt'}):
            return 'Comment/Audio/' + '/'.join(parts[1:])
        return value

    def walk(value, trail=()):
        if isinstance(value, list):
            return [walk(item, trail + ('[]',)) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            text_link = key == 'text' and ('entries' in trail or trail[:2] == ('readable_content', 'audio'))
            result[key] = target(item) if (key in _PATH_KEYS or text_link) and isinstance(item, str) else walk(item, trail + (key,))
        if isinstance(value.get('path'), str) and result.get('path') != value['path']:
            reference = references[result['path']]
            for key in ('sha256', 'size_bytes', 'byte_size'):
                if key in result:
                    result[key] = reference[key]
        return result

    result = walk(deepcopy(document))
    readable = result.get('readable_content')
    if isinstance(readable, dict) and 'digest' in readable:
        from .device_day_contract import digest
        readable['digest'] = digest([readable.get('audio', []), readable.get('multimodal', [])])
    return result


def publish_legacy_index(root, filename, value, *, writer=None, reader=None):
    """Keep the established index entry point current alongside Audio."""
    from .device_day_contract import atomic_json, read_json
    writer = writer or atomic_json
    reader = reader or read_json
    if filename not in {'Index.json', 'TimeIndex.json'}:
        raise ValueError('Unknown Comment compatibility index')
    path = root / 'Comment' / filename
    if not path.is_file() or reader(path) != value:
        writer(path, value)
