"""Word archive instructions and JSON reports; preserve retired owned HTML."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from ..device_day_contract import artifact, atomic_bytes, atomic_json, read_json, safe_child
from ..device_day_content_paths import aliases, _publish_aliases
from ..device_day_inputs import identity
from ..device_day_verification import verify_artifact_cached

VERSION = 'visioncortex-archive-documents/1'
GUIDE_SHA256 = '99833f4061637ad20382bb0f2b776f98206b89ed6ae1da5529cf4f5010dec3af'


def html_enabled(config):
    return config.get('device_day', {}).get('archive_html_enabled', True)


def _preserve(root, relative, directory=None):
    """Caller owns the day index lock, or the archive guide lock for root files."""
    allowed = {'Readme.html', 'DataRetentionGuide.docx',
               'LaboratoryDailyReport/LaboratoryDailyReport.html',
               'MultimodalUnderstanding/UnderstandingReport.html'}
    if relative not in allowed or directory != (relative.split('/')[0] if '/' in relative else None):
        raise ValueError('Only known owned archive documents may be retired')
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    source = safe_child(safe_child(root, directory), Path(relative).name) if directory else safe_child(root, relative)
    base = safe_child(root, (directory+'/' if directory else '')+'.Evidence/DocumentHistory')
    def complete(intent, journal):
        target = safe_child(root, intent['to'])
        if not intent['to'].startswith((directory+'/' if directory else '')+'.Evidence/DocumentHistory/'):
            raise ValueError('Retired archive document escapes its owning evidence directory')
        if not verify_artifact_cached(target.parent, intent['file']):
            raise ValueError('Retired archive document failed actual byte verification')
        if directory:
            mapping = {key: value for key, value in aliases(root).items() if key.startswith(directory+'/')}
            mapping[relative] = intent['to']
            _publish_aliases(SimpleNamespace(root=root), directory, mapping)
        atomic_json(journal, intent | {'status': 'completed'})
    if not source.exists():
        if base.is_dir():
            for journal in base.glob('*/Intent.json'):
                intent = read_json(journal)
                if intent.get('from') == relative and intent.get('status') is None:
                    complete(intent, journal)
        return
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 16*1024**2:
        raise ValueError('Owned archive document is not bounded regular data')
    before = identity(source)
    saved = artifact(source.parent, source)
    target = base/saved['sha256']/source.name
    journal = target.parent/'Intent.json'
    intent = {'schema_version': VERSION, 'from': relative, 'to': target.relative_to(root).as_posix(),
              'file': saved, 'source_identity': before, 'raw_media_changed': False, 'model_invoked': False}
    atomic_json(journal, intent)
    guarded_mkdir(target.parent)
    verify_before_submission()
    if identity(source) != before:
        atomic_json(journal, intent | {'status': 'superseded_source_changed'})
        raise ValueError('Owned archive document changed before retirement')
    if target.exists():
        complete(intent, journal)
        verify_before_submission()
        if identity(source) != before:
            raise ValueError('Owned archive document changed before reclaim')
        source.unlink()
    else:
        source.rename(target)
        complete(intent, journal)


def retire_day_html(layout):
    for directory, filename in (('LaboratoryDailyReport', 'LaboratoryDailyReport.html'),
                                ('MultimodalUnderstanding', 'UnderstandingReport.html')):
        _preserve(layout.root, directory+'/'+filename, directory)


def publish_guide(config):
    """Publish the reviewed packaged Word guide, with no model or raw mutation."""
    if html_enabled(config):
        return {'status': 'legacy_html_enabled'}
    from ..device_day import exclusive
    root = Path(config['storage']['archive_root'])
    runtime = Path(config['storage']['local_runtime_root'])/'device-day'
    with exclusive(runtime/'locks'/'ArchiveGuide.lock'):
        source = Path(__file__).parents[1]/'templates'/'DataRetentionGuide.docx'
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 1024**2:
            raise ValueError('Packaged archive guide is unavailable or exceeds its bound')
        before = identity(source)
        data = source.read_bytes()
        if sha256(data).hexdigest() != GUIDE_SHA256 or identity(source) != before:
            raise ValueError('Packaged Word guide differs from its reviewed identity')
        destination = safe_child(root, 'DataRetentionGuide.docx')
        if destination.exists() and (not destination.is_file() or destination.stat().st_size > 16*1024**2):
            raise ValueError('Existing archive guide is not bounded regular data')
        if destination.exists() and destination.read_bytes() != data:
            _preserve(root, 'DataRetentionGuide.docx')
        if not destination.exists():
            atomic_bytes(destination, data)
        if destination.read_bytes() != data:
            raise ValueError('Published Word guide differs from its reviewed bytes')
        _preserve(root, 'Readme.html')
        return {'status': 'published', 'guide': 'DataRetentionGuide.docx', 'sha256': GUIDE_SHA256,
                'archive_html_enabled': False, 'model_invoked': False}
