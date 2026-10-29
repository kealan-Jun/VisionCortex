"""Pin verified publication mounts; never create a directory on an absent NAS."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import time
import uuid


class NASUnavailable(OSError):
    reason = 'nas_unavailable'


CURRENT = ContextVar('device_day_publication_storage_guard', default=None)


def mount_snapshot():
    result = {}
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, details = left.split(), right.split()
        path = fields[4]
        for escaped, literal in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            path = path.replace(escaped, literal)
        major, minor = map(int, fields[2].split(':'))
        result[Path(path)] = {'mount_id': fields[0], 'device': os.makedev(major, minor),
                              'filesystem': details[0], 'source': details[1]}
    return result


@dataclass
class PinnedMount:
    path: Path
    expected: dict
    snapshot: dict
    descriptor: int


class PublicationGuard:
    def __init__(self, config):
        self.config = config
        self.local_only = bool(config.get('runtime', {}).get('local_only'))
        self.mounts = []
        self.source_mounts = []
        self.source_alias = None
        self.aliases = []
        self.local_roots = []
        checks = config.get('device_day', {}).get('publication_storage_checks')
        if self.local_only:
            return
        if not isinstance(checks, list) or not checks:
            raise NASUnavailable('Formal publication requires explicit verified storage checks')
        try:
            initial = mount_snapshot()
            for check in checks:
                if (not isinstance(check, dict) or not {'mount_path', 'filesystem', 'source'}.issubset(check)
                        or not all(isinstance(check[k], str) and check[k] for k in ('mount_path', 'filesystem', 'source'))):
                    raise NASUnavailable('Invalid publication storage identity configuration')
                path = Path(check['mount_path'])
                if not path.is_absolute() or '..' in path.parts:
                    raise NASUnavailable('Publication mount paths must be absolute and canonical')
                observed = initial.get(path)
                if not observed or any(observed[k] != check[k] for k in ('filesystem', 'source')):
                    raise NASUnavailable('Publication storage mount identity mismatch')
                descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                if os.fstat(descriptor).st_dev != observed['device']:
                    os.close(descriptor)
                    raise NASUnavailable('Publication directory does not belong to its verified mount')
                self.mounts.append(PinnedMount(path, check, observed, descriptor))
            self.validate()
            # Resolve only trusted configured roots, before artifact paths are
            # traversed. Internal artifact symlinks still fail O_NOFOLLOW.
            storage = config.get('storage') or {}
            for name in ('archive_root', 'local_runtime_root', 'local_cache_root', 'local_staging_root'):
                if storage.get(name):
                    lexical = Path(storage[name]).absolute()
                    resolved = lexical.resolve(strict=False)
                    if lexical != resolved:
                        self.aliases.append((lexical, resolved))
                    if name.startswith('local_'):
                        self.local_roots.append(resolved)
            self.validate(volume=False)
        except OSError as exc:
            self.close()
            if isinstance(exc, NASUnavailable):
                raise
            raise NASUnavailable('Publication mount cannot be verified or pinned') from exc
        except BaseException:
            self.close()
            raise

    def close(self):
        for mount in (*self.mounts, *self.source_mounts):
            os.close(mount.descriptor)
        self.mounts.clear()
        self.source_mounts.clear()

    def validate(self, *, volume=True):
        if self.local_only:
            return
        try:
            if volume:
                from .device_day_recovery_worker import verify_storage
                # Existing site checks also verify the backup volume marker.
                verify_storage(self.config)
                if any(lexical.resolve(strict=False) != resolved for lexical, resolved in self.aliases):
                    raise NASUnavailable('Configured publication storage alias changed')
            if self.source_alias is not None:
                lexical, resolved = self.source_alias
                if lexical.resolve(strict=False) != resolved:
                    raise NASUnavailable('Configured capture storage alias changed')
            actual = mount_snapshot()
            for mount in (*self.mounts, *self.source_mounts):
                observed = actual.get(mount.path)
                if observed != mount.snapshot or os.fstat(mount.descriptor).st_dev != mount.snapshot['device']:
                    raise NASUnavailable('Publication mount disappeared or changed after it was pinned')
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, NASUnavailable):
                raise
            raise NASUnavailable('Publication storage or its volume marker is unavailable') from exc

    def source_mount_for(self, path):
        """Pin a capture reader without granting that directory publication rights.

        The configured capture tree may contain read-only binds of either
        verified NAS share. Its nearest mount must have an already approved
        filesystem/share identity; unrelated mounts and alias escapes fail closed.
        """
        if self.local_only:
            return self.mount_for(path)
        self.validate(volume=False)
        configured = self.config.get('collection_ingest', {}).get('source_root')
        if not configured or not Path(configured).is_absolute() or '..' in Path(configured).parts:
            raise NASUnavailable('Capture verification requires an explicit source root')
        lexical = Path(configured).absolute()
        if self.source_alias is None:
            self.source_alias = lexical, lexical.resolve(strict=False)
        root = self.source_alias[1]
        path = Path(path).absolute()
        if '..' in path.parts or not (path.is_relative_to(lexical) or path.is_relative_to(root)):
            raise NASUnavailable('Capture path is outside its configured source root')
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise NASUnavailable('Capture path escaped its configured source root')
        actual = mount_snapshot()
        candidates = [parent for parent in actual if resolved.is_relative_to(parent)]
        if not candidates:
            raise NASUnavailable('Capture path has no verified source mount')
        parent = max(candidates, key=lambda value: len(value.parts))
        observed = actual[parent]
        approved = {(mount.expected['filesystem'], mount.expected['source']) for mount in self.mounts}
        if observed['filesystem'] != 'cifs' or (observed['filesystem'], observed['source']) not in approved:
            raise NASUnavailable('Capture source mount identity mismatch')
        pinned = next((mount for mount in (*self.mounts, *self.source_mounts) if mount.path == parent), None)
        if pinned is None:
            try:
                descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            except OSError as exc:
                raise NASUnavailable('Capture source mount cannot be pinned') from exc
            if os.fstat(descriptor).st_dev != observed['device']:
                os.close(descriptor)
                raise NASUnavailable('Capture directory does not belong to its verified mount')
            pinned = PinnedMount(parent, {'filesystem': observed['filesystem'], 'source': observed['source']},
                                 observed, descriptor)
            self.source_mounts.append(pinned)
        self.validate(volume=False)
        return pinned, resolved

    def mount_for(self, path):
        path = Path(path).absolute()
        if '..' in path.parts:
            raise NASUnavailable('Publication path is not canonical')
        aliases = [(lexical, resolved) for lexical, resolved in self.aliases if path.is_relative_to(lexical)]
        if aliases:
            lexical, resolved = max(aliases, key=lambda alias: len(alias[0].parts))
            path = resolved / path.relative_to(lexical)
        matches = [mount for mount in self.mounts if path.is_relative_to(mount.path)]
        if matches:
            return max(matches, key=lambda mount: len(mount.path.parts)), path
        if not self.local_only:
            if not any(path.is_relative_to(root) for root in self.local_roots):
                raise NASUnavailable('Formal publication destination has no verified mount or explicit local state root')
        from .local_storage import require_local_path
        try:
            require_local_path(path)
        except RuntimeError as exc:
            raise NASUnavailable('Network publication destination has no verified mount') from exc
        return None, path

    def directory(self, mount, parts):
        """Create/open relative to the pinned superblock, rejecting symlinks."""
        descriptor = os.dup(mount.descriptor)
        try:
            for part in parts:
                self.validate(volume=False)
                try:
                    next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(part, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                if os.fstat(descriptor).st_dev != mount.snapshot['device']:
                    raise NASUnavailable('Publication path crossed into an unverified filesystem')
            self.validate(volume=False)
            return descriptor
        except OSError as exc:
            os.close(descriptor)
            if isinstance(exc, NASUnavailable):
                raise
            raise NASUnavailable('Verified publication directory is unavailable or unsafe') from exc
        except BaseException:
            os.close(descriptor)
            raise

    def mkdir(self, path):
        mount, path = self.mount_for(path)
        if mount is None:
            path.mkdir(parents=True, exist_ok=True)
        else:
            self.validate(volume=False)
            descriptor = self.directory(mount, path.relative_to(mount.path).parts)
            os.close(descriptor)
            self.validate(volume=False)

    def atomic(self, path, blocks):
        mount, path = self.mount_for(path)
        if mount is None:
            return False
        self.validate(volume=False)
        parts = path.relative_to(mount.path).parts
        if not parts:
            raise NASUnavailable('Publication cannot replace the mount root')
        parent = self.directory(mount, parts[:-1])
        temporary = f'.{parts[-1]}.{uuid.uuid4().hex}.partial'
        operation = 'create_temporary'
        try:
            self.validate(volume=False)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o666, dir_fd=parent)
            with os.fdopen(descriptor, 'wb') as handle:
                operation = 'read_source_blocks'
                for block in blocks:
                    operation = 'write_temporary'
                    handle.write(block)
                    operation = 'read_source_blocks'
                operation = 'flush_temporary'
                handle.flush()
                operation = 'sync_temporary'
                os.fsync(handle.fileno())
            self.validate()
            operation = 'replace_destination'
            # A CIFS sharing lease may briefly deny an otherwise valid rename.
            # Keep the same fsynced temporary file and pinned directory; never
            # unlink the destination, change permissions, or bypass the guard.
            for attempt in range(5):
                try:
                    os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EBUSY} or attempt == 4:
                        raise
                    time.sleep(0.05 * 2**attempt)
                    self.validate()
            operation = 'sync_directory'
            os.fsync(parent)
            self.validate(volume=False)
            return True
        except OSError as exc:
            if isinstance(exc, NASUnavailable):
                raise
            if operation == 'read_source_blocks':
                raise  # A lazy source read is not a destination filesystem failure.
            failure = NASUnavailable(exc.errno,
                f'Verified publication filesystem rejected {operation}', str(path))
            failure.operation = operation
            raise failure from exc
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except OSError:
                pass
            os.close(parent)


@contextmanager
def storage_guard(config):
    guard = PublicationGuard(config)
    token = CURRENT.set(guard)
    try:
        yield guard
    finally:
        CURRENT.reset(token)
        guard.close()


def verify_before_submission():
    guard = CURRENT.get()
    if guard is not None:
        guard.validate()


def guarded_mkdir(path):
    guard = CURRENT.get()
    if guard is None:
        path.mkdir(parents=True, exist_ok=True)
    else:
        guard.mkdir(path)


def guarded_atomic(path, blocks):
    guard = CURRENT.get()
    return guard.atomic(path, blocks) if guard is not None else False
