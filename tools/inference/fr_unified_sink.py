"""Owned original-FR archive worker with a guarded unified projection facade.

Use the original FR interpreter/environment and existing private FIELD_* values.
No models, capture workers or new business database are started by this entry.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import signal
import sys


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--projection-only',action='store_true')
    parser.add_argument('--sequence',type=int)
    args=parser.parse_args(argv)
    from visioncortex.lab.field_unified_sink import UnifiedFRSink,load_manifest
    manifest=load_manifest(args.manifest)
    package=Path(manifest['fr_package_root'])
    if package.name!='field_recognition' or not (package/'__init__.py').is_file():
        raise ValueError('Explicit original field_recognition package directory required')
    if 'field_recognition' in sys.modules:
        if Path(sys.modules['field_recognition'].__file__).resolve().parent!=package:
            raise ValueError('Another FR implementation was already imported')
    else:
        spec=importlib.util.spec_from_file_location('field_recognition',package/'__init__.py',submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec);sys.modules['field_recognition']=module
        spec.loader.exec_module(module)
    # Only original source/runtime resources and SQLite are used by the store.
    os.environ['FIELD_SERVICE_ROLE']='archive'
    from field_recognition.runtime.production_config import ProductionConfig
    from field_recognition.runtime.factory import create_runtime
    from field_recognition.runtime.archive_worker import ArchiveWorker
    from field_recognition.runtime.worker_support import process_lock
    config=ProductionConfig.from_environment()
    if config.record_mode!='production' or config.use_sample_registry:
        raise ValueError('Existing production FR scope with no sample registry is required')
    if not Path(config.database_path).is_absolute() or not Path(config.database_path).is_file():
        raise ValueError('Explicit existing original FR SQLite path required')
    core=create_runtime(config,project_root=package.parent,service_role='archive')
    facade=UnifiedFRSink(core['archive_store'],manifest)
    worker=ArchiveWorker('unified-fr-archive-'+str(os.getpid()),config,core=core)
    worker.archive_store=facade
    if args.projection_only:
        import json
        try:
            with process_lock(core['DATA'],'archive'):
                print(json.dumps(facade.project_once(sequence=args.sequence),ensure_ascii=False))
        finally:worker.close()
        return
    if args.sequence is not None:raise ValueError('--sequence requires --projection-only')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:worker.stop.set())
    worker.run(persistent=not args.once)


if __name__=='__main__':main()
