"""Reuse the original resumable understanding, report and per-source publisher.

Vision/ASR are exact PostgreSQL parents, never re-inferred by this adapter.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import time
import hashlib

from .native_comment_layout import module as pipeline_module
from .media import safe_path
from .rules import Rules
from .storage import verify_root, storage_binding
from .domain import fingerprint


class ParentBackend:
    def __init__(self,config,request):
        self.config,self.request=config,request

    def vision(self,layout,retention,key):
        parent=next(p for p in self.request['parents'] if p['worker_kind']=='vision')
        return dict(deepcopy(parent['result']),postgres_parent_job_id=parent['id'],parent_model_reinvoked=False)

    def transcribe(self,layout,retention,key):
        parent=next((p for p in self.request['parents'] if p['worker_kind']=='asr'),None)
        if parent is None:
            # A no-audio result is emitted only after the original media probe.
            result=pipeline_module('device_day_stt').transcribe(dict(self.config,speech_recognition={'enabled':False}),layout,retention,key)
            if result.get('outcome')!='no_audio':
                raise ValueError('Audio requires its own completed PostgreSQL ASR parent')
            return result
        record=self.request['native_record']
        audio=record.get('audio') or {}
        origin=int(audio.get('start_us') or record['recording_start_us'])
        result=parent['result']
        if not result.get('execution_receipts') or any(r.get('actual_asr_invocation')!='PROVEN' for r in result['execution_receipts']):
            raise ValueError('ASR parent has no actual execution receipts')
        comments=[dict(comment_id=f"{parent['id']}:{n}",text=s['text'],start_us=origin+round(float(s['start_seconds'])*1e6),end_us=origin+round(float(s['end_seconds'])*1e6)) for n,s in enumerate(result.get('segments',[]))]
        return dict(outcome='transcribed' if comments else 'no_transcript',comments=comments,artifacts=[],chunks=[],model_invocation='PROVEN',accuracy='NOT_PROVEN',postgres_parent_job_id=parent['id'],execution_receipts=result['execution_receipts'],parent_model_reinvoked=False)

    def understand(self,layout,recording,vision,context,key):
        if self.request.get('external_calls_authorized') is not True:
            raise ValueError('Customer understanding is not configured/authorized')
        backend=pipeline_module('device_day_models').DeviceDayModels(self.config)
        return backend.understand(layout,recording,vision,context,key)


def configuration(request,source_root):
    runtime=Path(request['runtime_root'])/'native'
    config=Rules().vision_configuration(request['vision_model'],runtime)
    primary=Rules().profile('pipeline')['settings']
    config['device_day'].update(Rules().profile('native_pipeline')['settings']['device_day'])
    config['device_day'].update(enabled=True,raw_originals_read_only=True,delete_capture_sources=False,archive_html_enabled=False,vc_lab_archive_layout=Rules().profile('archive')['schema_version'])
    config['storage'].update(archive_root=str(source_root),local_staging_root=str(runtime/'staging'))
    config['collection_ingest']={'enabled':True,'source_root':str(source_root),'settle_seconds':.5}
    config['mllm']=dict(primary['mllm'],**request['model']['connection'])
    config['daily_report']=deepcopy(primary.get('daily_report',{}))
    vision=next(p for p in request['parents'] if p['worker_kind']=='vision')
    speech=next((p for p in request['parents'] if p['worker_kind']=='asr'),None)
    config['models']['vc_lab_parent_execution']=fingerprint(vision)
    config['speech_recognition']=dict(primary.get('speech_recognition',{}),enabled=bool(speech),vc_lab_parent_execution=fingerprint(speech))
    # Site-specific receipt/recovery/cleanup bindings from the old deployment
    # are deliberately absent. The existing algorithms/templates remain pinned.
    binding=storage_binding(source_root,os.environ.get('VC_LAB_NAS_MARKER_SHA256'))
    config['runtime'].update(binding['runtime'])
    config['device_day'].update(binding.get('device_day',{}))
    return config


def deliver(request):
    root=verify_root(Path(request['source_root']),os.environ.get('VC_LAB_NAS_MARKER_SHA256'))
    if request.get('unified_archive_required'):
        from .archive_layout import verify_archive_contract
        verify_archive_contract(root)
    record=request['native_record']
    video=safe_path(root,Path(record['video_path']).relative_to(root).as_posix())
    fresh=pipeline_module('nas_recordings')._inspect(root,video,time.time(),.5,allow_plain=False)
    if not fresh['processable'] or fresh['source_signature']!=record['source_signature']:
        raise ValueError('Native recording changed before semantic delivery')
    role=request['vision_model'].get('camera_roles',{}).get(record['camera_key'])
    if role not in {'first_person','third_person'}:
        raise ValueError('Native delivery requires this customer camera registration')
    # Keep the independently probed embedded-audio flag from intake.
    fresh['configured_role']=role
    config=configuration(request,root)
    guard=pipeline_module('device_day_write_guard')
    with guard.storage_guard(config):
        return _execute(request,fresh,config)


def _execute(request,fresh,config):
    runner=pipeline_module('device_day').DeviceDayRunner(config,backend=ParentBackend(config,request))
    result=runner.process(fresh,stage='all',retry=True)
    if result.get('status')!='completed':
        return dict(analysis_outcome='native_stage_pending',native_status=result.get('status'),native_result=result,model_invoked=False,normalized_fields=[])
    layout=runner.layout(fresh)
    contract=pipeline_module('device_day_contract')
    refs=[]; semantic=None
    for stage in ('retention','vision','stt','understanding','report'):
        path=layout.receipts/fresh['recording_id']/f'{stage}.json'
        data=contract.read_json(path)
        if data.get('status')!='completed':
            raise ValueError('Native stage is not actually completed')
        raw=path.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
        snapshot=path.parent/'history'/f'{stage}-postgres-{checksum}.json'
        if snapshot.exists() and snapshot.read_bytes()!=raw:
            raise ValueError('Immutable native stage snapshot changed')
        if not snapshot.exists():contract.atomic_bytes(snapshot,raw)
        refs.append(dict(stage=stage,reference=layout.backend_artifact(snapshot)))
        if stage=='understanding':semantic=data
    report=layout.reports/'LaboratoryDailyReport.json'
    if not report.is_file():
        raise ValueError('Original daily report did not become readable')
    raw=report.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
    saved=layout.reports/'.Evidence/Versions'/f'{checksum}.json'
    if saved.exists() and saved.read_bytes()!=raw:
        raise ValueError('Immutable report snapshot changed')
    if not saved.exists():contract.atomic_bytes(saved,raw)
    index=contract.read_json(layout.index)
    windows=[w for x in semantic.get('understandings',[]) for w in x.get('windows',[])]
    return dict(analysis_outcome='observations',model_invoked=any(not w.get('response_cache_reused',False) for w in windows),normalized_fields=[],association_status='single_device_activity',
                component='visioncortex.pipeline.DeviceDayRunner',native_result=result,native_stage_receipts=refs,
                daily_report=contract.artifact(layout.root,saved),daily_report_latest=contract.artifact(layout.root,report),retained_video_publication=index.get('retained_video_publication'),
                understandings=semantic.get('understandings',[]),physical_action_confirmed=False,accuracy='NOT_PROVEN')
