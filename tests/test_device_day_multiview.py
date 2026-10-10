from copy import deepcopy
from pathlib import Path
from visioncortex.device_day_multiview import _mapped, _materialize, shared_analysis
from visioncortex.schemas import AlignmentTransform, AlignmentSegmentTransform, ViewInput, VideoInfo, VideoSegmentInfo


def test_rebase_recorder_evidence_without_sharing_track_ids_across_slices():
    tr = AlignmentTransform(view_id='fp', reference_view_id='fp', scale=1.01, offset_ms=20)
    original = {'local_start_ms': 100, 'global_end_ms': 300, 'key_global_ms': 200,
                'evidence': [{'frame_index': 5, 'object_track_id': 7, 'observation_global_ms': 200}],
                'instance_signature': {'track_ids': [7], 'instance_keys': ['device:balance:7']},
                'provenance': {'interaction_state': {'release_observed_at_global_ms': 300}}}
    before = deepcopy(original)
    a = _mapped(original, 5000, tr, 150000, 1)
    b = _mapped(original, 6000, tr, 180000, 2)
    assert a['local_start_ms'] == 5100
    assert a['global_end_ms'] == tr.to_global(5300)
    assert a['key_global_ms'] == tr.to_global(5200)
    assert a['evidence'][0]['frame_index'] == 150005
    assert a['evidence'][0]['object_track_id'] != b['evidence'][0]['object_track_id']
    assert a['instance_signature']['track_ids'] == [a['evidence'][0]['object_track_id']]
    assert a['instance_signature']['instance_keys'] == ['device:balance:1000000007']
    assert a['provenance']['interaction_state']['release_observed_at_global_ms'] == tr.to_global(5300)
    assert original == before


def test_shared_analysis_uses_offline_functions_and_quarantines_unpaired_activity(default_config, tmp_path, monkeypatch):
    from visioncortex import alignment, actions, grouping
    views = [ViewInput(view_id=v, role=role, video=tmp_path/(v+'.mp4'))
             for v,role in [('fp','first_person'),('tp','third_person')]]
    infos = {v.view_id: VideoInfo(path=v.video, duration_ms=10000, fps=30, width=10, height=10, frame_count=300,
                    segments=[VideoSegmentInfo(path=v.video, virtual_start_ms=0, virtual_end_ms=10000,
                      frame_start_index=0, duration_ms=10000, fps=30, width=10, height=10, frame_count=300)]) for v in views}
    sources = {v.view_id:[{'record': {'recording_id':v.view_id, 'processing':{'batches':[], 'clock_mapping': {'basis':'recorder_csv_interpolation', 'points': [[0,1000000000],[10000,1010000000]]}}}, 'namespace':i+1,'fine_indexes':[]}]
               for i,v in enumerate(views)}
    calls=[]
    def align(views, infos, settings):
        calls.append('alignment')
        return {v.view_id:AlignmentTransform(view_id=v.view_id, reference_view_id='fp', state='aligned',confidence=1, segment_transforms=[AlignmentSegmentTransform(segment_index=0, local_start_ms=0,local_end_ms=10000, state='aligned',confidence=1)])
                for v in views}, {}
    monkeypatch.setattr(alignment,'build_alignments',align)
    for module,name in [(actions,'audit_candidates'),(actions,'refine_liquid_events_with_context'),
                        (actions,'build_experiment_segments'),(grouping,'normalize_experiment_segments'),
                        (grouping,'prepare_formal_experiment_segments'),(grouping,'build_experiment_groups'),
                        (grouping,'select_key_events')]:
        original=getattr(module,name)
        def spy(*args,_original=original,_name=name,**kwargs):
            calls.append(_name)
            return _original(*args,**kwargs)
        monkeypatch.setattr(module,name,spy)
    result=shared_analysis(views,infos,sources,default_config)
    assert result['groups']==[] and result['selected_key_events']==[]
    assert calls==['alignment','audit_candidates','refine_liquid_events_with_context','build_experiment_segments',
                   'normalize_experiment_segments','prepare_formal_experiment_segments','build_experiment_groups','select_key_events']


def test_device_placement_reuses_complete_offline_exporter(default_config,tmp_path,monkeypatch):
    from visioncortex import archive
    from visioncortex.device_day_contract import artifact
    c=deepcopy(default_config)
    c['storage']={'archive_root':str(tmp_path/'nas'), 'local_runtime_root':str(tmp_path/'runtime')}
    group={'group_id':'G','continuity_type':'continuous','atomic_experiment_ids':['S'],
           'global_start_ms':0,'global_end_ms':10000,'participating_views':['fp','tp'],
           'first_person_view':'fp','third_person_view':'tp','continuity_reason':'test only'}
    event={'experiment_id':'AlignedExperimentTest','group':group,'sources':[
        {'camera':'fp','archive':'2026-09-14_fp','start_us':1789350000000000,'end_us':1789350010000000},{'camera':'tp','archive':'2026-09-14_tp','start_us':1789350000000000,'end_us':1789350010000000}],
        'transforms':{v:AlignmentTransform(view_id=v,reference_view_id='fp').model_dump() for v in ['fp','tp']},
        'atomic_segments':[{'segment_id':'S','global_start_ms':0,'global_end_ms':10000,'event_ids':[],
                            'participating_views':['fp','tp']}], 'events':[]}
    for s in event['sources']:
        root=tmp_path/'nas'/s['archive']
        root.mkdir(parents=True)
        source=root/'Original.mp4'
        source.write_bytes(b'synthetic contract fixture')
        s['source_ref']=artifact(root,source)
    called=[]
    def export(layout,groups,segments,events,views,infos,transforms,config):
        called.append((groups[0].group_id,segments[0].segment_id))
        for key in ['first-person','third-person','aligned_first_third']:
            f=layout.experiment_clips/(key+'.mp4')
            f.write_bytes(key.encode())
            groups[0].videos[key]=f.relative_to(layout.root).as_posix()
    monkeypatch.setattr(archive,'materialize_experiment_clips',export)
    _materialize([event],[],{}, {},c)
    assert called==[('G','S')]
    from visioncortex.device_day_multiview import _experiment_folder
    for archive_name in ['2026-09-14_fp', '2026-09-14_tp']:
        folder = _experiment_folder(event, archive_name)
        assert folder.startswith('09-40-00_09-40-10_ExperimentActivity_')
        assert not (tmp_path/'nas'/archive_name/'ProcessedClips/Clips/AlignedExperimentTest').exists()
    assert (tmp_path/'nas/2026-09-14_fp/ProcessedClips/Clips'/folder/'AlignedFirstThird.mp4').read_bytes()==b'aligned_first_third'
    assert (tmp_path/'nas/2026-09-14_tp/ProcessedClips/Clips'/folder/'ExperimentActivity.mp4').read_bytes()==b'third-person'
    assert event['materializer']=='visioncortex.archive.materialize_experiment_clips'
    assert len(event['outputs'])==2
    _materialize([event],[],{}, {},c)
    assert called==[('G','S')]  # Adding another slice must not re-encode this group.


def test_bad_recording_does_not_remove_other_slices_and_valid_probes_are_reused(default_config,tmp_path,monkeypatch):
    from visioncortex import alignment, video_io
    from visioncortex.device_day_multiview import _probe_sources
    c=deepcopy(default_config)
    c['storage']['local_runtime_root']=str(tmp_path/'runtime')
    sources={'fp':[],'tp':[]}
    calls=[]
    for camera, name in [('fp','good'),('fp','bad'),('tp','good')]:
        video=tmp_path/(camera+name+'.mp4')
        clock=video.with_suffix('.csv')
        video.write_bytes(b'fixture')
        clock.write_text('clock fixture')
        sources[camera].append({'archive':'2026-09-14_'+camera,'record':{'recording_id':name},
                                'video':video,'clock':clock,'refs':{}})
    def probe(views,**kwargs):
        v=views[0]
        calls.append(v.video.name)
        return {v.view_id:VideoInfo(path=v.video,duration_ms=10_000,fps=30,width=10,height=10,frame_count=300)}
    def clocks(path,*args,**kwargs):
        if 'bad' in path.name:
            raise ValueError('Timestamp CSV needs two rows')
        return []
    monkeypatch.setattr(video_io,'probe_views',probe)
    monkeypatch.setattr(alignment,'read_timestamp_csv_bounded',clocks)
    original=deepcopy(sources)
    missing=[]
    views,infos=_probe_sources(sources,{'fp':'first_person','tp':'third_person'},c,missing)
    assert len(views)==2 and len(infos['fp'].segments)==1
    assert missing==[{'camera':'fp','archive':'2026-09-14_fp','recording_id':'bad','reason':'ValueError','stage':'media_probe'}]
    _probe_sources(original,{'fp':'first_person','tp':'third_person'},c,[])
    assert sorted(calls)==['fpbad.mp4','fpbad.mp4','fpgood.mp4','tpgood.mp4']


def test_real_recorder_gap_is_preserved_by_shared_clock_alignment(default_config,tmp_path,monkeypatch):
    from visioncortex import alignment, video_io
    from visioncortex.schemas import VideoSegmentInput
    from visioncortex.device_day_multiview import _place_recordings, _validate_capture_alignment
    from pytest import approx
    c=deepcopy(default_config)
    c['alignment']['reference_view']='fp'
    c['alignment']['reference_strategy']='configured'
    monkeypatch.setattr(alignment,'visual_anchor_calibration',lambda *a,**kw:(0.,0.,[]))
    views,infos,sources=[],{},{}
    for camera, starts in [('fp',[0,3_600_000]),('tp',[0,3_600_050])]:
        parts,media,records=[],[],[]
        for n,start in enumerate(starts):
            clock=tmp_path/f'{camera}{n}.csv'
            clock.write_text('frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n'+
                ''.join(f'{i*10},{i*1000},{1_000_000+start+i*1000},1\n' for i in range(10)))
            video=clock.with_suffix('.mp4')
            parts.append(VideoSegmentInput(video=video,timestamps_csv=clock))
            media.append(VideoInfo(path=video,duration_ms=10000,fps=10,width=64,height=48,frame_count=100))
            records.append({'archive':'2026-09-14_'+camera,'record':{'recording_id':str(n),'processing':{
                'clock_mapping':{'basis':'recorder_csv_interpolation',
                'points':[[0,(1_000_000+start)*1000],[10000,(1_010_000+start)*1000]]}}}})
        view=ViewInput(view_id=camera,role='first_person' if camera=='fp' else 'third_person',segments=parts)
        views.append(view)
        infos[camera]=video_io._build_segmented_view_info(view,media)
        sources[camera]=records
    _place_recordings(infos,sources)
    tr,_=alignment.build_alignments(views,infos,c)
    assert infos['fp'].segments[1].virtual_start_ms == 3_600_000
    assert infos['fp'].segments[1].frame_start_index == 100
    assert tr['tp'].to_global(infos['tp'].segments[1].virtual_start_ms+2000)==approx(3_602_050,abs=1)
    assert not _validate_capture_alignment(tr,infos,sources,c)
    # The former compressed timeline mapped recordings an hour apart to one
    # event. Even a fit labelled 'aligned' may not override retained CSV time.
    tr['tp'].segment_transforms[1].offset_ms-=3_600_000
    errors=_validate_capture_alignment(tr,infos,sources,c)
    assert len(errors)==1 and errors[0]['recording_id']=='1'
    assert tr['tp'].segment_transforms[1].state=='failed'


def test_unverified_context_does_not_publish_blank_activity_video(default_config,tmp_path,monkeypatch):
    from visioncortex import archive
    c=deepcopy(default_config)
    c['storage']['archive_root']=str(tmp_path/'nas')
    c['storage']['local_runtime_root']=str(tmp_path/'runtime')
    def forbidden(*a,**kw):
        raise AssertionError('An unverified interval must not be published as a blank camera recording')
    monkeypatch.setattr(archive,'materialize_experiment_clips',forbidden)
    e={'experiment_id':'Test','group':{'group_id':'G','continuity_type':'independent',
       'atomic_experiment_ids':['S'],'global_start_ms':0,'global_end_ms':5550,
       'participating_views':['fp','tp'],'first_person_view':'fp','third_person_view':'tp',
       'continuity_reason':'unreviewed contact event','view_timeline':[
           {'start_ms':0,'end_ms':2000,'third_person_view':None},
           {'start_ms':2000,'end_ms':2550,'third_person_view':'tp'},
           {'start_ms':2550,'end_ms':5550,'third_person_view':None}]}}
    _materialize([e],[],{},{},c)
    assert e['outputs']==[] and e['status']=='partial_view_coverage'
    assert e['playback_coverage']['unverified_third_person_ms']==5000
    assert not (tmp_path/'nas').exists()


def test_retirement_preserves_originals_and_recorder_segments(default_config,tmp_path):
    from visioncortex.device_day_multiview import retire_superseded_outputs
    from visioncortex.device_day_contract import atomic_json
    c=deepcopy(default_config)
    c['storage']['archive_root']=str(tmp_path/'nas')
    c['storage']['local_cache_root']=str(tmp_path/'cache')
    name='2026-09-14_fp'
    root=tmp_path/'nas'/name
    source=root/'MetaVideo/13-00-00_13-15-00.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'retained immutable original')
    clips=root/'ProcessedClips/Clips'
    for folder, scope in [('AlignedExperimentWrong','aligned_experiment'),('13-01-00_13-05-00_ExperimentActivity_1','recording_segment')]:
        p=clips/folder
        p.mkdir(parents=True)
        (p/'ExperimentActivity.mp4').write_bytes(b'fixture video')
        atomic_json(p/'ExperimentActivity.json',{'scope':scope,
            'materializer':'visioncortex.archive.materialize_experiment_clips'})
    moved=retire_superseded_outputs(c,name,[])
    assert len(moved)==1 and Path(moved[0]['to']).is_dir()
    assert not (clips/'AlignedExperimentWrong').exists()
    assert (clips/'13-01-00_13-05-00_ExperimentActivity_1/ExperimentActivity.mp4').read_bytes()==b'fixture video'
    assert source.read_bytes()==b'retained immutable original'
    assert retire_superseded_outputs(c,name,[])==[]
