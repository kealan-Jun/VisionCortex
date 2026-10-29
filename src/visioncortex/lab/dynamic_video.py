"""Opt-in original-pixel QR/registration/control-change producer, same Core.

Reference positioning is explicit, source-pinned configuration. Every usable
following transform is fitted from actual stable pixels. No hand/action model,
new physical identity, service or quantity fact is created here.
"""
from fractions import Fraction
import hashlib
import json
import math
import re
import time

from .domain import Conflict, fingerprint
from .dynamic_vision import frame_hash, interaction_pixels, polygon, project, qr_occurrences
from .panel_vision import libraries
from .temporal_registration import TemporalRegistrar, RegistrationConfig, normalized_box


SCHEMA='vc-lab-dynamic-video/1'


class DynamicVideoRejected(ValueError):
    pass


def _wire(value):
    if isinstance(value,dict): return {k:_wire(v) for k,v in value.items()}
    if isinstance(value,list): return [_wire(v) for v in value]
    if isinstance(value,float):
        if not math.isfinite(value): raise DynamicVideoRejected('Nonfinite pixel registration')
        return 0.0 if value==0 else value
    return value


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):
        raise DynamicVideoRejected('Explicit original-pixel SHA256 required')
    return value


def _point(meta):
    if type(meta.get('pts')) is not int or type(meta.get('time_base')) is not str or type(meta.get('stream_origin')) is not str:
        raise DynamicVideoRejected('Exact original PTS/time base/origin required')
    if abs(meta['pts'])>2**63 or len(meta['time_base'])>128 or len(meta['stream_origin'])>128:
        raise DynamicVideoRejected('Bounded exact source PTS representation required')
    base=Fraction(meta['time_base']);origin=Fraction(meta['stream_origin'])
    if base<=0: raise DynamicVideoRejected('Positive source time base required')
    return meta['pts'],str(base),str(origin)


def _bounds(poly):
    return [min(p[0] for p in poly),min(p[1] for p in poly),max(p[0] for p in poly),max(p[1] for p in poly)]


def _exclusions(stable_boxes,geometry):
    """Explicit positive stable region union -> registrar's exclusion interface."""
    for box in stable_boxes: normalized_box(box,(1000,1000))
    xs=sorted({0.,1.,*(float(v) for b in stable_boxes for v in [b[0],b[2]])})
    ys=sorted({0.,1.,*(float(v) for b in stable_boxes for v in [b[1],b[3]])})
    excluded=[]
    for x1,x2 in zip(xs,xs[1:]):
        for y1,y2 in zip(ys,ys[1:]):
            x,y=(x1+x2)/2,(y1+y2)/2
            if not any(float(b[0])<=x<=float(b[2]) and float(b[1])<=y<=float(b[3]) for b in stable_boxes):
                excluded.append([x1,y1,x2,y2])
    excluded.extend(_bounds(p) for p in [geometry['panel_polygon'],geometry['qr_region'],*geometry['control_polygons']])
    return excluded


class DynamicVideo:
    def __init__(self,context,media_id,source_session_key,source_ref,config):
        initialized=time.monotonic()
        if not isinstance(config,dict) or config.get('schema_version')!=SCHEMA or config.get('enabled') is not True:
            raise DynamicVideoRejected('Explicit opt-in dynamic pixel producer required')
        if type(config.get('revision')) is not str or not 1<=len(config['revision'])<=80:
            raise DynamicVideoRejected('Versioned dynamic producer configuration required')
        if len(json.dumps(config,allow_nan=False).encode())>64*1024:
            raise DynamicVideoRejected('Dynamic producer config exceeds64KiB')
        self.config=dict(config);self.context=context;self.media_id=str(media_id);self.session=source_session_key
        self.ref=dict(source_ref);self.config_sha=fingerprint(config)
        _sha(source_ref.get('sha256'))
        if type(source_ref.get('media_version')) is not int or source_ref['media_version']<1:
            raise DynamicVideoRejected('Exact original media version required')
        for key,default,lo,hi in [('max_frames',600,1,600),('batch_frames',5,1,32),('max_buffer_bytes',32*1024**2,1024,64*1024**2),('max_seconds',300,1,300)]:
            value=config.get(key,default)
            if type(value) is not int or not lo<=value<=hi: raise DynamicVideoRejected('Bounded producer '+key+' required')
            setattr(self,key,value)
        panels=config.get('panels',[])
        if not isinstance(panels,list) or len(panels)>8: raise DynamicVideoRejected('At most8 registered panels required')
        if len({p.get('track_id') for p in panels})!=len(panels): raise DynamicVideoRejected('Unique panel track IDs required')
        self.inputs=context.video_inputs(media_id,source_session_key,panels)
        if self.inputs['source_sha256']!=source_ref.get('sha256') or self.inputs['media_version']!=source_ref.get('media_version'):
            raise Conflict('Producer source SHA/version differs from actual original')
        self.start=Fraction(source_ref['start_seconds']);self.end=Fraction(source_ref['end_seconds'])
        if not 0<=self.start<self.end<=Fraction(self.inputs['duration_seconds']): raise DynamicVideoRejected('Producer range escaped source')
        self.states=[]
        for spec,link in zip(panels,self.inputs['panel_links']):
            if type(spec.get('track_id')) is not str or not 1<=len(spec['track_id'])<=120: raise DynamicVideoRejected('Explicit track ID required')
            reference=spec.get('reference_frame',{})
            point=_point(reference);offset=point[0]*Fraction(point[1])-Fraction(point[2])
            if not self.start<=offset<self.end: raise DynamicVideoRejected('Reference frame must be in this exact bounded source interval')
            _sha(reference.get('upright_rgb_sha256'))
            quad=polygon(reference.get('device_quad_normalized'))
            if len(quad)!=4 or any(not 0<=v<=1 for p in quad for v in p): raise DynamicVideoRejected('Source-pinned upright device quadrilateral required')
            stable=spec.get('stable_boxes');background=spec.get('background_boxes')
            if not isinstance(stable,list) or not 1<=len(stable)<=4 or not isinstance(background,list) or not 1<=len(background)<=4:
                raise DynamicVideoRejected('Explicit bounded stable/background regions required')
            for box in background: normalized_box(box,(1000,1000))
            exclusions=_exclusions(stable,link['geometry'])
            self.states.append(dict(spec=spec,link=link,point=point,exclusions=exclusions,registrar=None,last_matrix=None,reference_to_roi=None,roi_to_frame=None,shape=None,reference_image=None,previous_matrix=None,failed=False))
        self.pending=[];self.previous=None;self.last_offset=None;self.committed=None;self.started=initialized
        self.count=0;self.qr_calls=0;self.db_commits=0;self.peak_buffer_bytes=0;self.last_revision=None
        self.timings=dict(setup_seconds=time.monotonic()-initialized,qr_seconds=0.,registration_seconds=0.,pixel_seconds=0.,database_seconds=0.)
        self.failures=0;self.discarded_pending=0;self.closed=False
        self.frame_shape=None;self.processing_rules=None

    @property
    def buffer_bytes(self):
        arrays=sum((s['registrar'].memory_bytes+s['reference_image'].nbytes) for s in self.states if s['registrar'] is not None)
        return arrays+(self.previous.nbytes if self.previous is not None else 0)+len(json.dumps(self.pending,allow_nan=False).encode())

    def _budget(self):
        self.peak_buffer_bytes=max(self.peak_buffer_bytes,self.buffer_bytes)
        if self.buffer_bytes>self.max_buffer_bytes: raise DynamicVideoRejected('Producer retained buffer exceeds byte bound')
        if time.monotonic()-self.started>self.max_seconds: raise DynamicVideoRejected('Producer wall-time bound reached')

    def _registration(self,state,image,point,sha):
        cv,np=libraries();h,w=image.shape[:2]
        if state['registrar'] is None:
            if point!=state['point']: return None,dict(accepted=False,reason='reference_frame_not_yet_observed')
            if sha!=state['spec']['reference_frame']['upright_rgb_sha256']: raise Conflict('Actual reference-frame pixels differ from configured SHA')
            corners=np.asarray(state['spec']['reference_frame']['device_quad_normalized'],np.float32)*[w-1,h-1]
            rw=max(32,math.ceil(max(np.linalg.norm(corners[1]-corners[0]),np.linalg.norm(corners[2]-corners[3]))))
            rh=max(32,math.ceil(max(np.linalg.norm(corners[3]-corners[0]),np.linalg.norm(corners[2]-corners[1]))))
            if rw*rh>262144: raise DynamicVideoRejected('Registered device reference exceeds262144 native-plane pixels')
            reference_to_roi=cv.getPerspectiveTransform(corners.astype(np.float32),np.float32([[0,0],[rw-1,0],[rw-1,rh-1],[0,rh-1]]))
            reference=cv.warpPerspective(image,reference_to_roi,(rw,rh))
            registrar=TemporalRegistrar(reference,state['exclusions'],RegistrationConfig(mode=state['spec'].get('registration_mode','rigid')))
            state.update(registrar=registrar,reference_to_roi=reference_to_roi,shape=(rw,rh),reference_image=reference,
                         roi_to_frame=np.linalg.inv(reference_to_roi),canonical_to_roi=np.diag([rw-1.,rh-1.,1.]))
        cropped=cv.warpPerspective(image,state['reference_to_roi'],state['shape'])
        _,receipt=state['registrar'].register(cropped)
        if not receipt['accepted']:
            state['failed']=True
            return None,receipt
        fitted=np.linalg.inv(state['reference_to_roi'])@np.linalg.inv(np.asarray(receipt['matrix_current_to_reference']))@state['canonical_to_roi']
        fitted/=fitted[2,2]
        if not np.isfinite(fitted).all(): raise DynamicVideoRejected('Nonfinite actual registration transform')
        corners=project([[0,0],[1,0],[1,1],[0,1]],fitted)
        if any(x<0 or y<0 or x>=w or y>=h for x,y in corners):
            state['failed']=True
            return None,dict(receipt,accepted=False,reason='fitted_device_escaped_original_frame')
        status='reacquired' if state['failed'] else 'reliable'
        state['failed']=False;state['last_matrix']=_wire(fitted.tolist())
        return status,dict(receipt,canonical_to_upright=state['last_matrix'],reference_frame=state['spec']['reference_frame'],reference_coordinates='upright_original_frame',canonical_coordinates='verified_device_unit_plane',reference_positioning='explicit_source_pinned_configuration_not_autodetected')

    def observe(self,upright_bgr,frame_ref):
        if self.closed: raise DynamicVideoRejected('Producer already closed')
        try:
            return self._observe(upright_bgr,frame_ref)
        except Exception:
            self.discarded_pending+=len(self.pending);self.pending=[];self.closed=True
            raise

    def _observe(self,image,meta):
        cv,np=libraries();self._budget()
        if self.frame_shape is not None and image.shape!=self.frame_shape: raise DynamicVideoRejected('Original frame dimensions changed')
        if image.shape[0]*image.shape[1]>4000000: raise DynamicVideoRejected('Full upright frame exceeds4M pixel bound; no resize fallback')
        sha=frame_hash(image);point=_point(meta);offset=point[0]*Fraction(point[1])-Fraction(point[2])
        if not self.start<=offset<self.end or self.last_offset is not None and offset<=self.last_offset:
            raise DynamicVideoRejected('Producer requires increasing original PTS in exact source range')
        if Fraction(meta['media_offset'])!=offset or meta.get('source_sha256')!=self.ref['sha256']:
            raise Conflict('Producer original frame source/offset mismatch')
        if meta.get('upright_rgb_sha256') is not None and meta['upright_rgb_sha256']!=sha: raise Conflict('Upright pixel SHA mismatch')
        matrix=np.asarray(meta.get('source_to_upright'),dtype=float);h,w=image.shape[:2]
        identity=np.eye(3);rotate=np.array([[-1,0,w-1],[0,-1,h-1],[0,0,1]],float)
        if matrix.shape!=(3,3) or not (np.array_equal(matrix,identity) or np.array_equal(matrix,rotate)):
            raise DynamicVideoRejected('Only explicit native 0/180 original-to-upright mapping accepted')
        original=image if np.array_equal(matrix,identity) else np.ascontiguousarray(image[::-1,::-1])
        raw_bgr=hashlib.sha256(original.tobytes()).hexdigest();raw_rgb=frame_hash(original)
        if _sha(meta.get('original_bgr_sha256'))!=raw_bgr: raise Conflict('Original raw BGR pixels differ from source-frame hash')
        if self.count>=self.max_frames: raise DynamicVideoRejected('Producer source frame bound exceeded')
        tick=time.monotonic();decoded=qr_occurrences(image);self.qr_calls+=1;self.timings['qr_seconds']+=time.monotonic()-tick
        tracks=[]
        for state in self.states:
            tick=time.monotonic();status,registration=self._registration(state,image,point,sha);self.timings['registration_seconds']+=time.monotonic()-tick
            if status is None:
                self.failures+=1
                if state['last_matrix'] is None: continue
                status='lost'
            track=dict(track_id=state['spec']['track_id'],link_id=state['link']['id'],layout_revision=state['link']['layout_revision'],
                       canonical_to_frame=state['last_matrix'],status=status,registration=_wire(registration),
                       geometry_role='actual_fit' if status!='lost' else 'last_successful_fit_for_reset_only')
            if status!='lost' and self.previous is not None and state['previous_matrix'] is not None:
                previous_points=np.asarray(project([[0,0],[1,0],[1,1],[0,1]],state['previous_matrix']))
                current_points=np.asarray(project([[0,0],[1,0],[1,1],[0,1]],state['last_matrix']))
                displacement=float(np.linalg.norm(current_points-previous_points,axis=1).max())
                track['interaction_camera_motion_pixels']=displacement
                if displacement<=.75:
                    controls=[_wire(project(p,state['last_matrix'])) for p in state['link']['geometry']['control_polygons']]
                    backgrounds=[_wire(project([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],state['last_matrix'])) for b in state['spec']['background_boxes']]
                    tick=time.monotonic()
                    if controls:
                        track['interaction']=interaction_pixels(self.previous,image,control_polygons=controls,background_polygons=backgrounds)
                    self.timings['pixel_seconds']+=time.monotonic()-tick
                else: track['interaction_reason']='camera_motion_gate_no_operation_support'
            state['previous_matrix']=state['last_matrix'] if status!='lost' else None
            tracks.append(track)
        refs=[r['id'] for r in self.inputs['evidence_refs'] if r['start_seconds'] is not None and r['end_seconds'] is not None and Fraction(r['start_seconds'])<=offset<=Fraction(r['end_seconds'])]
        if not refs: raise Conflict('Actual source frame lacks original evidence coverage')
        event=dict(schema_version='vc-lab-dynamic-frame/1',frame_key='dv:'+fingerprint(dict(config=self.config_sha,point=point,pixels=sha)),
                   source_pts=point[0],time_base=point[1],stream_origin=point[2],frame_rgb_sha256=sha,
                   original_bgr_sha256=raw_bgr,original_rgb_sha256=raw_rgb,upright_rgb_sha256=sha,source_to_upright=_wire(matrix.tolist()),
                   frame_coordinate_basis='upright_original_frame; original_hashes_separately_pinned',
                   source_sha256=self.ref['sha256'],media_version=self.ref['media_version'],evidence_ids=refs,tracks=tracks,
                   qr_occurrences=decoded['occurrences'],pixel_producer=dict(schema_version=SCHEMA,revision=self.config['revision'],config_sha256=self.config_sha,
                   semantics='untrusted_pixel_observations_only_no_hand_model_or_confirmed_operation'))
        self.pending.append(_wire(event));self.previous=image.copy();self.frame_shape=image.shape;self.count+=1;self.last_offset=offset;self._budget()
        return self.flush() if len(self.pending)>=self.batch_frames else self._marker('buffered')

    def _marker(self,status):
        return dict(status=status,pending_frames=len(self.pending),committed_through_offset=str(self.committed) if self.committed is not None else None,revision=self.last_revision)

    def flush(self):
        if self.closed: raise DynamicVideoRejected('Producer closed after failure; pending evidence not committed')
        try:
            return self._flush()
        except Exception:
            self.discarded_pending+=len(self.pending);self.pending=[];self.closed=True
            raise

    def _flush(self):
        self._budget()
        if not self.pending: return self._marker('committed')
        batch=self.pending;tick=time.monotonic()
        result=self.context.append_batch('dv-batch:'+fingerprint(dict(media=self.media_id,session=self.session,config=self.config_sha,frames=[e['frame_key'] for e in batch])),self.media_id,self.session,batch)
        self.timings['database_seconds']+=time.monotonic()-tick
        if result['status']=='needs_attention':
            self.closed=True;self.discarded_pending+=len(batch);self.pending=[]
            raise DynamicVideoRejected('Dynamic context refused bounded source projection')
        self.db_commits+=1;self.pending=[];self.committed=self.last_offset;self.last_revision=result['revision'];self.processing_rules=result.get('snapshot',{}).get('processing_rules');self._budget()
        return self._marker('committed')

    def receipt(self):
        return dict(schema_version=SCHEMA,config_sha256=self.config_sha,input_frame_count=self.count,qr_calls=self.qr_calls,db_commits=self.db_commits,
                    processing_rules=self.processing_rules,
                    buffer_bytes=self.buffer_bytes,peak_buffer_bytes=self.peak_buffer_bytes,pending_frames=len(self.pending),discarded_pending_frames=self.discarded_pending,
                    registration_failures=self.failures,actual_timings=dict(self.timings,wall_seconds=time.monotonic()-self.started),
                    original_pixels_preserved=True,registered_panel_count=len(self.states),new_identities_created=0,confirmed_operations=0,
                    material_usage_or_quantity_writes=0,real_transfer_accuracy='NOT_ACCEPTED',limits=dict(max_frames=self.max_frames,batch_frames=self.batch_frames,max_buffer_bytes=self.max_buffer_bytes,max_seconds=self.max_seconds))
