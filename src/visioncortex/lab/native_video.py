"""Complete-clip input: original decoded pixels, every frame, bounded resources."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

from .subprocesses import run_bounded


class VideoCapacityExceeded(ValueError):
    pass


def command(args,*,limit=64*1024,timeout=180):
    return run_bounded(args,'',env={'PATH':os.environ.get('PATH','/usr/bin:/bin')},timeout=timeout,max_bytes=limit)


def probe(path,count=False):
    args=['ffprobe','-v','error','-select_streams','v:0']+(['-count_frames'] if count else [])
    d=json.loads(command(args+['-show_entries','stream=width,height,nb_frames,nb_read_frames,avg_frame_rate,duration:format=duration','-of','json',str(path)]))
    if len(d.get('streams',[]))!=1:raise ValueError('One valid video stream required')
    stream=d['streams'][0]
    return dict(stream,duration=stream.get('duration',d['format']['duration']))


def frame_hashes(path,*,crop=None):
    args=['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:v:0','-an']
    if crop:args+=['-vf',f'crop={crop[0]}:{crop[1]}:0:0']
    raw=command(args+['-f','framemd5','-'],limit=8*1024**2)
    return [line.rsplit(',',1)[1].strip() for line in raw.splitlines() if line and not line.startswith('#')]


def prepare_complete(path,ref,model,folder,bound):
    if ref.get('input_unit')!='complete_retained_clip':
        raise ValueError('Complete input requires a registered retained-clip task')
    if model.get('video_input_mode')!='complete_retained_clip' or model.get('spatial_policy')!='preserve_source_pixels':
        raise ValueError('Explicit native complete-clip provider required')
    spec=model['vision_token_geometry'];p=probe(path)
    if float(ref['start_seconds'])!=0 or abs(float(p['duration'])-float(ref['end_seconds']))>max(.05,1.1/float(Fraction(p['avg_frame_rate']))):
        raise ValueError('Complete clip must cover the entire original playback duration')
    stride=spec['patch_size']*spec['merge_size'];temporal=spec['temporal_patch_size']
    width,height=p['width'],p['height']
    if width*height>16_000_000:raise VideoCapacityExceeded('Native frame exceeds pixel bound')
    pw,ph=math.ceil(width/stride)*stride,math.ceil(height/stride)*stride
    try:n=int(p['nb_frames'])
    except (KeyError,ValueError):p=probe(path,True);n=int(p['nb_read_frames'])
    def tokens(frames):
        groups=math.ceil(frames/temporal)
        return groups*((pw//stride)*(ph//stride)+16)+model.get('context_text_reserve',1024)+model.get('max_tokens',2048)
    if model['max_model_len']<tokens(n)<model['max_model_len']*1.3:
        p=probe(path,True);n=int(p['nb_read_frames'])
    if tokens(n)>model['max_model_len']:
        raise VideoCapacityExceeded('Complete native clip exceeds the configured token budget')
    transform=dict(kind='complete_retained_clip',temporal_sampling=False,spatial_resize_and_lossy_encoding=False,
        original_size=[width,height],padded_size=[pw,ph],decoded_frame_count=n,
        input_token_upper_estimate=tokens(n),model_context_limit=model['max_model_len'],
        token_count_basis='declared_checkpoint_geometry_plus_bounded_timestamp_text_output_reserve',
        server_decoder_verification='explicit_deployment_configuration_not_per_request_attestation')
    if (pw,ph)==(width,height):
        if path.stat().st_size>=bound:raise VideoCapacityExceeded('Complete input exceeds transport byte budget')
        return path.read_bytes(),dict(transform,pixel_verification='original_encoded_file_bytes_unchanged')
    output=Path(folder)/('native-'+hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.mp4')
    try:
        command(['ffmpeg','-v','error','-nostdin','-y','-i',str(path),'-map','0:v:0','-an','-vf',f'pad={pw}:{ph}:0:0:black,setpts=PTS-STARTPTS',
                 '-vsync','0','-c:v','libx264','-crf','0','-preset','fast','-threads','2','-fs',str(bound),str(output)])
        if output.stat().st_size>=bound-1024:raise VideoCapacityExceeded('Complete native padding exceeds byte budget; no truncation accepted')
        original=frame_hashes(path)
        if not original or original!=frame_hashes(output,crop=(width,height)):
            raise ValueError('Native padded input changed decoded source pixels/frames')
        if tokens(len(original))>model['max_model_len']:raise VideoCapacityExceeded('Decoded native input exceeds context')
        transform.update(decoded_frame_count=len(original),pixel_verification='every_decoded_frame_equal_after_removing_black_padding',
                         frame_hash_list_sha256=hashlib.sha256('\n'.join(original).encode()).hexdigest())
        return output.read_bytes(),transform
    finally:output.unlink(missing_ok=True)
