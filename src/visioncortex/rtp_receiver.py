"""Persistent, source-filtered RTP/Opus receiver; local durable spool only.

This receiver never changes recorder NAS files or claims arrival time is capture
UTC. Existing NAS-finalized audio remains the authoritative ASR input.
"""
import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import time
import uuid
from zoneinfo import ZoneInfo

from .device_day_contract import atomic_json
from .rtp_opus import OggOpus, Opus, delta, parse, sender_reports


class Segment:
    def __init__(self, root, sender, packet, channels):
        date = datetime.fromtimestamp(packet.received_us/1e6, ZoneInfo('Asia/Shanghai'))
        self.folder = root/sender/date.strftime('%Y-%m-%d')/(date.strftime('%H-%M-%S.%f')+'_'+f'{packet.ssrc:08x}')
        self.folder.mkdir(parents=True, exist_ok=False)
        self.audio = (self.folder/'Audio.opus.partial').open('xb')
        self.timing = (self.folder/'Packets.jsonl.partial').open('x')
        self.ogg = OggOpus(self.audio, packet.ssrc, channels)
        self.first, self.last = packet, packet
        self.samples, self.count, self.missing, self.duplicates, self.channels = 0, 0, 0, 0, channels
        self.last_flush = time.monotonic()
        self.mapping = None
        atomic_json(self.folder/'Metadata.json', self.metadata('receiving'))

    def metadata(self, status, reason=None):
        mapping = self.mapping
        origin = (mapping['sender_unix_us']+round(delta(self.first.timestamp,mapping['rtp_timestamp'],32)*1e6/48000)
                  if mapping else None)
        return {'schema_version':'visioncortex-rtp-opus/1','status':status,'close_reason':reason,
                'source_ssrc':self.first.ssrc,'payload_type':self.first.payload_type,'channels':self.channels,
                'rtp_clock_rate':48000,'first_sequence':self.first.sequence,'last_sequence':self.last.sequence,
                'first_rtp_timestamp':self.first.timestamp,'last_rtp_timestamp':self.last.timestamp,
                'first_received_us':self.first.received_us,'last_received_us':self.last.received_us,
                'first_sender_reported_us':origin,'clock_mapping':mapping,
                'time_basis':'rtcp_sender_report' if mapping else 'receiver_arrival_only',
                'capture_clock_verified':False,'global_alignment':'PARTIAL_EVIDENCE' if mapping else 'NOT_PROVEN',
                'packets':self.count,'missing_packets':self.missing,'duplicate_or_late_packets':self.duplicates,
                'payload_samples':self.samples,'payload_duration_seconds':self.samples/48000,
                'audio':'Audio.opus','packet_timing':'Packets.jsonl',
                'time_alignment_note':'Use packet timings for missing-packet gaps; concatenated audio is not proof of continuous capture.',
                'original_opus_payload_preserved':True,'transcription_source':'existing_recorder_finalized_NAS_audio',
                'nas_archive_modified':False}

    def append(self, packet, samples, missing):
        self.count += 1
        self.missing += missing
        self.timing.write(json.dumps({'sequence':packet.sequence,'rtp_timestamp':packet.timestamp,
            'received_us':packet.received_us,'payload_offset_samples':self.samples,
            'payload_samples':samples,'missing_before':missing})+'\n')
        self.samples += samples
        self.ogg.write(packet.payload, self.samples)
        self.last = packet
        if time.monotonic()-self.last_flush >= 1:
            self.flush()

    def flush(self):
        for handle in (self.audio,self.timing):
            handle.flush()
            os.fsync(handle.fileno())
        self.last_flush = time.monotonic()

    def close(self, reason):
        self.ogg.finish()
        self.flush()
        self.audio.close(); self.timing.close()
        (self.folder/'Audio.opus.partial').rename(self.folder/'Audio.opus')
        (self.folder/'Packets.jsonl.partial').rename(self.folder/'Packets.jsonl')
        metadata = self.metadata('completed',reason)
        metadata['audio_sha256'] = hashlib.sha256((self.folder/'Audio.opus').read_bytes()).hexdigest()
        atomic_json(self.folder/'Metadata.json',metadata)
        atomic_json(self.folder/'Ready.json',{'ready':True,'metadata':'Metadata.json',
                                            'audio_sha256':metadata['audio_sha256'],'closed_at_us':time.time_ns()//1000})
        return self.folder


class Receiver:
    def __init__(self, settings):
        self.settings=settings
        self.root=Path(settings['output_root'])
        if not self.root.is_absolute() or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',settings['sender_id']):
            raise ValueError('Receiver requires an absolute spool path and safe sender identifier')
        self.root.mkdir(parents=True,exist_ok=True)
        self.opus=Opus()
        self.streams={}
        self.reports={}
        self.stop=False
        self.stats={'packets_received':0,'rejected_source':0,'invalid_packets':0,'completed_segments':0,
                    'duplicate_or_late_packets':0,'missing_packets':0}
        self.last_status=0
        self.last_packet=None
        self.started_us=time.time_ns()//1000

    def accept(self, packet, now):
        samples,channels=self.opus.info(packet.payload)
        state=self.streams.get(packet.ssrc)
        if state is None:
            if len(self.streams)>=8:
                raise ValueError('Concurrent SSRC limit reached')
            state=self.streams[packet.ssrc]={'next':packet.sequence,'pending':{},'segment':None,'last_seen':now}
        number=state['next']+delta(packet.sequence,state['next']&65535,16)
        if number < state['next'] or number in state['pending']:
            self.stats['duplicate_or_late_packets']+=1
            if state['segment']:
                state['segment'].duplicates+=1
            return
        if number-state['next']>3000:
            self.drain(state,now,force=True)
            if state['segment']:
                self.finish(state,'sequence_discontinuity')
            state['next']=number
        state['last_seen']=now
        state['pending'][number]=(packet,samples,channels,now)
        self.drain(state,now)

    def drain(self,state,now,force=False):
        while state['pending']:
            number=min(state['pending'])
            packet,samples,channels,arrived=state['pending'][number]
            if number!=state['next'] and not force and now-arrived < .12 and len(state['pending'])<64:
                break
            missing=number-state['next']
            self.stats['missing_packets']+=missing
            segment=state['segment']
            if segment and (channels!=segment.channels or delta(packet.timestamp,segment.last.timestamp,32)<0):
                self.finish(state,'stream_discontinuity')
                segment=None
            if segment is None:
                segment=state['segment']=Segment(self.root,self.settings['sender_id'],packet,channels)
            segment.mapping=self.reports.get(packet.ssrc)
            segment.append(packet,samples,missing)
            state['next']=number+1
            del state['pending'][number]
            if segment.samples/48000>=self.settings.get('segment_seconds',30):
                self.finish(state,'duration_limit')

    def finish(self,state,reason):
        folder=state['segment'].close(reason)
        state['segment']=None
        self.stats['completed_segments']+=1
        self.stats['last_completed_path']=str(folder)

    def tick(self,now):
        for ssrc,state in list(self.streams.items()):
            self.drain(state,now,force=self.stop)
            if self.stop or now-state['last_seen']>=self.settings.get('idle_seconds',2):
                if state['segment']:
                    self.finish(state,'service_stop' if self.stop else 'sender_idle')
                del self.streams[ssrc]
        if self.stop or now-self.last_status>=1:
            atomic_json(self.root/'Status.json',{'schema_version':'visioncortex-rtp-receiver-status/1',
                'status':'stopped' if self.stop else 'receiving' if self.streams else 'listening',
                'pid':os.getpid(),'started_us':self.started_us,'updated_us':time.time_ns()//1000,
                'bind_address':self.settings['bind_address'],'rtp_port':self.settings['rtp_port'],
                'rtcp_port':self.settings.get('rtcp_port'),'allowed_source':self.settings['allowed_source'],
                'sender_id':self.settings['sender_id'],'last_packet_received_us':self.last_packet,
                'rtcp_sender_report_count':len(self.reports),'capture_clock_verified':False,
                'active_ssrcs':list(self.streams),'stats':self.stats})
            self.last_status=now

    def run(self):
        sockets=[]
        try:
            with selectors.DefaultSelector() as selector:
                for kind in ('rtp','rtcp'):
                    port=self.settings.get(kind+'_port')
                    if port is None:
                        continue
                    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                    sockets.append(sock)
                    sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,2*1024*1024)
                    sock.bind((self.settings['bind_address'],port))
                    sock.setblocking(False)
                    selector.register(sock,selectors.EVENT_READ,kind)
                self.tick(time.monotonic())
                while not self.stop:
                    for key,_ in selector.select(.05):
                        for _ in range(128):
                            try:
                                data,address=key.fileobj.recvfrom(65535)
                            except BlockingIOError:
                                break
                            if address[0]!=self.settings['allowed_source']:
                                self.stats['rejected_source']+=1
                                continue
                            received=time.time_ns()//1000
                            try:
                                if key.data=='rtcp' or len(data)>1 and 192<=data[1]<=223:
                                    for report in sender_reports(data):
                                        if abs(report['sender_unix_us']-received)>86400*1e6:
                                            raise ValueError('RTCP sender clock outside one-day plausibility bound')
                                        self.reports[report['ssrc']]=report
                                else:
                                    packet=parse(data,received,self.settings.get('payload_type',111))
                                    self.accept(packet,time.monotonic())
                                    self.stats['packets_received']+=1
                                    self.last_packet=received
                            except ValueError:
                                self.stats['invalid_packets']+=1
                    self.tick(time.monotonic())
        finally:
            self.stop=True
            self.tick(time.monotonic())
            for sock in sockets:
                sock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    receiver=Receiver(settings)
    def stop(*_):
        receiver.stop=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    logging.basicConfig(level=logging.INFO)
    receiver.run()


if __name__=='__main__':
    main()
