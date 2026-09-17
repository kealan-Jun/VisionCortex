"""Bounded RTP/Opus parsing and loss-visible Ogg packet preservation (RFC 7587)."""
from dataclasses import dataclass
import ctypes
import ctypes.util
import struct


@dataclass(frozen=True)
class Packet:
    sequence: int
    timestamp: int
    ssrc: int
    payload_type: int
    payload: bytes
    received_us: int


def parse(data, received_us, payload_type=111):
    if len(data) < 12:
        raise ValueError('Truncated RTP header')
    first, second, seq, stamp, ssrc = struct.unpack('!BBHII', data[:12])
    if first >> 6 != 2 or second & 127 != payload_type:
        raise ValueError('Unexpected RTP version or payload type')
    offset = 12 + 4 * (first & 15)
    if first & 16:
        if len(data) < offset + 4:
            raise ValueError('Truncated RTP extension')
        words = struct.unpack('!H', data[offset+2:offset+4])[0]
        offset += 4 + 4 * words
    end = len(data)
    if first & 32:
        padding = data[-1]
        if not padding or padding > end - offset:
            raise ValueError('Invalid RTP padding')
        end -= padding
    if offset >= end:
        raise ValueError('Empty or truncated RTP payload')
    return Packet(seq, stamp, ssrc, second & 127, data[offset:end], received_us)


def delta(value, reference, bits):
    return (value-reference+(1 << (bits-1))) % (1 << bits) - (1 << (bits-1))


def sender_reports(data):
    """Return RTCP SR mappings; the sender clock is reported, not verified UTC."""
    result, offset = [], 0
    while offset + 4 <= len(data):
        first, kind, words = struct.unpack('!BBH', data[offset:offset+4])
        length = (words+1)*4
        if first >> 6 != 2 or offset + length > len(data):
            raise ValueError('Invalid compound RTCP packet')
        if kind == 200:
            if length < 28:
                raise ValueError('Truncated RTCP sender report')
            ssrc, seconds, fraction, stamp = struct.unpack('!IIII', data[offset+4:offset+20])
            epoch_us = (seconds-2208988800)*1_000_000 + round(fraction*1_000_000/(1 << 32))
            result.append({'ssrc':ssrc, 'rtp_timestamp':stamp, 'sender_unix_us':epoch_us})
        offset += length
    if offset != len(data):
        raise ValueError('Truncated RTCP tail')
    return result


class Opus:
    def __init__(self):
        library = ctypes.util.find_library('opus')
        if not library:
            raise RuntimeError('System libopus is required')
        self.library = ctypes.CDLL(library)
        self.library.opus_packet_get_nb_samples.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self.library.opus_packet_get_nb_samples.restype = ctypes.c_int
        self.library.opus_packet_get_nb_channels.argtypes = [ctypes.c_char_p]
        self.library.opus_packet_get_nb_channels.restype = ctypes.c_int

    def info(self, payload):
        count = self.library.opus_packet_get_nb_samples(payload, len(payload), 48000)
        channels = self.library.opus_packet_get_nb_channels(payload)
        if not 0 < count <= 5760 or channels not in (1, 2):
            raise ValueError('Invalid Opus packet')
        return count, channels


def _crc_table():
    values = []
    for n in range(256):
        crc = n << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04c11db7 if crc & 0x80000000 else 0)) & 0xffffffff
        values.append(crc)
    return values


_CRC = _crc_table()


class OggOpus:
    """Remux compressed packets without transcoding; timing gaps stay explicit."""
    def __init__(self, handle, serial, channels):
        self.handle, self.serial, self.page = handle, serial, 0
        self.granule = 0
        self.write(b'OpusHead'+struct.pack('<BBHIhB', 1, channels, 0, 48000, 0, 0), 0, 2)
        vendor = b'VisionCortex RTP receiver'
        self.write(b'OpusTags'+struct.pack('<I',len(vendor))+vendor+struct.pack('<I',0), 0)

    def write(self, payload, granule, flags=0):
        lacing = bytes([255]*(len(payload)//255)+[len(payload)%255])
        if len(lacing) > 255:
            raise ValueError('Opus packet exceeds one Ogg page')
        page = bytearray(b'OggS'+struct.pack('<BBQIIIB',0,flags,granule,self.serial,self.page,0,len(lacing))+lacing+payload)
        crc = 0
        for byte in page:
            crc = ((crc << 8) ^ _CRC[((crc >> 24) & 255) ^ byte]) & 0xffffffff
        page[22:26] = struct.pack('<I',crc)
        self.handle.write(page)
        self.page += 1
        self.granule = granule

    def finish(self):
        # An empty end-of-stream page does not invent an Opus audio packet.
        page = bytearray(b'OggS'+struct.pack('<BBQIIIB',0,4,self.granule,self.serial,self.page,0,0))
        crc = 0
        for byte in page:
            crc = ((crc << 8) ^ _CRC[((crc >> 24) & 255) ^ byte]) & 0xffffffff
        page[22:26] = struct.pack('<I',crc)
        self.handle.write(page)
