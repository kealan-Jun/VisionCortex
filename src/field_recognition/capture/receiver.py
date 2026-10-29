"""Single selected camera from receiver GWHP main stream; bounded CPU decoding."""

from __future__ import annotations

import json
import queue
import struct
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction

import av

BASE = struct.Struct("<4sHHIIIIQQ")
# Frame freshness limits what may be scanned. A service stall is a separate,
# sustained failure, using the same interval as the stream read timeout.
FRAME_MAX_AGE_SECONDS = 1
SERVICE_STALL_SECONDS = 8
RECONNECTABLE_FAILURES = {
    "camera_stream_error",
    "camera_decode_error",
    "camera_frames_stale",
    "receiver_status_unavailable",
    "camera_not_found",
    "camera_session_changed",
    "camera_service_offline",
}


def read_exact(stream, count):
    parts = bytearray()
    while len(parts) < count:
        piece = stream.read(count - len(parts))
        if not piece:
            raise EOFError("GWHP stream closed")
        parts.extend(piece)
    return bytes(parts)


@dataclass
class Packet:
    payload: bytes
    flags: int
    width: int
    height: int
    timestamp_us: int
    sequence: int
    global_timestamp_us: int
    version: int

    @property
    def sync_valid(self):
        return self.version >= 2 and bool(self.flags & 4) and self.global_timestamp_us > 0

    @property
    def pts(self):
        return self.global_timestamp_us if self.sync_valid else self.timestamp_us


def read_packet(stream):
    magic, version, header_size, size, flags, width, height, timestamp, sequence = BASE.unpack(
        read_exact(stream, BASE.size)
    )
    if magic != b"GWHP" or version not in (1, 2):
        raise ValueError("Invalid GWHP magic/version")
    if not 40 <= header_size <= 256 or (version == 2 and header_size < 48):
        raise ValueError("Invalid GWHP header size")
    if (
        not 0 < size <= 16 * 1024 * 1024
        or not 0 < width * height <= 16_000_000
        or min(width, height) == 0
    ):
        raise ValueError("Invalid GWHP payload/dimensions")
    extension = read_exact(stream, header_size - 40)
    global_us = struct.unpack_from("<Q", extension)[0] if version == 2 else 0
    return Packet(
        read_exact(stream, size), flags, width, height, timestamp, sequence, global_us, version
    )


class ReceiverCamera:
    def __init__(self, receiver, target):
        self.receiver = receiver.rstrip("/")
        self.target = target
        self.stop = threading.Event()
        self.lock = threading.Lock()
        # H.264 packets may arrive in bursts. Two slots discarded reference frames
        # during normal jitter, freezing the preview until the next keyframe.
        self.packets = queue.Queue(maxsize=16)
        self.latest = None
        self.response = None
        self.started = False
        self.on_service_status = None
        self.service_lock = threading.RLock()
        self.invalid_before = 0.0
        self.frames_expected_since = time.monotonic()
        self.info = {
            "configured": True,
            "id": target,
            "mode": "gwhp_main",
            "status": "not_started",
            "decoder": "PyAV CPU / single-camera",
            "receiver": self.receiver,
            "reconnect_count": 0,
            "sequence_gaps": 0,
            "decode_errors": 0,
            "dropped_packets": 0,
            "decoded_frames": 0,
            "online_cameras": [],
        }

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
        self.reader = threading.Thread(target=self._read_loop, name="gwhp-reader", daemon=True)
        self.decoder = threading.Thread(
            target=self._decode_loop, name="gwhp-cpu-decoder", daemon=True
        )
        self.reader.start()
        self.decoder.start()
        self.monitor = threading.Thread(target=self._status_loop, name="gwhp-status", daemon=True)
        self.monitor.start()

    def _observe_service(self, observation, *, available=True):
        with self.service_lock:
            self._apply_service(observation, available=available)

    def _apply_service(self, observation, *, available):
        observation = observation | {"observed_at": datetime.now(timezone.utc).isoformat()}
        with self.lock:
            previous = self.info.get("service_status") or {}
            changed = any(
                previous.get(key) not in (None, 0)
                and observation.get(key) not in (None, 0)
                and previous[key] != observation[key]
                for key in ("media_session_id", "device_boot_id")
            )
            rebooted = bool(
                previous.get("device_boot_id")
                and observation.get("device_boot_id")
                and previous["device_boot_id"] != observation["device_boot_id"]
            )
            if changed and observation["online"] and not rebooted:
                # The frame currently cached may belong to the old ingress.
                # Recovery must wait for a newly decoded frame after this edge.
                observation = observation | {
                    "online": False,
                    "failure_reason": "camera_session_changed",
                    "frame_verified": False,
                }
            observation["reconnectable"] = (
                observation["online"] or observation.get("failure_reason") in RECONNECTABLE_FAILURES
            ) and not rebooted
            for key in ("media_session_id", "device_boot_id"):
                if key not in observation and previous.get(key):
                    observation[key] = previous[key]
            if not observation["online"] or previous.get("online") is False or changed:
                self.invalid_before = time.monotonic()
                self.latest = None
                self.info["service_status_available"] = False
            self.info["service_status"] = observation
        # Do not hold the frame lock while the scanner invalidates its database session.
        if self.on_service_status:
            self.on_service_status(observation)
        with self.lock:
            self.info["service_status_available"] = available

    def _status_loop(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self.stop.is_set():
            try:
                with opener.open(self.receiver + "/api/status", timeout=5) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("Receiver status too large")
                status = json.loads(raw)
                if status.get("receiver_admin_stale"):
                    raise ValueError("Receiver status is stale")
                matches = [
                    c
                    for c in status.get("cameras", [])
                    if c.get("camera_key") == self.target
                    or f"{c.get('sender_id')}_{c.get('camera_id')}" == self.target
                ]
                if len(matches) == 1:
                    camera = matches[0]
                    if not all(
                        isinstance(camera.get(k), bool) for k in ("status_live", "media_live")
                    ):
                        raise ValueError("Receiver service liveness is unavailable")
                    # This is the receiver's RGB ingress session, not its recording session.
                    observation = {
                        "online": bool(camera.get("status_live") and camera.get("media_live")),
                        "media_session_id": camera.get("rgb_ingress_session_id"),
                    }
                    # Optional explicit device-origin boot identity. Never substitute
                    # recording/ingress session counters or Receiver process IDs.
                    boot = camera.get("device_boot_id")
                    if isinstance(boot, str) and 0 < len(boot.strip()) <= 128:
                        observation["device_boot_id"] = boot
                    if not observation["online"]:
                        observation["failure_reason"] = "camera_service_offline"
                    self._observe_receiver_service(observation)
                else:
                    self._observe_service(
                        {"online": False, "failure_reason": "camera_not_found"}, available=False
                    )
            except Exception:
                # Suspend capture authorization while a bounded reconnect window
                # retains occupancy; this does not prove physical power loss.
                self._observe_service(
                    {"online": False, "failure_reason": "receiver_status_unavailable"},
                    available=False,
                )
            self.stop.wait(1)

    def _observe_receiver_service(self, observation):
        with self.service_lock:
            with self.lock:
                frame_at = self.latest[2] if self.latest else self.frames_expected_since
                age = time.monotonic() - frame_at
                previous = self.info.get("service_status") or {}
                fresh = self.latest is not None and age <= FRAME_MAX_AGE_SECONDS
                observation = observation | {"frame_verified": fresh}
                if observation["online"]:
                    if previous.get("online") is False and not fresh:
                        # A healthy heartbeat alone cannot restore failed RGB
                        # acquisition. Wait for an actual newly decoded frame.
                        observation = observation | {
                            "online": False,
                            "failure_reason": previous.get("failure_reason", "camera_frames_stale"),
                        }
                    elif age > SERVICE_STALL_SECONDS:
                        observation = observation | {
                            "online": False,
                            "failure_reason": "camera_frames_stale",
                        }
            self._apply_service(observation, available=True)

    def close(self):
        self.stop.set()
        response = self.response
        if response:
            response.close()

    def snapshot(self):
        with self.lock:
            data = dict(self.info)
            if self.latest:
                data["frame_age_ms"] = round((time.monotonic() - self.latest[2]) * 1000)
            if data.get("service_status", {}).get("online") is False:
                data["status"] = "service_unavailable"
            elif data["status"] == "streaming" and (
                not self.latest or data["frame_age_ms"] > FRAME_MAX_AGE_SECONDS * 1000
            ):
                data["status"] = "waiting_frame"
            data["encoded_queue_depth"] = self.packets.qsize()
            return data

    def frame(self):
        with self.lock:
            if self.info.get("service_status", {}).get("online") is False:
                raise ValueError("设备采集服务已离线，等待重新启动")
            if not self.latest or time.monotonic() - self.latest[2] > FRAME_MAX_AGE_SECONDS:
                raise ValueError("尚未取得一秒内的新鲜主码流画面，请检查接收端和设备在线状态")
            image, metadata, received = self.latest
            return image.copy(), dict(metadata) | {
                "frame_age_ms": round((time.monotonic() - received) * 1000)
            }

    def _read_loop(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        generation, attempt = 0, 0
        while not self.stop.is_set():
            try:
                with self.lock:
                    self.info["status"] = "discovering"
                with opener.open(self.receiver + "/api/status", timeout=5) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                    if len(raw) > 4 * 1024 * 1024:
                        raise ValueError("Receiver status too large")
                    status = json.loads(raw)
                cameras = [
                    c for c in status.get("cameras", []) if c.get("online") and c.get("media_live")
                ]
                with self.lock:
                    self.info["online_cameras"] = [
                        {
                            "sender_id": c["sender_id"],
                            "camera_id": c["camera_id"],
                            "camera_key": c.get("camera_key"),
                        }
                        for c in cameras
                    ]
                matches = [
                    c
                    for c in cameras
                    if c.get("camera_key") == self.target
                    or f"{c.get('sender_id')}_{c.get('camera_id')}" == self.target
                ]
                if len(matches) != 1:
                    with self.lock:
                        self.info.update(
                            status="camera_offline", message="接收端未发现这台在线挂脖相机"
                        )
                    self.stop.wait(2)
                    continue
                camera = matches[0]
                query = urllib.parse.urlencode(
                    {
                        "sender_id": camera["sender_id"],
                        "camera_id": camera["camera_id"],
                        "quality": "main",
                        "metadata": "global",
                    }
                )
                self.response = opener.open(
                    self.receiver + "/api/preview/rgb-h264-frames?" + query,
                    timeout=SERVICE_STALL_SECONDS,
                )
                if self.response.headers.get("X-GWV3-Rgb-Stream") != "main":
                    raise ValueError("Receiver did not return main stream")
                generation += 1
                need_key = True
                previous_sequence = None
                with self.lock:
                    self.info.update(
                        status="waiting_keyframe",
                        sender_id=camera["sender_id"],
                        camera_id=camera["camera_id"],
                        message=None,
                        error_type=None,
                    )
                while not self.stop.is_set():
                    packet = read_packet(self.response)
                    if previous_sequence is not None and packet.sequence != previous_sequence + 1:
                        need_key = True
                        generation += 1
                        with self.lock:
                            self.info["sequence_gaps"] += 1
                    previous_sequence = packet.sequence
                    if need_key and packet.flags & 3 != 3:
                        continue
                    if self.packets.full():
                        while True:
                            try:
                                self.packets.get_nowait()
                            except queue.Empty:
                                break
                        generation += 1
                        need_key = True
                        with self.lock:
                            self.info["dropped_packets"] += 1
                        if packet.flags & 3 != 3:
                            continue
                    self.packets.put_nowait((generation, packet, time.monotonic(), camera))
                    need_key = False
                    attempt = 0
            except Exception as exc:
                with self.lock:
                    self.latest = None
                    self.info.update(
                        status="reconnecting",
                        error_type=type(exc).__name__,
                        message="接收端暂不可用，自动重试；不会显示旧帧为实时画面",
                    )
                    self.info["reconnect_count"] += 1
                self._observe_service(
                    {"online": False, "failure_reason": "camera_stream_error"}, available=False
                )
                self.stop.wait([0.5, 1, 2, 5][min(attempt, 3)])
                attempt += 1
            finally:
                if self.response:
                    self.response.close()
                    self.response = None

    def _decode_loop(self):
        decoder, context_key, last_pts, metadata = None, None, None, {}
        while not self.stop.is_set():
            try:
                generation, packet, received, camera = self.packets.get(timeout=0.5)
            except queue.Empty:
                continue
            if time.monotonic() - received > 0.25 or received <= self.invalid_before:
                # Never turn the burst allowance into a delayed playback queue.
                decoder, context_key, metadata = None, None, {}
                with self.lock:
                    self.info["dropped_packets"] += 1
                continue
            key = (generation, packet.width, packet.height)
            if key != context_key or (last_pts is not None and packet.pts < last_pts - 1_000_000):
                decoder, metadata = None, {}
                if packet.flags & 3 != 3:
                    continue
                decoder = av.CodecContext.create("h264", "r")
                decoder.thread_count = 1
                context_key = key
            last_pts = packet.pts
            if decoder is None:
                continue
            metadata[packet.pts] = (packet, received)
            while len(metadata) > 120:
                metadata.pop(next(iter(metadata)))
            try:
                encoded = av.Packet(packet.payload)
                encoded.pts = packet.pts
                encoded.time_base = Fraction(1, 1_000_000)
                for frame in decoder.decode(encoded):
                    match = metadata.pop(frame.pts, None)
                    if match is None:
                        continue
                    source, received = match
                    if time.monotonic() - received > FRAME_MAX_AGE_SECONDS:
                        continue
                    if (frame.width, frame.height) != (source.width, source.height):
                        raise ValueError("Decoded frame dimensions differ from GWHP header")
                    full_range = camera.get("rgb_h264_full_range")
                    if isinstance(full_range, bool):
                        frame.color_range = 2 if full_range else 1
                    pixels = frame.to_ndarray(format="bgr24")
                    record = {
                        "sender_id": camera["sender_id"],
                        "camera_id": camera["camera_id"],
                        "sequence": source.sequence,
                        "timestamp_us": source.timestamp_us,
                        "global_timestamp_us": source.global_timestamp_us,
                        "clock_sync_valid": source.sync_valid,
                        "effective_timestamp_us": source.pts,
                        "gwhp_version": source.version,
                        "stream": "main",
                        "width": frame.width,
                        "height": frame.height,
                        "rgb_h264_full_range": full_range,
                        "decoder": "PyAV CPU",
                    }
                    record["receive_to_decode_ms"] = round((time.monotonic() - received) * 1000, 1)
                    with self.lock:
                        if received <= self.invalid_before:
                            continue
                        record["decoded_frame_id"] = self.info["decoded_frames"] + 1
                        self.latest = (pixels, record, received)
                        self.frames_expected_since = received
                        self.info.update(
                            status="streaming",
                            last_sequence=source.sequence,
                            clock_sync_valid=source.sync_valid,
                            width=frame.width,
                            height=frame.height,
                            receive_to_decode_ms=record["receive_to_decode_ms"],
                        )
                        self.info["decoded_frames"] += 1
            except Exception:
                decoder = None
                context_key = None
                with self.lock:
                    self.info["decode_errors"] += 1
                self._observe_service(
                    {"online": False, "failure_reason": "camera_decode_error"}, available=False
                )
