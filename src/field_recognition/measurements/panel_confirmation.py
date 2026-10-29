"""Track per-display observations and stability using distinct video frames."""

from collections import deque
import copy
import re

from field_recognition.measurements.readout_rules import READOUT


def value_key(lines):
    return tuple(re.sub(r"\s+", "", line["text"]).replace(",", ".") for line in lines)


def overlap(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


class PanelConfirmation:
    window_seconds = 1.6

    def __init__(self):
        self.reset()

    def reset(self):
        self.scope = None
        self.windows = {}
        self.saved = {}

    @staticmethod
    def key(region):
        return region["instrument_id"], region["panel_id"]

    def observe(self, regions, token, observed_at, now, scope):
        if self.scope != scope:
            self.reset()
            self.scope = scope
        visible = {self.key(r) for r in regions}
        self.windows = {k: v for k, v in self.windows.items() if k in visible}
        observed = []
        for region in regions:
            key = self.key(region)
            previous = self.windows.get(key)
            if previous and overlap(previous["bbox"], region["bbox"]) < 0.25:
                previous = None
                self.saved.pop(key, None)
            history = previous["history"] if previous else deque(maxlen=3)
            while history and now - history[0]["time"] > self.window_seconds:
                history.popleft()
            lines = [
                l
                for l in region["local_ocr"].get("lines", [])
                if not l.get("quality_issue")
                and not l.get("quality_issues")
                and READOUT.fullmatch(l.get("text", ""))
            ]
            value = value_key(lines)
            fresh = not any(h["token"] == token for h in history)
            if fresh:
                history.append(
                    {"token": token, "time": now, "observed_at": observed_at, "value": value}
                )
            self.windows[key] = {"bbox": region["bbox"], "history": history}
            supporters = [h for h in history if value and h["value"] == value]
            receipt = {
                "status": "confirmed" if len(supporters) >= 2 else "pending",
                "rule": "2_of_3_distinct_frames",
                "votes": len(supporters),
                "samples": len(history),
                "window_seconds": self.window_seconds,
                "observation_eligible": bool(lines) and fresh,
                "stability_required_for_recording": False,
                "elapsed_seconds": round(now - supporters[0]["time"], 3) if supporters else 0,
                "supporting_frames": [
                    {
                        "frame_key": list(h["token"]),
                        "observed_at": h["observed_at"],
                        "texts": list(h["value"]),
                    }
                    for h in supporters
                ],
            }
            region["temporal_confirmation"] = receipt
            if receipt["observation_eligible"]:
                for line in lines:
                    result = copy.deepcopy(line)
                    result["temporal_confirmation"] = copy.deepcopy(receipt)
                    observed.append(result)
        return observed

    def changed(self, region):
        return region.get("temporal_confirmation", {}).get(
            "observation_eligible", False
        ) and value_key(region["local_ocr"].get("lines", [])) != self.saved.get(
            self.key(region), {}
        ).get("value")

    def can_save(self, region, now):
        return now - self.saved.get(self.key(region), {}).get("time", -1000) >= 2

    def periodic_due(self, region, now, interval):
        previous = self.saved.get(self.key(region))
        return bool(
            previous
            and region.get("temporal_confirmation", {}).get("observation_eligible", False)
            and now - previous["time"] >= interval
        )

    def mark_saved(self, regions, now=0):
        for region in regions:
            if region.get("temporal_confirmation", {}).get("observation_eligible", False):
                self.saved[self.key(region)] = {
                    "value": value_key(region["local_ocr"].get("lines", [])),
                    "time": now,
                }
