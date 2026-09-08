"""Derived transcript postings. Original text, occurrence IDs and evidence stay intact."""
from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

from . import speech_worker

CONTROL = "JSON-Config-Files/speech_search.json"
ALIASES = (
    ("移液", "加液", "加样", "加入试剂", "pipette", "pipetting"),
    ("离心", "centrifuge", "centrifugation"),
    ("混匀", "混勻", "涡旋", "渦旋", "vortex", "mixing"),
    ("拍照", "拍攝", "拍摄", "take photo"),
    ("温度", "溫度", "temperature"),
)


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and unicodedata.category(c)[0] not in {"P", "Z"})


def source_hint(row: dict) -> dict:
    if row.get("nearby_device_playback"):
        return {"kind": "possible_device_playback", "label": "疑似设备播报",
                "basis": "附近有设备播放记录；时间邻近不能确认声源", "confirmed": False}
    if any(term in normalize(row["text"]) for terms in ALIASES[:3] for term in terms):
        return {"kind": "possible_operation_narration", "label": "疑似操作口述",
                "basis": "文字包含操作术语；未识别说话人，也不确认动作", "confirmed": False}
    return {"kind": "unknown", "label": "声源未知", "basis": "没有可确认的声源依据", "confirmed": False}


def _grams(text: str) -> set[str]:
    return set(text) | {text[i:i+2] for i in range(len(text)-1)}


def _verified_sources(root: Path, payload: dict) -> list[tuple[dict, dict, dict, Path]]:
    files = []
    for source in payload.get("sources", []):
        for chunk in source.get("chunks", []):
            spec = chunk["files"]["aligned-transcript.json"]
            path = root / spec["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("转写产物路径无效")
            if speech_worker.file_record(path) != {k: spec[k] for k in ("size", "sha256")}:
                raise ValueError("转写产物完整性检查未通过")
            files.append((source, chunk, spec, path))
    return files


def _build(root: Path, payload: dict, identity: str) -> dict:
    rows, postings = [], {}
    for source, chunk, spec, path in _verified_sources(root, payload):
        for row in speech_worker.read_json(path, 32*1024*1024)["segments"]:
            text = normalize(row["text"])
            item = {**row, "chunk_id": chunk["id"], "view_id": source["view_id"],
                    "reference_id": f"{chunk['id']}:{row['id']}", "normalized_text": text,
                    "phrase_id": hashlib.sha256((chunk["id"] + "\0" + text).encode()).hexdigest()[:24],
                    "source_hint": source_hint(row), "evidence_kind": "spoken_mention",
                    "physical_action_confirmation": False,
                    "transcript_path": spec["path"], "transcript_sha256": spec["sha256"]}
            for gram in _grams(text):
                postings.setdefault(gram, []).append(len(rows))
            rows.append(item)
            if len(rows) > 100000:
                raise ValueError("录音搜索索引超出单实验容量")
    return {"schema_version": "visioncortex-speech-search/1", "speech_sha256": identity,
            "implementation_sha256": speech_worker.sha256(Path(__file__)),
            "rows": rows, "postings": postings}


def build(root: Path) -> dict:
    path = root / "JSON-Config-Files/speech.json"
    identity = speech_worker.sha256(path)
    result = _build(root, speech_worker.read_json(path, 16*1024*1024), identity)
    if identity != speech_worker.sha256(path):
        raise ValueError("建立索引期间转写版本变化")
    speech_worker.atomic_json(root / CONTROL, result)
    speech_worker.atomic_json(root / "JSON-Config-Files/speech_search_receipt.json", {
        "speech_sha256": identity, "index": speech_worker.file_record(root / CONTROL)})
    return {"status": "completed", "segment_count": len(result["rows"]), "speech_sha256": identity}


@lru_cache(maxsize=4)
def _load(root_string: str, identity: str, index_identity: str, implementation: str) -> dict:
    root = Path(root_string)
    if index_identity:
        result = speech_worker.read_json(root / CONTROL, 64*1024*1024)
        if result.get("speech_sha256") != identity:
            raise ValueError("搜索索引引用了其他转写版本")
        if result.get("implementation_sha256") == implementation:
            return result
    # Older archives can be searched without mutating a published release.
    return _build(root, speech_worker.read_json(root / "JSON-Config-Files/speech.json", 16*1024*1024), identity)


def search(root: Path, payload: dict, query: str = "", offset: int = 0, limit: int = 100,
           chunk: str | None = None, *, fold: bool = False, phrase: str | None = None,
           aliases: bool = True, hint: str | None = None) -> dict:
    _verified_sources(root, payload)  # Integrity is checked even on index cache hits.
    path = root / "JSON-Config-Files/speech.json"
    identity = speech_worker.sha256(path)
    index = root / CONTROL
    if index.is_file():
        receipt = speech_worker.read_json(root / "JSON-Config-Files/speech_search_receipt.json")
        if receipt.get("speech_sha256") != identity or receipt.get("index") != speech_worker.file_record(index):
            raise ValueError("搜索索引完整性检查未通过")
    result = _load(str(root.resolve()), identity, speech_worker.sha256(index) if index.is_file() else "",
                   speech_worker.sha256(Path(__file__)))
    term = normalize(query)
    terms = {term}
    if aliases and term:
        for family in ALIASES:
            if term in {normalize(value) for value in family}:
                terms.update(normalize(value) for value in family)
    candidates = set()
    for value in terms:
        sets = [set(result["postings"].get(gram, [])) for gram in _grams(value)]
        candidates.update(set.intersection(*sets) if sets else range(len(result["rows"])))
    matches = []
    for i in sorted(candidates):
        row = result["rows"][i]
        matched = sorted(value for value in terms if value in row["normalized_text"])
        if not matched or (chunk and chunk != row["chunk_id"]) or (phrase and phrase != row["phrase_id"]) or (hint and hint != row["source_hint"]["kind"]):
            continue
        matches.append({**row, "matched_terms": matched})
    count = len(matches)
    if fold:
        groups = {}
        for row in matches:
            if row["phrase_id"] not in groups:
                groups[row["phrase_id"]] = {**row, "occurrence_count": 0}
            groups[row["phrase_id"]]["occurrence_count"] += 1
        matches = list(groups.values())
    if identity != speech_worker.sha256(path):
        raise ValueError("查询期间转写版本变化")
    return copy.deepcopy({"segments": matches[offset:offset+limit], "total": len(matches),
                          "occurrence_total": count, "folded": fold,
                          "next_offset": offset+limit if offset+limit < len(matches) else None,
                          "search": {"kind": "inverted_character_postings", "speech_sha256": identity,
                                     "expanded_terms": sorted(terms), "aliases_enabled": aliases,
                                     "evidence_kind": "spoken_mention"}})


def evaluate(root: Path, reference: dict) -> dict:
    """CER on explicitly human reviewed references; never promote ASR as truth."""
    path = root / "JSON-Config-Files/speech.json"
    identity = speech_worker.sha256(path)
    if reference.get("speech_sha256") != identity or reference.get("human_reviewed") is not True or not str(reference.get("reviewer") or "").strip():
        raise ValueError("需要匹配本次转写版本的人工校对基线和校对人")
    items = reference.get("segments")
    if not isinstance(items, list) or not 1 <= len(items) <= 500:
        raise ValueError("每次评测需提供 1–500 条人工校对文字")
    payload = speech_worker.read_json(path, 16*1024*1024)
    rows = {row["reference_id"]: row for row in _build(root, payload, identity)["rows"]}
    seen, scores = set(), []
    for item in items:
        key = item.get("reference_id")
        text = item.get("text")
        if key not in rows or key in seen or not isinstance(text, str) or len(text) > 2000:
            raise ValueError("基线包含重复、未知引用或无效文字")
        seen.add(key)
        expected, actual = normalize(text), normalize(rows[key]["text"])
        if len(actual) > 2000:
            raise ValueError("转写过长，请按原始字幕片段评测")
        previous = list(range(len(actual)+1))
        for i, char in enumerate(expected, 1):
            current = [i]
            for j, other in enumerate(actual, 1):
                current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(char != other)))
            previous = current
        scores.append({"reference_id": key, "reference_characters": len(expected), "edit_distance": previous[-1]})
    total = sum(item["reference_characters"] for item in scores)
    if not total:
        raise ValueError("基线至少需要一个有效字符")
    return {"schema_version": "visioncortex-speech-evaluation/1", "status": "PARTIAL_EVIDENCE",
            "speech_sha256": identity, "reference_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "reviewer": reference["reviewer"], "character_error_rate": sum(item["edit_distance"] for item in scores)/total,
            "normalization": "NFKC, casefold, whitespace/punctuation removed; no alias substitution",
            "evaluated_segments": len(scores), "total_segments": len(rows), "segments": scores,
            "limitation": "仅适用于提交的人工校对样本；声源、时间同步和动作准确率未验证"}
