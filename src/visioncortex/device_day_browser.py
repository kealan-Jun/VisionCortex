"""A readable view of the five fixed device/day output directories."""
from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote

from .device_day_contract import DIRECTORIES, read_json, safe_child
from .device_day_reports import clock


def render_archive(root: Path, data: dict) -> str:
    name = data["archive"]
    def href(path):
        if path.lower().endswith('.mp4'):
            return f'/device-days/{quote(name)}/watch/{quote(path, safe="/")}'
        return f'/api/device-days/{quote(name)}/files/{quote(path, safe="/")}'
    def link(path, label):
        return f'<a href="{href(path)}">{html.escape(label)}</a>'
    sources = []
    for record in data.get("recordings", []):
        metadata = []
        for source in record.get("sources", []):
            if source["kind"] not in {"video", "audio_audio"}:
                metadata.append(source["retained"])
                continue
            ref = source["retained"]
            sources.append(f'<li>{link(ref["path"], Path(ref["path"]).name)} · {ref["size_bytes"] / 1024**2:.1f} MB</li>')
        if record.get("capture_manifest"):
            manifest = record["capture_manifest"]
            catalog = read_json(safe_child(root, manifest["path"]))
            media_paths = {s["retained"]["path"] for s in record.get("sources", []) if s["kind"] in {"video", "audio_audio"}}
            metadata = [f["retained"] for f in catalog.get("files", []) if f["retained"]["path"] not in media_paths]
            metadata.append(manifest)
        if metadata:
            sources.append('<li><details><summary>对应采集数据：CSV、时间戳、标定与日志</summary><ul>' +
                           ''.join(f'<li>{link(ref["path"], Path(ref["path"]).name)}</li>' for ref in metadata) + '</ul></details></li>')
    cards = []
    for segment in data.get("segments", []):
        def frames(rows, label):
            return ''.join(f'<figure><a href="{href(frame["path"])}"><img loading="lazy" src="{href(frame["path"])}" alt="{label}"></a>'
                           f'<figcaption>{label} · 原片 {frame["local_ms"] / 1000:.2f} 秒'
                           f'<p>{html.escape(frame.get("understanding_text", ""))}</p></figcaption></figure>' for frame in rows)
        video = segment.get("video") or segment["source_ref"]
        start = 0 if segment.get("video") else segment["start_ms"] / 1000
        end = (segment["end_ms"] - segment["start_ms"]) / 1000 if segment.get("video") else segment["end_ms"] / 1000
        media = f'{href(video["path"])}#t={start},{end}'
        cards.append(f'<article><h3>{clock(segment["start_us"])}–{clock(segment["end_us"])} · {html.escape(segment["activity_label"])}</h3>'
                     f'<p><a href="{media}">{"打开切分实验视频" if segment.get("video") else "打开原片对应区间"}</a> · '
                     f'{link(segment["json_path"], "本片段 JSON")}</p>'
                     f'<p>动作关键帧：{len(segment["key_frames"])}</p><div class="frames">{frames(segment["key_frames"], "动作关键帧")}</div>'
                     f'<details><summary>场景采样帧（{len(segment["scene_frames"])}）</summary><div class="frames">{frames(segment["scene_frames"], "场景采样帧")}</div></details></article>')
    comments_path = root / DIRECTORIES[4] / "Comment.jsonl"
    comments = []
    if comments_path.is_file():
        import json
        comments = [json.loads(line) for line in comments_path.read_text().splitlines() if line.strip()]
    protocol_path = root / DIRECTORIES[4] / "Protocol.json"
    protocol = read_json(protocol_path).get("text", "") if protocol_path.is_file() else "当天尚未提供 protocol。"
    notes = ''.join(f'<li>{clock(row["start_us"])} · {html.escape(row["text"])}</li>' for row in comments)
    headings = ("原视频与原录音", "切分片段与帧索引", "多模态理解", "实验室日报", "comment 与录音识别")
    ids = ("media", "clips", "understanding", "report", "comment")
    content = [f'<ul>{"".join(sources) or "<li>等待原片留存。</li>"}</ul>',
               f'<p>{link(DIRECTORIES[1] + "/Index.json", "下载总索引 JSON")}</p>' + (''.join(cards) or '<p>等待 YOLO 处理产出。</p>'),
               f'<p>{link(DIRECTORIES[2] + "/UnderstandingReport.html", "查看按时间整理的理解报告")} · {link(DIRECTORIES[2] + "/Understanding.json", "理解 JSON")}</p>',
               f'<p>{link(DIRECTORIES[3] + "/LaboratoryDailyReport.html", "打开当天整合日报")} · {link(DIRECTORIES[3] + "/LaboratoryDailyReport.json", "日报 JSON")}</p>',
               f'<p><a href="/device-days/{quote(name)}/stt">查看录音播放器与 STT 识别文本</a></p><ul>{notes or "<li>当天尚无人的 comment。</li>"}</ul><p>{html.escape(protocol)}</p>']
    sections = ''.join(f'<section id="{anchor}"><h2>{html.escape(folder)}</h2><p>{heading}</p>{body}</section>'
                       for anchor, folder, heading, body in zip(ids, DIRECTORIES, headings, content, strict=True))
    nav = ' · '.join(f'<a href="#{anchor}">{label}</a>' for anchor, label in zip(ids, headings, strict=True))
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
           '<title>设备日实验归档</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:36px auto;padding:20px;color:#23303b}a{color:#006b9d}section{border-top:1px solid #ddd;margin:24px 0;padding:16px 0}article{background:#f6f8fa;padding:16px;margin:16px 0;border-radius:10px}h1{overflow-wrap:anywhere}figure{margin:0;max-width:320px}img{width:100%;border-radius:6px}.frames{display:flex;gap:16px;flex-wrap:wrap}summary{cursor:pointer}</style>' \
           f'<a href="/device-days">返回设备日列表</a><h1>{html.escape(name)}</h1><nav>{nav}</nav>' \
           '<p>有活动、无活动与无关区间均保留来源。场景采样帧与五类动作关键帧分别显示；模型结果的判断依据和不确定项保留在索引与报告中。</p>' + sections + '</html>'
