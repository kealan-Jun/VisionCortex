"""A single canonical record plus PascalCase business indexes; no copied evidence."""

import html
import json
import posixpath
from collections import OrderedDict

ROOT = "Browse"
ENTRY = "Readme.html"
FOLDERS = {
    "Bindings": ("BindingEvents", "绑定记录", "谁在什么时间绑定了哪台仪器或哪个场景"),
    "Handoffs": ("DeviceHandoffs", "设备交接", "交出人与接收人、时间及交接凭证"),
    "InstrumentReadings": ("PanelReadings", "设备面板读数", "按仪器核对指标、数值、单位与照片"),
    "VoicePhotos": ("VoicePhotoReadings", "语音拍照读数", "从语音拍照文件追溯同一次测量的读数"),
    "PhotoDrafts": ("MeasurementDrafts", "测量草稿", "连拍合并后的字段、冲突与校正历史"),
    "ExperimentRecords": ("ExperimentRecords", "已确认实验记录", "确认后写入并读回的正式测量"),
    "WorkbenchReadings": ("WorkbenchReadings", "实验台总览", "同一实验台的仪器读数，逐项注明归属"),
    "Photos": ("SourceMaterials", "原始材料", "照片及扫码凭证；保留材料不代表已确认归属"),
}


def raw_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()


def build_readable(records, page):
    views, canonical, groups = {}, OrderedDict(), {key: [] for key in FOLDERS}
    esc = lambda value: html.escape(str(value if value is not None else "未记录"), quote=True)
    for category, data, path in records:
        key = (data["entity"], data["entity_id"])
        if key not in canonical:
            canonical[key] = {
                "data": dict(data),
                "path": path,
                "categories": [],
                "readings": {},
                "measurements": {},
                "evidence": {},
                "targets": [],
            }
        item = canonical[key]
        if category == "InstrumentReadings" and data.get("target") not in item["targets"]:
            item["targets"].append(data.get("target"))
        item["categories"].append(FOLDERS[category][0])
        for reading in data.get("readings", []):
            item["readings"][json.dumps(reading, sort_keys=True, ensure_ascii=False)] = reading
        if data.get("measurement"):
            item["measurements"][data.get("folder_instrument_id") or "Unassigned"] = data[
                "measurement"
            ]
            item["evidence"][data.get("folder_instrument_id") or "Unassigned"] = data.get(
                "measurement_evidence"
            )
        # Category-specific fields live in the index; canonical content remains complete.
        instrument_id = data.get("folder_instrument_id")
        group_key = (key, instrument_id)
        if not any(e["_key"] == group_key for e in groups[category]):
            groups[category].append(
                {
                    "_key": group_key,
                    "entity": key[0],
                    "entity_id": key[1],
                    "local_time": data["local_time"],
                    "operator": data.get("operator"),
                    "camera_id": data.get("camera_id"),
                    "target": data.get("target"),
                    "instrument_id": instrument_id,
                    "page": path,
                    "record": posixpath.join(posixpath.dirname(path), "Record.json"),
                }
            )
    for item in canonical.values():
        data = item["data"]
        for key in ("measurement", "measurement_evidence", "folder_instrument_id"):
            data.pop(key, None)
        data.update(
            schema="field-recognition-record/2",
            derived_view=True,
            categories=list(dict.fromkeys(item["categories"])),
            readings=list(item["readings"].values()),
        )
        if item["measurements"]:
            data["instrument_measurements"] = item["measurements"]
            data["instrument_measurement_evidence"] = item["evidence"]
        if item["targets"]:
            data["target"] = "、".join(t for t in item["targets"] if t)
        path = item["path"]
        directory = posixpath.dirname(path)
        link = lambda target: esc(posixpath.relpath(target, directory))
        views[directory + "/Record.json"] = raw_json(data)
        draft = data.get("measurement_draft") or {}
        fields = draft.get("fields") or data.get("readings", [])
        body = (
            '<p><a href="'
            + link("Readme.html")
            + '">归档首页</a> · <a href="Record.json">完整记录 JSON</a></p>'
        )
        body += (
            "<article><h2>"
            + esc(data.get("target") or "测量记录")
            + "</h2><p>"
            + esc(data["local_time"])
            + "（北京时间） · "
            + esc(data.get("operator"))
            + " · "
            + esc(data.get("camera_id"))
            + "</p>"
        )
        body += (
            "<p>状态："
            + esc(data.get("status"))
            + " · 用途："
            + esc(data.get("record_scope"))
            + "</p>"
        )
        for key, label in [
            ("source_ref", "语音照片来源"),
            ("started_at", "开始时间"),
            ("ended_at", "结束时间"),
        ]:
            if data.get(key):
                body += "<p>" + label + "：" + esc(data[key]) + "</p>"
        if fields:
            body += "<table><tr><th>仪器</th><th>指标</th><th>值</th><th>单位</th><th>原始识别 / 校正</th></tr>"
            for f in fields:
                original = f.get("text") or "、".join(
                    str(c.get("raw_text") or "") for c in f.get("original_candidates", [])
                )
                vals = [
                    (f.get("instrument") or {}).get("name"),
                    f.get("name") or f.get("measurement_name"),
                    f.get("value"),
                    f.get("unit"),
                    str(original) + (" · 已校正" if f.get("corrected") else ""),
                ]
                body += "<tr>" + "".join("<td>" + esc(v) + "</td>" for v in vals) + "</tr>"
            body += "</table>"
        if draft.get("blockers") and draft.get("status") != "confirmed":
            body += "<p>待处理：" + esc("；".join(draft["blockers"])) + "</p>"
        body += "</article>"
        seen = set()
        for name, photo in data.get("photos", {}).items():
            if photo["path"] in seen:
                continue
            seen.add(photo["path"])
            body += (
                '<article><a href="'
                + link(photo["path"])
                + '"><img loading="lazy" src="'
                + link(photo["path"])
                + '" alt="原始证据"></a><p>'
                + esc(name)
                + "</p><small>SHA-256 "
                + esc(photo["sha256"])
                + "</small></article>"
            )
        body += (
            "<details><summary>历史回执与修订</summary><ul>"
            + "".join(
                '<li><a href="' + link(v["receipt"]) + '">版本 ' + str(v["sequence"]) + "</a></li>"
                for v in data["receipt_versions"]
            )
            + "</ul></details>"
        )
        views[path] = page(data["local_time"] + " · " + str(data.get("target") or "测量"), body)
    catalog = {}
    intro = '<p>从下面的业务分类进入，按日期、实验员、相机和仪器查找。每项结果都可打开原图核对。</p><div class="business-grid">'
    for key, (folder, title, description) in FOLDERS.items():
        entries = sorted(groups[key], key=lambda e: e["local_time"], reverse=True)
        catalog[folder] = [{k: v for k, v in e.items() if k != "_key"} for e in entries]
        directory = ROOT + "/" + folder
        listing = (
            '<p><a href="../../Readme.html">归档首页</a></p><p>'
            + esc(description)
            + "</p><p>"
            + str(len(entries))
            + ' 条</p><label>筛选 <input id="search" placeholder="日期 / 实验员 / 仪器"></label><ul id="records">'
        )
        for e in entries:
            label = " · ".join(
                str(e[k] or "未记录") for k in ("local_time", "operator", "camera_id", "target")
            )
            listing += (
                '<li><a href="'
                + esc(posixpath.relpath(e["page"], directory))
                + '">'
                + esc(label)
                + "</a></li>"
            )
        listing += '</ul><script>document.getElementById("search").oninput=function(){const words=this.value.trim().toLowerCase().split(/\\s+/);document.querySelectorAll("#records li").forEach(e=>e.hidden=!words.every(w=>e.textContent.toLowerCase().includes(w)));};</script>'
        views[directory + "/Readme.html"] = page(title, listing)
        intro += (
            '<article><h2><a href="'
            + directory
            + '/Readme.html">'
            + title
            + "</a></h2><p>"
            + esc(description)
            + "</p><code>"
            + directory
            + "</code><p>"
            + str(len(entries))
            + " 条</p></article>"
        )
    intro += '</div><details><summary>数据结构与留存原则</summary><ul><li>Browse：分类索引，只保存记录引用。</li><li>Records：每个实体只有一份最新记录，含读数、单位和证据链接。</li><li>Objects：按 SHA-256 去重的照片原件。</li><li>Receipts：不可覆盖的历史版本，是溯源依据。</li><li>Integrity：完整性检查报告。</li></ul><p>一次连拍的正式测量只在 ExperimentRecords 中形成一条记录。分类交叉引用不会重复生成测量或照片。</p><a href="Audit.html">全量版本索引</a></details>'
    views["Readme.html"] = page("现场识别归档", intro)
    views["Browse/Catalog.json"] = raw_json(
        {"schema": "field-recognition-folders/2", "categories": catalog}
    )
    # Historical entry URLs remain valid, without a second copy of the navigation.
    for alias in ("Browse/Readme.html", "00_归档导航.html"):
        target = posixpath.relpath("Readme.html", posixpath.dirname(alias) or ".")
        views[alias] = page(
            "归档入口",
            '<p><a href="'
            + target
            + '">打开归档首页</a></p><meta http-equiv="refresh" content="0;url='
            + target
            + '">',
        )
    return views
