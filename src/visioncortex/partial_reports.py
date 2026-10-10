"""Readable stage reports, without producing a formal acceptance manifest."""

from __future__ import annotations

import hashlib
import html
import io
import re
from pathlib import Path
from urllib.parse import quote

from .report_brand import BRAND_COLORS, brand_logo_data_url
from .report_presentations import _register_fonts, professional_cover_flowables


def clock(value):
    if not isinstance(value, (float, int)):
        return "未记录"
    seconds = max(0, int(value / 1000))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def paragraphs(value):
    return [
        part.strip() for part in re.split(r"[\n；]+", str(value or "")) if part.strip()
    ]


def readable_uncertainties(values):
    """Present the practical limit; detailed diagnostic text remains in JSON."""
    result = []
    for value in values:
        note = str(value)
        if re.search(r"decode offset|\bPTS\b|clip_timeline|nominal", note, re.I):
            note = "部分画面的时间对应仍需核对，请结合连续视频查看。"
        elif "调用失败" in note:
            note = "本段步骤说明未完整生成，已保存的操作记录仍需结合视频核对。"
        elif re.search(r"CV.*(?:实例|关联)|same_action_pair|pair_evidence_status", note):
            note = "两个视角中的操作对象尚未确认一致。"
        else:
            note = re.sub(r"[（(][^()（）]*\bCV\b[^()（）]*[）)]", "", note)
            for token, label in {"container_state_change": "容器开闭变化", "hand_object_contact": "手与物体接触",
                                 "gloved_hand": "戴手套的手", "bottle_cap": "瓶盖", "pipette": "移液器"}.items():
                note = note.replace(token, label)
        if note and note not in result:
            result.append(note)
    return result


def report_groups(export):
    for group in export.get("experiment_groups", []):
        model = group.get("model_understanding") or {}
        yield group, model.get("steps") or []


def retained_report_visuals(
    root: Path, groups: list[dict], events: list[dict]
) -> list[dict]:
    by_id = {event.get("event_id"): event for event in events}
    result = []
    for group in groups:
        refs = [
            ref
            for step in (group.get("model_understanding") or {}).get("steps", [])
            for ref in step.get("supporting_event_ids", [])
        ]
        for event_id in dict.fromkeys(refs):
            frame = (by_id.get(event_id, {}).get("key_frames") or {}).get(
                "aligned_first_third"
            )
            if not isinstance(frame, str):
                continue
            path = root / frame
            if (
                not path.resolve().is_relative_to(root.resolve())
                or not frame.startswith("Key-Materials/")
                or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
                or not path.is_file()
                or path.stat().st_size > 8 * 1024 * 1024
            ):
                continue
            result.append(
                {
                    "group_id": group.get("group_id"),
                    "event_id": event_id,
                    "path": frame,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "evidence_classification": "PARTIAL_EVIDENCE",
                }
            )
            break
    return result


def render_stage_reports(root: Path, export: dict, receipt: dict) -> dict:
    """Reuse the product typography and palette; read no video or model data."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )

    regular, bold = _register_fonts()
    folder = root / "Partial-Results"
    folder.mkdir(parents=True, exist_ok=True)
    identifier = export.get("experiment_id") or root.name
    brand, muted = (
        colors.HexColor(BRAND_COLORS["brand"]),
        colors.HexColor(BRAND_COLORS["text_muted"]),
    )
    styles = {
        "title": ParagraphStyle(
            "stage-title",
            fontName=bold,
            fontSize=24,
            leading=32,
            textColor=brand,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "stage-h1",
            keepWithNext=True,
            fontName=bold,
            fontSize=17,
            leading=24,
            textColor=brand,
            spaceBefore=12,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "stage-h2",
            keepWithNext=True,
            fontName=bold,
            fontSize=11,
            leading=17,
            textColor=brand,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "stage-body",
            wordWrap="CJK",
            fontName=regular,
            fontSize=9.5,
            leading=16,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "stage-small",
            fontName=regular,
            fontSize=8,
            leading=12,
            textColor=muted,
            wordWrap="CJK",
            spaceAfter=5,
        ),
    }

    def p(value, style="body"):
        return Paragraph(
            html.escape(str("未记录" if value is None or value == "" else value)),
            styles[style],
        )

    def table(rows, widths):
        t = Table(
            [[p(c, "small") for c in row] for row in rows],
            colWidths=widths,
            repeatRows=1,
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf5f1")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dce7e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return t

    groups = list(report_groups(export))
    notes = "本报告整理已保存的操作、时间和证据引用。整次分析的质量检查尚未通过；内容仍需核对，不表示全部实验操作已识别或实验已经结束。"
    story = professional_cover_flowables(
        [
            ["实验档案", identifier],
            ["报告类型", "阶段成果 · 待核对"],
            ["质量状态", "未通过完整质量检查；未正式发布"],
            [
                "记录范围",
                f"{len(groups)} 个片段 / {sum(len(steps) for _, steps in groups)} 条操作记录",
            ],
        ],
        notes,
        stage=True,
    )
    if len(groups) > 1:
        story.extend([p("1. 已保存的操作与辅助活动", "h1"), p(notes)])
    overview_index = len(story)
    overview_rows = [["片段", "时间范围", "操作记录", "结束状态"]]
    states = {
        "ongoing_at_recording_end": "录像结束，实验待续",
        "observed_complete": "已观察到结束",
        "unresolved": "结束位置待核对",
        "unreviewed": "边界尚未核对",
    }
    sections = []
    from .activity_review import assessment
    for index, (g, steps) in enumerate(groups, 1):
        activity = assessment(g)
        title = g.get("experiment_name") or f"实验操作片段 {index}"
        if title.startswith("待模型命名"):
            title = "操作片段（说明待补全）"
        state = states.get(g.get("completion_status"), "结束位置待核对")
        if activity["is_auxiliary"]:
            title = activity["label"] + "记录"
            state = "辅助活动，不计入实验"
        interval = (
            f"{clock(g.get('global_start_ms'))} - {clock(g.get('global_end_ms'))}"
        )
        overview_rows.append([f"{index}. {title}", interval, len(steps), state])
        if index > 1:
            story.append(PageBreak())
        story.extend(
            [
                p(f"{index:02d}  {title}", "h1"),
                p(f"{interval} · {len(steps)} 条操作记录 · {state}", "small"),
            ]
        )
        visual = next(
            (
                v
                for v in export.get("representative_visuals", [])
                if v["group_id"] == g.get("group_id")
            ),
            None,
        )
        visual_html = ""
        if visual:
            from reportlab.lib.utils import ImageReader

            source = root / visual["path"]
            image_bytes = source.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != visual["sha256"]:
                raise ValueError("Representative frame changed during report rendering")
            width, height = ImageReader(io.BytesIO(image_bytes)).getSize()
            scale = min(170 * mm / width, 78 * mm / height)
            frame = Image(io.BytesIO(image_bytes), width=width * scale, height=height * scale)
            story.extend(
                [
                    frame,
                    p(
                        f"图 {index} · 留存对照画面，动作仍需核对。",
                        "small",
                    ),
                ]
            )
            visual_html = f'<figure><img style="width:100%;height:auto;border-radius:12px" src="../{quote(visual["path"], safe="/")}"><figcaption>留存对照画面 · 动作仍需核对</figcaption></figure>'
        step_html = []
        for n, step in enumerate(steps, 1):
            name = step.get("operation_title") or f"操作记录 {n}"
            when = f"{clock(step.get('start_global_ms'))} - {clock(step.get('end_global_ms'))}"
            story.extend([p(f"{n:02d}  {name}", "h2"), p(when, "small")])
            body = paragraphs(step.get("current_step")) or ["尚无完整操作说明"]
            story.extend(p(x) for x in body)
            if step.get("physical_change"):
                story.extend(
                    [
                        p("前后变化", "h2"),
                        *[p(x) for x in paragraphs(step["physical_change"])],
                    ]
                )
            if step.get("next_step") and step.get("next_step_status") in {
                "observed",
                "inferred",
            }:
                label = (
                    "已观察到的后续"
                    if step["next_step_status"] == "observed"
                    else "预测后续（画面未确认）"
                )
                story.append(p(f"{label}：{step['next_step']}"))
            story.append(p(f"关联素材：{len(step.get('supporting_event_ids') or [])} 项", "small"))
            step_html.append(
                f'<article class="step"><header><b>{n:02d} {html.escape(name)}</b><small>{when}</small></header>'
                + "".join(f"<p>{html.escape(x)}</p>" for x in body)
                + f"<footer>关联素材：{len(step.get('supporting_event_ids') or [])} 项</footer></article>"
            )
        story.append(
            p(
                f"对应视频：可在实验记录中按 {interval} 回看。",
                "small",
            )
        )
        limits = readable_uncertainties((g.get("model_understanding") or {}).get("uncertainties") or [])
        if limits:
            story.append(p("本段需要核对的内容", "h2"))
            story.extend(p(x, "small") for x in limits)
        sections.append(
            f'<section><header><small>{"辅助活动" if activity["is_auxiliary"] else "操作片段"} {index:02d} · {interval}</small><h2>{html.escape(title)}</h2><span class="badge">{state}</span></header>'
            + visual_html
            + "".join(step_html)
            + (
                "<details><summary>本段需要核对的内容</summary>"
                + "".join(f"<p>{html.escape(str(x))}</p>" for x in limits)
                + "</details>"
                if limits
                else ""
            )
            + "</section>"
        )
    if len(groups) > 1:
        story.insert(
            overview_index, table(overview_rows, [65 * mm, 45 * mm, 20 * mm, 40 * mm])
        )
    story.extend([p("查阅对应画面", "h2"), p(
        "可在应用的实验记录中按步骤时间回看视频、查看对应素材。完整来源引用随本次实验档案保存。",
        "small")])

    def footer(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFillColor(brand)
            canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, stroke=0, fill=1)
        canvas.setFont(regular, 8)
        canvas.setFillColor(muted)
        canvas.drawString(
            20 * mm, 11 * mm, "VisionCortex · 阶段成果 · 待核对"
        )
        canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    pdf = folder / "Stage-Evidence-Report.pdf"
    temp = pdf.with_suffix(".partial.pdf")
    SimpleDocTemplate(
        str(temp),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=21 * mm,
        title=f"VisionCortex 阶段证据报告 - {identifier}",
        author="VisionCortex",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    temp.replace(pdf)
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionCortex 实验室阶段日报</title>
<style>body{{margin:0;background:#f3f7f5;color:#29443d;font:16px/1.75 sans-serif}}main{{max-width:1080px;margin:40px auto;padding:0 24px}}.cover,section{{padding:28px;margin-bottom:22px;background:white;border:1px solid #dce7e1;border-radius:16px}}.brand{{display:flex;align-items:center;gap:14px}}.brand img{{width:46px}}h1{{font-size:30px;line-height:1.35}}h2{{font-size:22px}}small,footer{{color:#6a8178;font-size:13px}}.badge{{display:inline-block;color:#795e2d;background:#faf1df;padding:4px 12px;border-radius:20px}}nav{{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}}nav a{{padding:10px 16px;border:1px solid #cadbd3;border-radius:8px;color:#205950;text-decoration:none}}.step{{padding:20px 0;border-top:1px solid #e2eae6}}.step header{{display:flex;justify-content:space-between;gap:16px}}.step p{{margin:10px 0}}details{{margin-top:20px}}summary{{cursor:pointer}}code{{display:block;overflow-wrap:anywhere;font-size:12px}}li{{padding:8px 0}}@media(max-width:600px){{main{{padding:0 12px}}.cover,section{{padding:20px}}.step header{{display:block}}}}</style>
<main><div class="cover"><div class="brand"><img src="{brand_logo_data_url()}" alt="VisionCortex"><b>VisionCortex · 实验室日报</b></div><h1>{html.escape(identifier)}</h1><span class="badge">阶段日报 · 待核对</span><p>{notes}</p><small>{len(groups)} 个片段 · {sum(len(s) for _, s in groups)} 条操作记录</small><nav><a href="Stage-Evidence-Report.pdf">下载专业 PDF（阶段版）</a><a href="Analysis-Result.json">下载可追溯 JSON</a></nav></div>{"".join(sections)}<section><h2>阅读说明</h2><p>本次分析尚有内容需要核对。可在应用的实验记录中按时间回看画面，来源引用随实验档案保存。</p></section></main></html>'''
    document = document.replace(
        "</html>",
        "<script>\nif(['/api/staging-file','/api/archive-file'].includes(location.pathname)) {\n const current = new URL(location.href), file = current.searchParams.get('path');\n if(file) document.querySelectorAll('a[href],img[src]').forEach(a => {\n  const attr = a.tagName === 'IMG' ? 'src' : 'href';\n  const target = new URL(a.getAttribute(attr), 'https://relative.invalid/' + file);\n  if(target.origin !== 'https://relative.invalid') return;\n  const link = new URL(current); link.searchParams.set('path', decodeURIComponent(target.pathname.slice(1)));\n  a.setAttribute(attr, link.href);\n });\n}\n</script></html>",
    )
    daily = folder / "Stage-Lab-Daily-Report.html"
    temporary = daily.with_suffix(".partial.html")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(daily)
    return {
        key: {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in [("pdf", pdf), ("daily_html", daily)]
    }
