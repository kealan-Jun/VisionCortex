from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .report_brand import BRAND_COLORS, brand_logo_data_url, brand_logo_png


def _duration(seconds: float | int | None) -> str:
    value = max(0.0, float(seconds or 0))
    hours, remainder = divmod(value, 3600)
    minutes, remaining = divmod(remainder, 60)
    if hours:
        return f"{int(hours)} 小时 {int(minutes)} 分 {remaining:.1f} 秒"
    if minutes:
        return f"{int(minutes)} 分 {remaining:.1f} 秒"
    return f"{remaining:.1f} 秒"


def _headline_status(report: dict[str, Any]) -> tuple[str, str]:
    overview = report["overview"]
    if not overview.get("evidence_package_eval_passed"):
        return "结构检查未通过", "danger"
    if (report.get("human_review") or {}).get("status") == "rejected":
        return "人工复核未通过", "danger"
    quality = report.get("quality_acceptance") or {}
    if quality.get("passed") is not True:
        if quality.get("status") in {"failed", "rejected"}:
            return "质量检查未通过", "danger"
        return "待质量复核", "warning"
    if (report.get("human_review") or {}).get("status") != "approved":
        return "自动检查通过，待人工复核", "warning"
    if report.get("contradictions") or report.get("uncertainties"):
        return "通过，含需关注项", "warning"
    return "证据验收通过", "success"


def render_daily_markdown(report: dict[str, Any]) -> str:
    overview = report["overview"]
    performance = report["performance"]
    status, _ = _headline_status(report)
    lines = [
        f"# VisionCortex 实验室日报 · {report['report_date']}",
        "",
        f"**档案：** `{report['experiment_id']}`  ",
        f"**结果：** {status}",
        "",
        "## 今日概览",
        "",
        f"- {overview['experiment_group_count']} 个有界实验，{overview['input_view_count']} 路输入视角",
        f"- {overview['key_event_count']} 个关键事件，{overview['physical_change_count']} 项物理状态变化",
        f"- 流水线耗时 {_duration(performance.get('total_duration_seconds'))}，总 Token {performance.get('total_tokens') or 0:,}",
        "- 日报复用已有理解，不新增模型调用或 Token",
        "",
        "> “当前步骤/下一步骤”属于双视角证据支持的模型理解；直接观察事实和不确定项在 JSON 中分别保留。",
        "",
        "## 实验简报",
        "",
    ]
    for index, group in enumerate(report["experiment_timeline"], 1):
        continuity = "连续实验" if group["continuity_type"] == "continuous" else "独立实验"
        lines.extend(
            [
                f"### {index}. {group['experiment_name']}",
                "",
                f"- 时间：`{group['start_timecode']}`—`{group['end_timecode']}`（{continuity}，{group['duration_seconds']:.1f} 秒）",
                f"- 结果摘要：{group.get('overall_summary') or '暂无摘要'}",
            ]
        )
        visual = group.get("representative_visual")
        if visual:
            lines.extend(
                [
                    f"- 代表证据：`{visual['event_id']}` · {visual['action_label']} · `{visual['peak_timecode']}`",
                    f"![{visual['event_id']} 双视角证据](../../{visual['image_path']})",
                ]
            )
        else:
            lines.append("- 代表证据：未找到满足双视角门的已归档关键帧")
        if group.get("steps"):
            first = group["steps"][0]
            last = group["steps"][-1]
            next_status = str(last.get("next_step_status") or "unknown")
            next_label = (
                "已观察后续"
                if next_status == "observed"
                else "预测后续"
                if next_status == "inferred"
                else "后续未知"
            )
            lines.extend(
                [
                    f"- 开始阶段：{first.get('current_step') or '未说明'}",
                    f"- {next_label}：{last.get('next_step') or '证据不足'}",
                ]
            )
        action_text = "；".join(
            f"{item['action_label']} {item['event_count']}"
            for item in group.get("key_action_summary") or []
            if item["event_count"]
        )
        lines.extend([f"- 动作类别：{action_text or '无保留动作'}", ""])
    lines.extend(
        [
            "## 关注事项与交接",
            "",
            f"- 质量状态：{status}",
            f"- 不确定性：{len(report['uncertainties'])} 组；跨视角矛盾：{len(report['contradictions'])} 项",
            f"- 对齐状态：{report['alignment_summary']['aligned']}/{report['alignment_summary']['view_count']} 路完成",
            "- 完整步骤、关键事件、素材路径与来源信息请在同档案 Web 页面或 JSON 索引中查看。",
            "",
            "## 自动验收与备注",
            "",
            "- 状态：仅以自动证据与质量门禁结果为准",
            "- 不确定项：证据不足时自动隔离，不进入正式事实",
            "",
        ]
    )
    return "\n".join(lines)


def render_daily_html(report: dict[str, Any]) -> str:
    def e(value: Any) -> str:
        return html.escape(str(value if value not in (None, "") else "-"))

    def archive_url(value: str) -> str:
        return "../../" + quote(value.replace("\\", "/"), safe="/+._-")

    status, status_class = _headline_status(report)
    overview = report["overview"]
    performance = report["performance"]
    experiment_cards: list[str] = []
    for index, group in enumerate(report["experiment_timeline"], 1):
        visual = group.get("representative_visual")
        if visual:
            visual_html = (
                f"<a class='visual' href='{e(archive_url(visual['clip_path']))}'>"
                f"<img loading='lazy' src='{e(archive_url(visual['image_path']))}' "
                f"alt='{e(group['experiment_name'])} {e(visual['event_id'])} 双视角关键帧'>"
                f"<span>{e(visual['event_id'])} · {e(visual['action_label'])} · {e(visual['peak_timecode'])}</span></a>"
            )
        else:
            visual_html = "<div class='visual missing'>本实验暂无满足双视角门的代表图片</div>"
        actions = "".join(
            f"<span class='pill'>{e(item['action_label'])} <b>{item['event_count']}</b></span>"
            for item in group.get("key_action_summary") or []
            if item["event_count"]
        )
        empty_actions = '<span class="pill">无保留动作</span>'
        first_step = group["steps"][0] if group.get("steps") else {}
        last_step = group["steps"][-1] if group.get("steps") else {}
        next_status = str(last_step.get("next_step_status") or "unknown")
        next_label = (
            "已观察后续"
            if next_status == "observed"
            else "预测后续"
            if next_status == "inferred"
            else "后续未知"
        )
        continuity = "连续实验" if group["continuity_type"] == "continuous" else "独立实验"
        experiment_cards.append(
            f"<section class='experiment'><div class='section-kicker'>实验 {index:02d}</div>"
            f"<h2>{e(group['experiment_name'])}</h2>"
            f"<p class='meta'>{e(group['start_timecode'])}—{e(group['end_timecode'])} · {continuity} · {group['duration_seconds']:.1f} 秒</p>"
            f"<div class='experiment-grid'>{visual_html}<div class='brief'>"
            f"<h3>结果摘要</h3><p>{e(group.get('overall_summary') or '暂无摘要')}</p>"
            f"<dl><dt>开始阶段</dt><dd>{e(first_step.get('current_step') or '未说明')}</dd>"
            f"<dt>{next_label}</dt><dd>{e(last_step.get('next_step') or '证据不足')}</dd></dl>"
            f"<div class='pills'>{actions or empty_actions}</div>"
            f"</div></div></section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisionCortex 实验室日报 {e(report['report_date'])}</title>
<style>
:root{{--brand:{BRAND_COLORS['brand']};--brand2:{BRAND_COLORS['brand_secondary']};--soft:{BRAND_COLORS['brand_soft']};--accent:{BRAND_COLORS['accent']};--bg:{BRAND_COLORS['background']};--surface:{BRAND_COLORS['surface']};--text:{BRAND_COLORS['text']};--muted:{BRAND_COLORS['text_muted']};--border:{BRAND_COLORS['border']};--success:{BRAND_COLORS['success']};--warning:{BRAND_COLORS['warning']};--danger:{BRAND_COLORS['danger']}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.68 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}main{{max-width:1120px;margin:auto;padding:28px 22px 48px}}.hero,.experiment,.panel{{background:var(--surface);border:1px solid var(--border);border-radius:18px;box-shadow:0 8px 24px rgba(30,74,82,.07)}}.hero{{overflow:hidden;margin-bottom:18px}}.brandbar{{height:9px;background:linear-gradient(90deg,var(--brand),var(--brand2),var(--accent))}}.hero-body{{padding:26px 28px}}.brand{{display:flex;align-items:center;gap:14px}}.brand img{{width:58px;height:58px}}.brand-name{{font-size:14px;letter-spacing:.12em;color:var(--brand);font-weight:700}}h1{{font-size:30px;line-height:1.25;margin:3px 0 8px}}h2{{font-size:22px;margin:0 0 5px}}h3{{font-size:15px;color:var(--brand);margin:0 0 7px}}.meta,.note{{color:var(--muted)}}.status{{display:inline-block;padding:5px 11px;border-radius:999px;color:white;font-weight:700;background:var(--{status_class})}}.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:22px}}.stat{{padding:14px;background:#f2f7f7;border-radius:12px}}.stat b{{display:block;font-size:23px;color:var(--brand)}}.stat span{{color:var(--muted);font-size:13px}}.section-title{{margin:27px 2px 12px;font-size:20px}}.experiment{{padding:22px;margin-bottom:16px}}.section-kicker{{font-size:12px;font-weight:700;letter-spacing:.12em;color:var(--brand2)}}.experiment-grid{{display:grid;grid-template-columns:minmax(330px,1.04fr) minmax(320px,.96fr);gap:22px;margin-top:14px}}.visual{{display:block;color:var(--muted);text-decoration:none;border:1px solid var(--border);border-radius:13px;overflow:hidden;background:#edf3f3}}.visual img{{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}}.visual span{{display:block;padding:8px 11px;font-size:12px}}.visual.missing{{min-height:220px;display:grid;place-items:center;padding:20px}}.brief p{{margin-top:0}}dl{{display:grid;grid-template-columns:76px 1fr;gap:7px 12px;margin:14px 0}}dt{{font-weight:700;color:var(--brand)}}dd{{margin:0}}.pills{{display:flex;flex-wrap:wrap;gap:7px}}.pill{{background:var(--soft);color:var(--brand);padding:5px 9px;border-radius:999px;font-size:12px}}.panel{{padding:22px;margin-top:16px}}.attention{{border-left:5px solid var(--accent)}}.footer-note{{font-size:12px;color:var(--muted);margin-top:18px}}@media(max-width:850px){{.stats{{grid-template-columns:1fr 1fr}}.experiment-grid{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}.hero,.experiment,.panel{{box-shadow:none;break-inside:avoid}}}}
</style></head><body><main>
<header class="hero"><div class="brandbar"></div><div class="hero-body"><div class="brand"><img src="{brand_logo_data_url()}" alt="VisionCortex Logo"><div><div class="brand-name">VISIONCORTEX</div><h1>实验室日报 · {e(report['report_date'])}</h1></div></div>
<p class="meta">档案：{e(report['experiment_id'])}</p><span class="status">{e(status)}</span>
<div class="stats"><div class="stat"><b>{overview['experiment_group_count']}</b><span>有界实验</span></div><div class="stat"><b>{overview['input_view_count']}</b><span>输入视角</span></div><div class="stat"><b>{overview['key_event_count']}</b><span>关键事件</span></div><div class="stat"><b>{overview['physical_change_count']}</b><span>状态变化</span></div><div class="stat"><b>{performance.get('total_tokens') or 0:,}</b><span>总 Token</span></div></div>
<p class="note">本日报面向日常查看与交接，呈现已归档分析和代表性双视角证据，质量状态以复核记录为准。完整技术账本保留在同一档案中。</p></div></header>
<h2 class="section-title">实验简报</h2>{''.join(experiment_cards)}
<section class="panel attention"><h2>关注事项与交接</h2><p>质量状态：<b>{e(status)}</b>。时间对齐完成 {report['alignment_summary']['aligned']}/{report['alignment_summary']['view_count']} 路；记录不确定性 {len(report['uncertainties'])} 组、跨视角矛盾 {len(report['contradictions'])} 项。</p><p>流水线总耗时 {_duration(performance.get('total_duration_seconds'))}；预处理 {_duration((performance.get('preprocessing_sla') or {}).get('actual_seconds'))}；模型用量 {performance.get('total_input_tokens') or 0:,} 输入 + {performance.get('total_output_tokens') or 0:,} 输出 = {performance.get('total_tokens') or 0:,} Token。</p></section>
  <section class="panel"><h2>自动验收与不确定项</h2><p>状态：仅以自动证据与质量门禁结果为准。</p><p>证据不足的事件自动隔离，不进入正式事实。</p></section>
<p class="footer-note">当前步骤和下一步骤属于已归档证据支持的模型理解；图片可点击打开对应关键片段。日报生成不新增模型调用或 Token。</p>
</main></body></html>"""


def _register_fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc")),
        (Path("C:/Windows/Fonts/simsun.ttc"), Path("C:/Windows/Fonts/simhei.ttf")),
        (
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
            Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            try:
                pdfmetrics.getFont("VCReport")
            except KeyError:
                try:
                    pdfmetrics.registerFont(TTFont("VCReport", str(regular)))
                    pdfmetrics.registerFont(TTFont("VCReportBold", str(bold)))
                except Exception:
                    # ReportLab cannot load every TTC/OTF outline flavor. Try
                    # the next installed candidate before falling back to the
                    # built-in fonts.
                    continue
            return "VCReport", "VCReportBold"
    return "Helvetica", "Helvetica-Bold"


def render_professional_pdf(path: Path, report: dict[str, Any], archive_root: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        CondPageBreak,
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.utils import ImageReader

    regular, bold = _register_fonts()
    palette = {key: colors.HexColor(value) for key, value in BRAND_COLORS.items()}
    styles = getSampleStyleSheet()
    cover_title = ParagraphStyle("CoverTitle", parent=styles["Title"], fontName=bold, fontSize=24, leading=32, alignment=TA_CENTER, textColor=palette["brand"])
    cover_subtitle = ParagraphStyle("CoverSub", parent=styles["BodyText"], fontName=regular, fontSize=11, leading=17, alignment=TA_CENTER, textColor=palette["text_muted"])
    h1 = ParagraphStyle("ReportH1", parent=styles["Heading1"], fontName=bold, fontSize=16, leading=22, spaceBefore=9, spaceAfter=7, textColor=palette["brand"])
    h2 = ParagraphStyle("ReportH2", parent=h1, fontSize=12, leading=17, spaceBefore=7, spaceAfter=5, textColor=palette["brand_secondary"])
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontName=regular, fontSize=9, leading=14, textColor=palette["text"])
    small = ParagraphStyle("ReportSmall", parent=body, fontSize=7.2, leading=10.5, textColor=palette["text_muted"])
    caption = ParagraphStyle("ReportCaption", parent=small, fontSize=6.8, leading=9, spaceBefore=3, alignment=TA_CENTER)

    def p(value: Any, style=body) -> Paragraph:
        return Paragraph(html.escape(str(value if value not in (None, "") else "-")), style)

    def styled_table(
        rows: list[list[Any]], widths: list[float], *, header: str = "brand_soft"
    ) -> Table:
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), bold),
                    ("FONTNAME", (0, 1), (-1, -1), regular),
                    ("BACKGROUND", (0, 0), (-1, 0), palette[header]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), palette["brand"]),
                    ("GRID", (0, 0), (-1, -1), 0.35, palette["border"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def image_flowable(reference: str, max_width: float, max_height: float):
        source = archive_root / Path(reference)
        if not source.is_file():
            placeholder = Table([[p(f"图片不可用：{reference}", small)]], colWidths=[max_width], rowHeights=[min(max_height, 32 * mm)])
            placeholder.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, palette["warning"]), ("BACKGROUND", (0, 0), (-1, -1), palette["surface"]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            return placeholder
        reader = ImageReader(str(source))
        width, height = reader.getSize()
        scale = min(max_width / width, max_height / height)
        return Image(str(source), width=width * scale, height=height * scale)

    def footer(canvas, document):
        canvas.saveState()
        if document.page > 1:
            canvas.setFillColor(palette["brand"])
            canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, stroke=0, fill=1)
            canvas.setFillColor(palette["text_muted"])
            canvas.setFont(regular, 7)
            canvas.drawString(18 * mm, 10 * mm, f"VisionCortex · {report['experiment_id']}")
            canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {document.page} 页")
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=17 * mm,
        title=f"VisionCortex 多视角实验分析专业证据报告 {report['report_date']}",
        author="VisionCortex",
    )
    status, _ = _headline_status(report)
    overview = report["overview"]
    logo_buffer = io.BytesIO(brand_logo_png())
    logo = Image(logo_buffer, width=34 * mm, height=34 * mm)
    logo.hAlign = "CENTER"
    story: list[Any] = [
        Spacer(1, 25 * mm), logo, Spacer(1, 9 * mm),
        Paragraph("多视角实验分析<br/>专业证据报告", cover_title),
        Spacer(1, 5 * mm),
        Paragraph("Professional Multi-View Laboratory Evidence Report", cover_subtitle),
        Spacer(1, 18 * mm),
        styled_table(
            [
                ["报告日期", report["report_date"]],
                ["实验档案", p(report["experiment_id"], small)],
                ["质量状态", status],
                ["证据范围", f"{overview['experiment_group_count']} 个有界实验 / {overview['key_event_count']} 个关键事件 / {overview['input_view_count']} 路视角"],
                ["报告模板", report["presentation_contract"]["professional_template_id"]],
            ],
            [37 * mm, 118 * mm],
        ),
        Spacer(1, 20 * mm),
        Paragraph("本报告由归档的实验片段、双视角素材、步骤理解和运行记录自动生成。质量状态见上表；完整证据与媒体文件保留在同一实验档案中。", cover_subtitle),
        PageBreak(),
        Paragraph("1. 结论摘要", h1),
        Paragraph(
            f"本次分析识别并归档 {overview['experiment_group_count']} 个有界实验，记录 {overview['key_event_count']} 个关键事件和 {overview['physical_change_count']} 项物理状态变化。证据包状态为“{status}”。正文按实验展示代表性第一/第三人称对齐图片、过程摘要和细粒度步骤；完整事件明细保留在可索引 JSON 与 SQLite 中。",
            body,
        ),
    ]
    matrix = [["实验", "时间范围", "类型", "步骤", "关键事件", "代表图片"]]
    for group in report["experiment_timeline"]:
        matrix.append(
            [
                p(group["experiment_name"], small),
                p(f"{group['start_timecode']}—{group['end_timecode']}", small),
                "连续" if group["continuity_type"] == "continuous" else "独立",
                len(group["steps"]),
                len(group["key_events"]),
                "有" if group.get("representative_visual") else "缺失",
            ]
        )
    story.extend(
        [
            Spacer(1, 4 * mm),
            styled_table(matrix, [53 * mm, 39 * mm, 17 * mm, 16 * mm, 20 * mm, 20 * mm]),
            Paragraph("2. 报告范围与证据口径", h1),
            Paragraph("本报告使用已归档的分析证据，结构检查不能替代内容质量复核。图片必须来自第一人称与第三人称对齐产物；模型生成的“当前步骤、下一步骤和实验摘要”作为证据支持的解释呈现，不替代直接观察事实。任何不确定或矛盾内容都保留为显式记录。", body),
            Paragraph("阅读方式", h2),
            Paragraph("先阅读本节结论，再按实验查看过程与图片。质量负责人可在最后的附录中核对时间对齐、耗时、Token 和来源路径；无需理解检测模型、解码队列等工程细节即可使用正文。", body),
        ]
    )
    for index, group in enumerate(report["experiment_timeline"], 1):
        story.append(PageBreak())
        story.append(Paragraph(f"3.{index} {html.escape(group['experiment_name'])}", h1))
        continuity = "连续实验" if group["continuity_type"] == "continuous" else "独立实验"
        facts = [
            ["时间范围", "实验类型", "持续时间", "步骤", "关键事件", "物理变化"],
            [
                Paragraph(
                    f"{html.escape(group['start_timecode'])}<br/>— {html.escape(group['end_timecode'])}",
                    small,
                ),
                continuity,
                f"{group['duration_seconds']:.1f} 秒",
                len(group["steps"]),
                len(group["key_events"]),
                group["physical_change_count"],
            ],
        ]
        story.extend([styled_table(facts, [44*mm,24*mm,24*mm,18*mm,22*mm,33*mm]), Spacer(1, 4*mm)])
        visual = group.get("representative_visual")
        if visual:
            image = image_flowable(visual["image_path"], 165 * mm, 86 * mm)
            image.hAlign = "CENTER"
            support_note = (
                "双视角直接支持"
                if visual.get("support_scope") == "dual_role_direct"
                else "对齐双视角；仅单视角直接证明动作，对侧为同步上下文"
            )
            story.extend([image, Paragraph(f"图 {index}-1　{html.escape(visual['action_label'])} · {visual['event_id']} · {visual['peak_timecode']} · 对象：{html.escape('、'.join(visual['objects']) or '未明确')} · {support_note}", caption)])
        else:
            story.append(image_flowable("", 165 * mm, 30 * mm))
        story.extend([Paragraph("实验结果摘要", h2), Paragraph(html.escape(group.get("overall_summary") or "暂无摘要"), body)])
        action_text = "；".join(f"{item['action_label']} {item['event_count']}" for item in group.get("key_action_summary") or [] if item["event_count"]) or "无保留动作"
        story.append(Paragraph(f"动作类别分布：{html.escape(action_text)}。", body))
        step_rows = [["步骤", "时间", "当前在做什么", "后续状态"]]
        for step in group["steps"]:
            next_status = str(step.get("next_step_status") or "unknown")
            next_prefix = (
                "已观察："
                if next_status == "observed"
                else "预测："
                if next_status == "inferred"
                else "未知："
            )
            step_rows.append(
                [
                    str(step.get("step_index") or "-"),
                    Paragraph(
                        f"{html.escape(step['start_timecode'])}<br/>— {html.escape(step['end_timecode'])}",
                        small,
                    ),
                    p(step.get("current_step") or "未说明", small),
                    p(next_prefix + str(step.get("next_step") or "证据不足"), small),
                ]
            )
        story.extend([Paragraph("步骤级理解", h2), styled_table(step_rows, [12*mm,40*mm,56.5*mm,56.5*mm])])
        gallery = group.get("evidence_gallery") or []
        if gallery:
            story.append(Paragraph("动作类别代表证据", h2))
            cells: list[Any] = []
            for item in gallery:
                cells.append(
                    [
                        image_flowable(item["image_path"], 79 * mm, 38 * mm),
                        Paragraph(
                            f"{html.escape(item['action_label'])} · {item['event_id']} · {item['peak_timecode']}<br/>"
                            f"对象：{html.escape('、'.join(item['objects']) or '未明确')}",
                            caption,
                        ),
                    ]
                )
            if len(cells) % 2:
                cells.append("")
            gallery_table = Table([cells[i:i+2] for i in range(0, len(cells), 2)], colWidths=[82.5*mm,82.5*mm], hAlign="LEFT")
            gallery_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 2), ("RIGHTPADDING", (0,0), (-1,-1), 2), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
            story.append(gallery_table)
    story.extend([CondPageBreak(78 * mm), Paragraph("4. 质量、局限与可追溯性", h1)])
    alignment_rows = [["视角", "角色", "状态", "置信度", "CSV RMSE(ms)"]]
    for item in report["alignment_summary"]["views"]:
        alignment_rows.append([p(item.get("view_id"), small), item.get("role"), item.get("state"), f"{float(item.get('confidence') or 0):.4f}", "-" if item.get("csv_rmse_ms") is None else f"{float(item['csv_rmse_ms']):.3f}"])
    story.extend([
        Paragraph(f"对齐完成 {report['alignment_summary']['aligned']}/{report['alignment_summary']['view_count']} 路，平均置信度 {report['alignment_summary']['mean_confidence']}。记录不确定性 {len(report['uncertainties'])} 组、跨视角矛盾 {len(report['contradictions'])} 项。", body),
        styled_table(alignment_rows, [62*mm,30*mm,25*mm,24*mm,25*mm]),
        Paragraph("5. 耗时与模型用量", h1),
    ])
    performance = report["performance"]
    token_by_stage = {"experiment_understanding": performance.get("tokens", {}).get("experiment_groups", {}), "mllm": performance.get("tokens", {}).get("key_materials", {}), "daily_report": performance.get("tokens", {}).get("daily_report", {})}
    stage_rows = [["阶段", "耗时(s)", "输入 Token", "输出 Token", "总 Token", "调用"]]
    for stage in performance.get("stage_durations") or []:
        usage = token_by_stage.get(stage.get("stage"), {})
        stage_rows.append([p(stage.get("stage"), small), f"{float(stage.get('duration_seconds') or 0):.2f}", usage.get("input_tokens") or 0, usage.get("output_tokens") or 0, usage.get("total_tokens") or 0, usage.get("call_count") or 0])
    stage_rows.append(["TOTAL", f"{float(performance.get('total_duration_seconds') or 0):.2f}", performance.get("total_input_tokens") or 0, performance.get("total_output_tokens") or 0, performance.get("total_tokens") or 0, sum(int(item.get("call_count") or 0) for item in token_by_stage.values())])
    story.append(styled_table(stage_rows, [54*mm,25*mm,28*mm,28*mm,27*mm,17*mm]))
    story.extend([
        CondPageBreak(112 * mm),
        Paragraph("6. 运行审计与归档索引", h1),
        Paragraph("完整证据不重复塞入 PDF。本页摘录零拷贝、TensorRT、火山引擎、GPU/内存与质量真值账本；机器可读原始记录保留在同一实验档案。", body),
    ])
    runtime_audit = performance.get("runtime_audit") or {}
    source_audit = runtime_audit.get("source") or {}
    trt_audit = runtime_audit.get("tensorrt") or {}
    encoder_audit = runtime_audit.get("video_encoder") or {}
    mllm_audit = runtime_audit.get("mllm") or {}
    telemetry_audit = runtime_audit.get("telemetry") or {}
    peaks = telemetry_audit.get("peaks") or {}
    quality_audit = runtime_audit.get("quality") or {}

    def peak_text(name: str, suffix: str) -> str:
        item = peaks.get(name) or {}
        if item.get("value") is None:
            return "-"
        return f"{item['value']}{suffix} @ {item.get('stage') or '-'}"

    def metric_text(value: Any) -> str:
        if value is None:
            return "-"
        return f"{float(value):.3f}"

    audit_rows = [
        ["审计项", "结果"],
        ["输入/零拷贝", p(f"{source_audit.get('input_mode') or '-'}；source_copy_bytes={source_audit.get('source_copy_bytes') if source_audit.get('source_copy_bytes') is not None else '-'}；新建连续源副本={source_audit.get('continuous_source_copies_created') if source_audit.get('continuous_source_copies_created') is not None else '-'}", small)],
        ["源文件验证", p(f"已验证 {source_audit.get('verified_file_count') or 0}；冷 stat {source_audit.get('fresh_stat_count') or 0}；命中验证缓存 {source_audit.get('cache_hit_count') or 0}", small)],
        ["TensorRT", p(f"v{trt_audit.get('version') or '-'}；{trt_audit.get('role_count') or 0} 个角色引擎；全部反序列化={trt_audit.get('all_deserialized')}；batch={','.join(str(item) for item in trt_audit.get('build_batches') or []) or '-'}", small)],
        ["NVENC", p(f"{encoder_audit.get('selected') or '-'}；可用={encoder_audit.get('usable')}；软件回退={encoder_audit.get('software_fallback_active')}", small)],
        ["火山引擎/Ark", p(f"{mllm_audit.get('completed_count') or 0}/{mllm_audit.get('call_count') or 0} completed；failed={mllm_audit.get('failed_count') or 0}；reused={mllm_audit.get('cache_reused_call_count') or 0}；model={','.join(mllm_audit.get('models') or []) or '-'}", small)],
        ["GPU / VRAM / RAM 峰值", p(f"{peak_text('gpu_compute_percent', '%')}；{peak_text('gpu_memory_used_mib', ' MiB')}；{peak_text('host_memory_percent', '%')}", small)],
        ["NVDEC / NVENC 峰值", p(f"{peak_text('nvdec_percent', '%')}；{peak_text('nvenc_percent', '%')}", small)],
        ["功耗 / 温度峰值", p(f"{peak_text('gpu_power_w', ' W')}；{peak_text('gpu_temperature_c', ' °C')}；监控样本={telemetry_audit.get('sample_count') or 0}；错误={telemetry_audit.get('sampling_error_count') or 0}", small)],
        ["质量真值", p(f"边界 P/R={metric_text(quality_audit.get('boundary_precision'))}/{metric_text(quality_audit.get('boundary_recall'))}；关键素材 @IoU0.5 P/R/F1={metric_text(quality_audit.get('key_material_precision_at_iou_0_5'))}/{metric_text(quality_audit.get('key_material_recall_at_iou_0_5'))}/{metric_text(quality_audit.get('key_material_f1_at_iou_0_5'))}；状态={quality_audit.get('status') or '-'}", small)],
    ]
    story.extend([Spacer(1, 3*mm), styled_table(audit_rows, [43*mm,122*mm])])
    provenance_rows = [["用途", "归档相对路径"]] + [[p(key.replace("_", " "), small), p(value, small)] for key, value in report.get("provenance", {}).items()]
    provenance_rows.extend([["可检索数据库", p("JSON-Config-Files/evidence_index.sqlite", small)], ["关键素材", p("Key-Materials/", small)], ["实验片段", p("Experiment-Clips/", small)]])
    story.append(
        KeepTogether(
            [
                Spacer(1, 3 * mm),
                Paragraph("归档路径索引", h2),
                styled_table(provenance_rows, [46 * mm, 119 * mm]),
            ]
        )
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
