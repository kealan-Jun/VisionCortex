"""Render labels on the exact inferred frame; keep only a JPEG in memory."""

from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=4)
def font(size):
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    return (
        ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)
    )


def render(frame, panels):
    height, width = frame.shape[:2]
    scale = min(1, 960 / width)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)))
    draw = ImageDraw.Draw(image)
    text_font = font(16)
    labels = []
    for panel in panels:
        x1, y1, x2, y2 = [round(v * scale) for v in panel["bbox"]]
        texts = []
        for line in panel["local_ocr"].get("lines", []):
            if line.get("quality_issue"):
                texts.append("字段冲突，需校正")
            elif line.get("normalized_value"):
                value = line["normalized_value"]
                texts.append(
                    str(value["value"]) + (" " + value["unit"] if value["unit"] else "")
                    if value["value"] is not None
                    else "无有效读数"
                )
            elif line.get("text"):
                texts.append(line["text"])
        if panel.get("display_state"):
            texts = [panel["display_state"]["text"]]
        color = "#2fd5a5" if texts else "#ffce66"
        name = panel.get("instrument_name") or ("设备 A" if panel["class_id"] == 0 else "设备 B")
        label = f"{name} · {panel.get('measurement_name') or '面板'} · {' / '.join(texts) or '无数字读数'}"
        label = label[:90]
        box = draw.textbbox((0, 0), label, font=text_font)
        label_width = min(image.width, box[2] + 12)
        left = max(0, min(x1, image.width - label_width))
        # Adjacent temperature/speed windows need separate readable labels.
        tops = [max(0, y1 - 29) - 31 * i for i in range(7)] + [y2 + 4 + 31 * i for i in range(7)]
        top = next(
            (
                t
                for t in tops
                if 0 <= t <= image.height - 27
                and not any(
                    left < r[2] and left + label_width > r[0] and t < r[3] and t + 27 > r[1]
                    for r in labels
                )
            ),
            max(0, y1 - 29),
        )
        labels.append((left, top, left + label_width, top + 27))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.rectangle((left, top, left + label_width, top + 27), fill="#12343d")
        draw.text((left + 5, top + 2), label, font=text_font, fill=color)
    if not panels:
        draw.rectangle((12, 12, 320, 42), fill="#12343d")
        draw.text((18, 14), "当前画面未检测到已绑定面板", font=text_font, fill="white")
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def multipart(jpeg):
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(jpeg)).encode()
        + b"\r\n\r\n"
        + jpeg
        + b"\r\n"
    )
