"""Optional single-class LCD locator before the existing PP-OCR backend.

The detector supplies source geometry only. No instrument, person, field role,
unit, decimal rule or business confirmation can come from its class number.
"""
import hashlib
import math
from pathlib import Path
import time


def settings(value):
    spec = value.get('panel_locator')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {'checkpoint', 'confidence', 'iou', 'max_panels', 'padding', 'tiled_detection'}:
        raise ValueError('Explicit bounded panel locator settings required')
    if not isinstance(spec.get('checkpoint'), str) or not spec['checkpoint']:
        raise ValueError('Panel locator requires a verified checkpoint')
    result = dict(checkpoint=spec['checkpoint'], confidence=spec.get('confidence', .25),
                  iou=spec.get('iou', .45), max_panels=spec.get('max_panels', 24), padding=spec.get('padding', .08),
                  tiled_detection=spec.get('tiled_detection', False))
    if type(result['tiled_detection']) is not bool:
        raise ValueError('Explicit tiled detection boolean required')
    for name, low, high in [('confidence', .05, .95), ('iou', .1, .9), ('padding', 0, .2)]:
        v = result[name]
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError('Invalid panel locator threshold: ' + name)
    if type(result['max_panels']) is not int or not 1 <= result['max_panels'] <= 24:
        raise ValueError('Panel count exceeds bound')
    return result | {'processing_revision': 'source-lcd-yolo-onnx/1',
                     'backend_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class PanelOCR:
    def __init__(self, backend, model_root, configuration):
        import onnxruntime as ort
        self.backend, self.configuration = backend, settings(configuration)
        if self.configuration is None or configuration.get('input_mode') != 'full_image':
            raise ValueError('Panel localization requires full-image OCR input')
        path = Path(model_root) / 'model.onnx'
        self.weights_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = configuration.get('cpu_threads', 1)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (self.session.get_providers() != ['CPUExecutionProvider'] or len(inputs) != 1 or len(outputs) != 1
                or inputs[0].type != 'tensor(float)' or inputs[0].shape != [1, 3, 640, 640]
                or outputs[0].type != 'tensor(float)' or outputs[0].shape != [1, 5, 8400]):
            raise ValueError('Only the pinned single-class static YOLOv8n LCD graph is supported')

    def _locate_single(self, image):
        import cv2
        import numpy as np
        h, w = image.shape[:2]
        ratio = min(640 / h, 640 / w)
        rh, rw = round(h * ratio), round(w * ratio)
        top, left = round((640 - rh) / 2 - .1), round((640 - rw) / 2 - .1)
        resized = cv2.resize(image, (rw, rh))
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top+rh, left:left+rw] = resized
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        tick = time.monotonic()
        prediction = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0].T
        if not np.isfinite(prediction).all() or (prediction[:, 4] < 0).any() or (prediction[:, 4] > 1).any():
            raise ValueError('Nonfinite LCD detections')
        found = prediction[prediction[:, 4] >= self.configuration['confidence']]
        if len(found):
            boxes = np.concatenate((found[:, :2] - found[:, 2:4] / 2,
                                    found[:, :2] + found[:, 2:4] / 2), axis=1)
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            order = found[:, 4].argsort()[::-1]; keep = []
            while len(order):
                i = order[0]; keep.append(int(i)); rest = order[1:]
                lo = np.maximum(boxes[i, :2], boxes[rest, :2]); hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
                intersection = np.maximum(hi-lo, 0).prod(axis=1)
                area = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(axis=1)
                iou = intersection / np.maximum(area[i]+area[rest]-intersection, 1e-9)
                order = rest[iou <= self.configuration['iou']]
            candidates = [{'box': boxes[i].tolist(), 'score': float(found[i, 4])} for i in keep]
        else:
            candidates = []
        valid = [c for c in candidates if c['box'][2]-c['box'][0] >= 8 and c['box'][3]-c['box'][1] >= 8]
        selected = sorted(valid[:self.configuration['max_panels']], key=lambda c: (c['box'][1], c['box'][0]))
        return selected, dict(status='detected' if selected else 'no_panel_detected',
            source_size=[w, h], settings=self.configuration, weights_sha256=self.weights_sha256,
            seconds=time.monotonic()-tick, candidate_count=len(candidates),
            excluded_count=len(candidates)-len(selected), physical_identity_inferred=False,
            input_tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest())

    def locate(self, image):
        boxes, receipt = self._locate_single(image)
        if not self.configuration['tiled_detection']:
            return boxes, receipt
        h, w = image.shape[:2]; views = [dict(source_box=[0, 0, w, h], execution=receipt)]
        for x, y in [(0, 0), (.35, 0), (0, .35), (.35, .35)]:
            a, b, c, d = round(x*w), round(y*h), round((x+.65)*w), round((y+.65)*h)
            found, execution = self._locate_single(image[b:d, a:c])
            boxes.extend(dict(box=[v['box'][0]+a, v['box'][1]+b, v['box'][2]+a, v['box'][3]+b], score=v['score']) for v in found)
            views.append(dict(source_box=[a, b, c, d], execution=execution))
        found_count=len(boxes); kept=[]
        for candidate in sorted(boxes, key=lambda row: row['score'], reverse=True):
            a=candidate['box']; duplicate=False
            for selected in kept:
                b=selected['box']; intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
                if intersection/max(union,1e-9)>self.configuration['iou']:duplicate=True;break
            if not duplicate:kept.append(candidate)
        selected=sorted(kept[:self.configuration['max_panels']],key=lambda row:(row['box'][1],row['box'][0]))
        receipt=dict(receipt,status='detected' if selected else 'no_panel_detected',views=views,
            candidate_count=sum(v['execution']['candidate_count'] for v in views),
            excluded_count=sum(v['execution']['excluded_count'] for v in views)+found_count-len(selected),
            seconds=sum(v['execution']['seconds'] for v in views))
        return selected, receipt

    def predict(self, image):
        boxes, receipt = self.locate(image)
        h, w = image.shape[:2]; pages, executions = [], []
        # A specialized locator is not a coverage gate. Keep whole-image OCR
        # even when some displays were found: another display may be missed.
        try:
            whole = self.backend.predict(image)
            pages.extend(whole['results'])
            executions.append(dict(panel_id=None, source_box=[0, 0, w, h], ocr_execution=whole.get('onnx_execution')))
        except ValueError as exc:
            executions.append(dict(panel_id=None, status='needs_attention', error=str(exc)[:250]))
        for index, detection in enumerate(boxes):
            x1, y1, x2, y2 = detection['box']; padding = self.configuration['padding']
            dx, dy = (x2-x1)*padding, (y2-y1)*padding
            a, b, c, d = max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)), min(w, math.ceil(x2+dx)), min(h, math.ceil(y2+dy))
            crop = image[b:d, a:c]
            try:
                output = self.backend.predict(crop)
            except ValueError as exc:
                executions.append(dict(panel_id='panel-' + str(index+1), source_box=[a, b, c, d],
                                       status='needs_attention', error=str(exc)[:250]))
                continue
            for page in output['results']:
                page = dict(page, panel_id='panel-' + str(index+1), panel_box=[a, b, c, d],
                            detected_panel_box=detection['box'], panel_score=detection['score'],
                            geometry_basis='model_detected_lcd_screen_not_instrument_identity')
                page['rec_polys'] = [[[x+a, y+b] for x, y in poly] for poly in page.get('rec_polys', [])]
                page['rec_boxes'] = [[x1+a, y1+b, x2+a, y2+b] for x1, y1, x2, y2 in page.get('rec_boxes', [])]
                pages.append(page)
            executions.append(dict(panel_id='panel-' + str(index+1), source_box=[a, b, c, d],
                                   crop_pixels_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                                   ocr_execution=output.get('onnx_execution')))
        if not pages:
            raise ValueError('No OCR page completed in whole-image or panel paths')
        return dict(results=pages, panel_localization=receipt | {'fallback': 'whole_image_raw_ocr_always_retained'},
                    panel_executions=executions)
