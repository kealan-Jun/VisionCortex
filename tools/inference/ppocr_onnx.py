"""Local pinned PP-OCR ONNX inference, with explicit field/image semantics.

This implements recorded preprocessing and CTC/DB postprocessing, not the whole
PaddleOCR SDK. A field ROI is supplied geometry, never a detector prediction.
"""
import hashlib
import math
from pathlib import Path
import re
import time


def settings(value):
    mode = value.get('input_mode')
    threads = value.get('cpu_threads', 1)
    maximum = value.get('max_recognitions', 32)
    if (mode not in {'field_roi', 'full_image'} or value.get('device') != 'cpu'
            or type(threads) is not int or not 1 <= threads <= 8
            or type(maximum) is not int or not 1 <= maximum <= 64):
        raise ValueError('ONNX OCR requires explicit field_roi/full_image mode, CPU and bounded threads')
    for role in ('detection', 'recognition'):
        suffix = 'det' if role == 'detection' else 'rec'
        model_name = value.get(role + '_model_name')
        if type(model_name) is not str or not re.fullmatch('PP-OCRv6_(medium|small|tiny)_' + suffix, model_name):
            raise ValueError('Explicit supported PP-OCRv6 detection/recognition model names required')
    return dict(engine='ppocr-onnx', input_mode=mode, device='cpu', cpu_threads=threads,
        max_recognitions=maximum,
        detection_limit_side_len=736, detection_limit_type='min', detection_max_side_limit=4000,
        recognition_height=48, recognition_min_width=320, recognition_max_width=3200,
        processing_revision='recorded-paddlex-processor-compatible-ctc-db/1',
        backend_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def validate_model_configuration(config, kind, declared):
    if config.get('Global', {}).get('model_name') != declared:
        raise ValueError('Declared OCR model differs from inference.yml model identity')
    ops = {key: value for item in config.get('PreProcess', {}).get('transform_ops', []) for key, value in item.items()}
    if ops.get('DecodeImage') != dict(channel_first=False, img_mode='BGR'):
        raise ValueError('Unsupported OCR channel processor configuration')
    post = config.get('PostProcess', {})
    if kind == 'rec':
        chars = post.get('character_dict')
        if (set(ops) != {'DecodeImage', 'MultiLabelEncode', 'RecResizeImg', 'KeepKeys'}
                or ops.get('RecResizeImg') != dict(image_shape=[3, 48, 320]) or post.get('name') != 'CTCLabelDecode'
                or not isinstance(chars, list) or not 1 <= len(chars) <= 100000
                or any(type(c) is not str or not c for c in chars)):
            raise ValueError('Unsupported recognition processor or CTC dictionary')
    else:
        expected = dict(mean=[.485, .456, .406], std=[.229, .224, .225], scale='1./255.', order='hwc')
        if (set(ops) != {'DecodeImage', 'DetLabelEncode', 'DetResizeForTest', 'NormalizeImage', 'ToCHWImage', 'KeepKeys'}
                or ops['DetResizeForTest'] is not None
                or ops.get('NormalizeImage') != expected or post.get('name') != 'DBPostProcess'
                or type(post.get('max_candidates')) is not int or not 1 <= post['max_candidates'] <= 3000
                or any(type(post.get(k)) not in (int, float) or not math.isfinite(post[k]) or post[k] <= 0
                       for k in ('thresh', 'box_thresh', 'unclip_ratio'))
                or post.get('thresh', 0) >= 1 or post.get('box_thresh', 0) > 1 or post.get('unclip_ratio', 0) > 10):
            raise ValueError('Unsupported detection processor configuration')


class PPOCRONNX:
    def __init__(self, det, rec, configuration):
        import cv2
        import numpy as np
        import onnxruntime as ort
        import yaml
        self.cv, self.np, self.configuration = cv2, np, settings(configuration)
        self.models, self.configs = {}, {}
        for name, root in [('det', det), ('rec', rec)]:
            config = yaml.safe_load((Path(root) / 'inference.yml').read_text())
            validate_model_configuration(config, name, configuration[('detection' if name == 'det' else 'recognition') + '_model_name'])
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.configuration['cpu_threads']
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            model = ort.InferenceSession(str(Path(root) / 'inference.onnx'), sess_options=options,
                                        providers=['CPUExecutionProvider'])
            if model.get_providers() != ['CPUExecutionProvider']:
                raise ValueError('Explicit CPU ONNX execution required')
            inputs, outputs = model.get_inputs(), model.get_outputs()
            if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != 'tensor(float)'
                    or outputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3
                    or (name == 'rec' and (inputs[0].shape[2] != 48 or len(outputs[0].shape) != 3
                        or outputs[0].shape[-1] != len(config['PostProcess']['character_dict']) + 2))
                    or (name == 'det' and (len(outputs[0].shape) != 4 or outputs[0].shape[1] != 1))):
                raise ValueError('Unsupported pinned OCR ONNX tensor shapes/types')
            self.models[name], self.configs[name] = model, config
        self.characters = ['blank'] + self.configs['rec']['PostProcess']['character_dict'] + [' ']

    def _run(self, kind, tensor):
        tick = time.monotonic()
        model = self.models[kind]
        output = model.run(None, {model.get_inputs()[0].name: tensor})
        self.calls.append(dict(kind=kind, seconds=time.monotonic() - tick,
            input_shape=list(tensor.shape), input_tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
            outputs=[dict(shape=list(v.shape), sha256=hashlib.sha256(v.tobytes()).hexdigest()) for v in output],
            provider='CPUExecutionProvider'))
        return output[0]

    def recognize(self, image):
        cv, np = self.cv, self.np
        height, width = image.shape[:2]
        padded = min(3200, int(48 * max(320 / 48, width / height)))
        resized = min(padded, math.ceil(48 * width / height))
        pixels = cv.resize(image, (resized, 48)).astype(np.float32).transpose(2, 0, 1)
        tensor = np.zeros((1, 3, 48, padded), np.float32)
        tensor[0, :, :, :resized] = (pixels / 255 - .5) / .5
        scores = np.asarray(self._run('rec', tensor))
        if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[-1] != len(self.characters):
            raise ValueError('Recognizer dictionary and output shape differ')
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite OCR output')
        indices, confidence = scores[0].argmax(-1), scores[0].max(-1)
        text, selected, previous = [], [], None
        for index, score in zip(indices.tolist(), confidence.tolist(), strict=True):
            if index and index != previous:
                text.append(self.characters[index])
                selected.append(score)
            previous = index
        return ''.join(text), float(np.mean(selected)) if selected else 0.

    def _mini(self, contour):
        cv, np = self.cv, self.np
        rectangle = cv.minAreaRect(np.asarray(contour, np.float32))
        points = sorted(cv.boxPoints(rectangle).tolist(), key=lambda p: p[0])
        left, right = sorted(points[:2], key=lambda p: p[1]), sorted(points[2:], key=lambda p: p[1])
        return np.float32([left[0], right[0], right[1], left[1]]), min(rectangle[1])

    def detect(self, image):
        import pyclipper
        cv, np = self.cv, self.np
        height, width = image.shape[:2]
        ratio = max(1., 736 / min(height, width))
        rh, rw = int(height * ratio), int(width * ratio)
        if max(rh, rw) > 4000:
            ratio = 4000 / max(rh, rw)
            rh, rw = int(rh * ratio), int(rw * ratio)
        rh, rw = max(32, round(rh / 32) * 32), max(32, round(rw / 32) * 32)
        resized = cv.resize(image, (rw, rh)).astype(np.float32) / 255
        normalized = (resized - np.float32([.485, .456, .406])) / np.float32([.229, .224, .225])
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        prediction = np.asarray(self._run('det', tensor))[0, 0]
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite detection output')
        parameters = self.configs['det']['PostProcess']
        contours, _ = cv.findContours((prediction > parameters['thresh']).astype(np.uint8) * 255,
                                     cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        examined = min(len(contours), parameters['max_candidates'])
        self.detection_info = dict(total_contours=len(contours), examined_contours=examined,
            unexamined_contours=len(contours)-examined, contour_limit=parameters['max_candidates'],
            coverage='partial_unexamined_contours' if len(contours) > examined else 'all_found_contours_examined')
        boxes = []
        for contour in contours[:parameters['max_candidates']]:
            points, side = self._mini(contour)
            if side < 3:
                continue
            x1, x2 = max(0, math.floor(points[:, 0].min())), min(prediction.shape[1]-1, math.ceil(points[:, 0].max()))
            y1, y2 = max(0, math.floor(points[:, 1].min())), min(prediction.shape[0]-1, math.ceil(points[:, 1].max()))
            mask = np.zeros((y2-y1+1, x2-x1+1), np.uint8)
            cv.fillPoly(mask, [(points-[x1, y1]).astype(np.int32)], 1)
            score = float(cv.mean(prediction[y1:y2+1, x1:x2+1], mask)[0])
            if score < parameters['box_thresh']:
                continue
            perimeter = cv.arcLength(points, True)
            if perimeter <= 0:
                continue
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            paths = offset.Execute(cv.contourArea(points) * parameters['unclip_ratio'] / perimeter)
            if len(paths) != 1:
                continue
            points, side = self._mini(paths[0])
            if side < 5:
                continue
            points[:, 0] = np.clip(np.rint(points[:, 0]*width/prediction.shape[1]), 0, width)
            points[:, 1] = np.clip(np.rint(points[:, 1]*height/prediction.shape[0]), 0, height)
            boxes.append(points.astype(int).tolist())
        return sorted(boxes, key=lambda p: (np.mean(np.asarray(p)[:, 1]), np.min(np.asarray(p)[:, 0])))

    def crop(self, image, polygon):
        cv, np = self.cv, self.np
        points = np.asarray(polygon, np.float32)
        width = max(1, int(max(np.linalg.norm(points[0]-points[1]), np.linalg.norm(points[2]-points[3]))))
        height = max(1, int(max(np.linalg.norm(points[0]-points[3]), np.linalg.norm(points[1]-points[2]))))
        matrix = cv.getPerspectiveTransform(points, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
        return cv.warpPerspective(image, matrix, (width, height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    def predict(self, pixels):
        self.calls = []
        self.detection_info = None
        height, width = pixels.shape[:2]
        mode = self.configuration['input_mode']
        polygons = ([[[0, 0], [width, 0], [width, height], [0, height]]]
                    if mode == 'field_roi' else self.detect(pixels))
        if len(polygons) > self.configuration['max_recognitions']:
            raise ValueError('OCR recognition capacity exceeded; no recognition calls executed')
        texts, scores = [], []
        for polygon in polygons:
            text, score = self.recognize(pixels if mode == 'field_roi' else self.crop(pixels, polygon))
            texts.append(text)
            scores.append(score)
        return dict(results=[dict(rec_texts=texts, rec_scores=scores, rec_polys=polygons,
            rec_boxes=[[min(p[0] for p in box), min(p[1] for p in box), max(p[0] for p in box), max(p[1] for p in box)] for box in polygons])],
            onnx_execution=dict(settings=self.configuration, calls=self.calls,
                detection_invoked=mode == 'full_image',
                detection_coverage=self.detection_info,
                geometry_basis='supplied_field_extent_not_detector_or_physical_identity' if mode == 'field_roi' else 'model_detected_text_polygons',
                det_parameters=self.configs['det']['PostProcess'],
                preprocessing='BGR; det resize/mean/std; rec height48, normalize [-1,1], zero right padding; reported model tensors are transformed',
                scores_are_calibrated_accuracy=False, complete_paddleocr_sdk_pipeline=False))
