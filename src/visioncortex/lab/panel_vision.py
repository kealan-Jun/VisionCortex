"""Local panel registration by stable features, with perspective and ambiguity gates.

Templates describe layouts, not physical instrument identities. Dynamic digits
are masked out of feature matching. No old customer detector weights are used.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re

from PIL import Image

from .domain import fingerprint
from .gauge import exact
from .ocr_fields import ROLES

REVISION = 'vc-lab-panel-registration/1'


def libraries():
    import cv2
    import numpy as np
    return cv2, np


def read_image(path):
    cv, np = libraries()
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 24*1024**2:
        raise ValueError('Image path/byte bound exceeded')
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as header:
        if header.width*header.height > 16_000_000 or getattr(header, 'n_frames', 1) != 1:
            raise ValueError('Invalid or oversized image')
    image = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    if image is None or image.shape[0]*image.shape[1] > 16_000_000:
        raise ValueError('Invalid or oversized image')
    return image, hashlib.sha256(raw).hexdigest()


def quad(value, width, height):
    _, np = libraries()
    points = np.asarray(value, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError('Normalized TL/TR/BR/BL quadrilateral required')
    return points * np.array([width-1, height-1], np.float32)


def warp(image, source_quad, size):
    cv, np = libraries()
    width, height = size
    if type(width) is not int or type(height) is not int or not 32 <= width <= 2048 or not 32 <= height <= 2048:
        raise ValueError('Rectified panel size must be 32..2048')
    points = np.asarray(source_quad, np.float32)
    contour = points.reshape(-1, 1, 2)
    if not cv.isContourConvex(contour) or cv.contourArea(contour, oriented=True) <= 16:
        raise ValueError('Panel quadrilateral is degenerate, mirrored or unordered')
    target = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], np.float32)
    matrix = cv.getPerspectiveTransform(points, target)
    return cv.warpPerspective(image, matrix, (width,height)), matrix


class PanelProfiles:
    def __init__(self, path, expected_sha256, *, data_scope, model_root):
        self.path, self.root = Path(path), Path(model_root).resolve()
        if self.path.is_symlink() or not self.path.resolve().is_relative_to(self.root) or self.path.stat().st_size > 256*1024:
            raise ValueError('Invalid panel manifest')
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Panel manifest fingerprint mismatch')
        self.document = json.loads(raw)
        if self.document.get('schema_version') != 'vc-lab-panel-profiles/1' or not self.document.get('revision'):
            raise ValueError('Versioned panel manifest required')
        if self.document.get('status') not in {'reviewed', 'research'} or (data_scope == 'customer' and self.document['status'] != 'reviewed'):
            raise ValueError('Research layouts cannot be activated for customer data')
        self.fingerprint = fingerprint(self.document)
        profiles = self.document.get('profiles', [])
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= 32:
            raise ValueError('One to 32 panel profiles required')
        self.profiles = []
        ids = set()
        cv, np = libraries()
        for profile in profiles:
            if not isinstance(profile.get('id'), str) or not 1 <= len(profile['id']) <= 128 or profile['id'] in ids or 'instrument_id' in profile:
                raise ValueError('Unique layout identity required; appearance cannot assign an instrument')
            ids.add(profile['id'])
            ref = profile['reference']
            relative = Path(ref['path'])
            if relative.is_absolute() or '..' in relative.parts or not re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')):
                raise ValueError('Relative source-pinned local template required')
            path = (self.root / ref['path'])
            if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(self.root)) or not path.resolve().is_relative_to(self.root):
                raise ValueError('Panel reference escaped its local model root')
            image, digest = read_image(path)
            if digest != ref['sha256']:
                raise ValueError('Panel reference fingerprint mismatch')
            canonical, _ = warp(image, quad(ref['quad'], image.shape[1], image.shape[0]), profile['size'])
            mask = np.full(canonical.shape[:2], 255, np.uint8)
            if profile.get('identity_qr_box'):
                x1,y1,x2,y2=map(float,profile['identity_qr_box'])
                if not 0<=x1<x2<=1 or not 0<=y1<y2<=1:
                    raise ValueError('Invalid registered panel QR area')
                cv.rectangle(mask,(int(x1*mask.shape[1]),int(y1*mask.shape[0])),
                             (int(x2*mask.shape[1]),int(y2*mask.shape[0])),0,-1)
            regions = profile.get('regions', [])
            if not isinstance(regions, list) or len(regions) > 128:
                raise ValueError('At most 128 layout regions')
            keys = set()
            for region in regions:
                key = (region.get('field_key'), region.get('role'), region.get('channel', ''))
                if not isinstance(key[0], str) or not key[0] or key[1] not in ROLES or key in keys:
                    raise ValueError('Unique named field/role/channel required')
                keys.add(key)
                dtype = region.get('data_type', 'number')
                if dtype not in {'number', 'text', 'clock'} or region.get('display_color') not in {None, 'red', 'green'}:
                    raise ValueError('Unsupported field type/display color')
                if region.get('display_color') and dtype != 'number':
                    raise ValueError('LED numeric masks cannot select text/clock fields')
                bases = {'verified_on_panel', 'verified_instrument_specification'}
                if self.document['status'] == 'research':
                    bases.add('source_visible_research_annotation')
                if region.get('unit') and region.get('unit_basis') not in bases:
                    raise ValueError('Field unit lacks its verified/research basis')
                if region.get('decimal_places') is not None:
                    if type(region['decimal_places']) is not int or not 0 <= region['decimal_places'] <= 8 or region.get('decimal_places_basis') not in bases:
                        raise ValueError('Explicit verified display precision required')
                if not 0 <= exact(region.get('minimum_score', '0.8')) <= 1:
                    raise ValueError('Invalid field score threshold')
                for bound in ('min_value', 'max_value'):
                    if region.get(bound) is not None:
                        exact(region[bound])
                if region.get('min_value') is not None and region.get('max_value') is not None and exact(region['min_value']) > exact(region['max_value']):
                    raise ValueError('Invalid field range')
                x1,y1,x2,y2 = [float(x) for x in region['box']]
                if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                    raise ValueError('Invalid layout field rectangle')
                cv.rectangle(mask, (int(x1*mask.shape[1]),int(y1*mask.shape[0])),
                             (int(x2*mask.shape[1]),int(y2*mask.shape[0])), 0, -1)
            gauges = profile.get('gauges', [])
            if not isinstance(gauges, list) or len(gauges) > 16 or not regions and not gauges:
                raise ValueError('Bounded nonempty fields/gauges required')
            for gauge in gauges:
                if not gauge.get('field_key'):
                    raise ValueError('Named gauge required')
                if 'center' not in gauge and 'radius' not in gauge:
                    continue  # Generic rim candidate: no calibrated value until a profile supplies geometry.
                center, radius = gauge.get('center'), gauge.get('radius')
                if not isinstance(center, list) or len(center) != 2 or not all(math.isfinite(float(v)) for v in [*center, radius]):
                    raise ValueError('Finite gauge center/radius required')
                x, y = map(float, center)
                radius = float(radius)
                if not 24 <= radius <= 2048 or not radius <= x < mask.shape[1]-radius or not radius <= y < mask.shape[0]-radius:
                    raise ValueError('Complete gauge must lie in the rectified panel')
                calibration = gauge.get('calibration')
                if calibration and (calibration.get('status') != 'verified' or not calibration.get('instrument_id') or calibration.get('unit') != gauge.get('unit')):
                    raise ValueError('Gauge calibration must be verified for a physical instrument/unit')
                cv.circle(mask, tuple(round(float(v)) for v in center), round(radius*.91), 0, -1)
            feature = cv.SIFT_create(nfeatures=4000, contrastThreshold=.025)
            points, descriptors = feature.detectAndCompute(cv.cvtColor(canonical,cv.COLOR_BGR2GRAY), mask)
            if descriptors is None or len(points) < 12:
                raise ValueError('Panel template has insufficient stable features')
            self.profiles.append((profile, points, descriptors))

    def locate(self, image):
        cv, np = libraries()
        height, width = image.shape[:2]
        scale = min(1, 1600/max(height,width))
        scene = cv.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        points, descriptors = cv.SIFT_create(nfeatures=6000,contrastThreshold=.025).detectAndCompute(cv.cvtColor(scene,cv.COLOR_BGR2GRAY),None)
        if descriptors is None:
            return dict(status='unmatched', algorithm_revision=REVISION, candidates=[])
        candidates = []
        for profile, reference_points, reference_descriptors in self.profiles:
            pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(reference_descriptors, descriptors, k=2)
            good = [m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance]
            # A scene feature may not vote repeatedly for several template points.
            unique = {}
            for m in sorted(good,key=lambda m:m.distance):
                unique.setdefault(m.trainIdx,m)
            good=list(unique.values())
            if len(good)<12:
                continue
            source=np.float32([reference_points[m.queryIdx].pt for m in good])
            target=np.float32([points[m.trainIdx].pt for m in good])
            matrix,mask=cv.findHomography(source,target,cv.RANSAC,3,maxIters=2000,confidence=.995)
            if matrix is None or not np.isfinite(matrix).all():
                continue
            inliers=mask.ravel().astype(bool);n=int(inliers.sum());ratio=n/len(good)
            pw,ph=profile['size']
            coverage=cv.contourArea(cv.convexHull(source[inliers]))/(pw*ph)
            occupied={(int(x>=pw/2),int(y>=ph/2)) for x,y in source[inliers]}
            if n<12 or ratio<.5 or coverage<.15 or len(occupied)<3:
                continue
            corners=np.float32([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]])
            projected=cv.perspectiveTransform(corners.reshape(1,-1,2),matrix).reshape(4,2)/scale
            contour=projected.reshape(-1,1,2)
            area=cv.contourArea(contour,oriented=True)
            if not cv.isContourConvex(contour) or not .005*width*height <= area <= width*height or (projected<0).any() or (projected[:,0]>=width).any() or (projected[:,1]>=height).any():
                continue
            predicted=cv.perspectiveTransform(source[inliers].reshape(1,-1,2),matrix).reshape(-1,2)
            error=float(np.median(np.linalg.norm(predicted-target[inliers],axis=1)))/scale
            score=ratio*min(1,coverage/.5)*min(1,n/40)
            candidates.append(dict(profile_id=profile['id'],quad=projected.tolist(),size=profile['size'],
                                   inliers=n,matches=len(good),inlier_ratio=ratio,coverage=coverage,
                                   reprojection_error_pixels=error,score=score))
        candidates.sort(key=lambda x:x['score'],reverse=True)
        if not candidates:
            return dict(status='unmatched',algorithm_revision=REVISION,candidates=[])
        if len(candidates)>1 and candidates[0]['score']-candidates[1]['score'] < .12:
            return dict(status='ambiguous_layout',algorithm_revision=REVISION,candidates=candidates[:4])
        return dict(status='candidate',algorithm_revision=REVISION,match=candidates[0],candidates=candidates[:4],
                    profile_fingerprint=self.fingerprint,physical_instrument_identity='unassigned')

    def extract(self, image, match):
        canonical, matrix = warp(image,match['quad'],match['size'])
        profile=next(p for p,_,_ in self.profiles if p['id']==match['profile_id'])
        return canonical,matrix,profile
