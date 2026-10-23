"""Recall lab-work video context without promoting physical action events.

Fine-action acceptance is intentionally stricter than watchable process recall.
This owner uses repeated, source-bound hand motion relative to laboratory
anchors. A hand/object label, camera pan or a stationary glove cannot suffice.
"""
import math
from statistics import median

ACTORS = frozenset({'hand', 'gloved_hand'})
OBJECTS = frozenset({'balance', 'beaker', 'container', 'reagent_bottle',
    'reagent_bottle_open', 'sample_bottle', 'sample_bottle_blue', 'pipette',
    'spearhead', 'spatula', 'tube', 'tube_rack', 'magnetic_stirrer'})
RECIPE = 'source-bound-local-lab-work-recall/1'


def center(box):
    a, b, c, d = box['xyxy_norm']
    return ((a+c)/2, (b+d)/2)


def gap(left, right):
    a, b = left['xyxy_norm'], right['xyxy_norm']
    return math.hypot(max(0, a[0]-b[2], b[0]-a[2]), max(0, a[1]-b[3], b[1]-a[3]))


def area(box):
    a, b, c, d = box['xyxy_norm']
    return max(0, c-a)*max(0, d-b)


def observations(boxes, *, confidence=.25, maximum_gap=.12):
    actors = [b for b in boxes if b['class_name'] in ACTORS and b['confidence'] >= confidence
              and .0001 <= area(b) <= .10]
    objects = [b for b in boxes if b['class_name'] in OBJECTS and b['confidence'] >= confidence
               and .0001 <= area(b) <= .40]
    # Multiple visible laboratory anchors supply scene context, not a claim
    # that any particular 2-D overlap proves contact or object identity.
    actors = [a for a in actors if len(objects) >= 2 and any(gap(a, b) <= maximum_gap for b in objects)]
    return actors, objects


def _translation(previous, current):
    pairs, used = [], set()
    for left in previous:
        options = [(math.dist(center(left), center(right)), n, right)
                   for n, right in enumerate(current)
                   if n not in used and left['class_name'] == right['class_name']
                   and .5 <= area(left)/max(area(right), 1e-9) <= 2]
        if not options:
            continue
        distance, n, right = min(options, key=lambda item: item[:2])
        if distance <= .15:
            used.add(n)
            a, b = center(left), center(right)
            pairs.append((b[0]-a[0], b[1]-a[1]))
    if len(pairs) < 2:
        return None
    return median(x for x, _ in pairs), median(y for _, y in pairs)


def selected_runs(rows, *, period_ms, motion_threshold=.01, minimum_observations=3):
    """Bound selection by an uninterrupted series of actual observations.

    Missing actors, unreadable/unbound frames and sampling holes split a run.
    Endpoints are actual observed native times; there is no minimum video
    length, invented padding, state-change claim or bridge over unknown rows.
    """
    runs, current = [], []
    for row in rows:
        valid = row.get('source_verified') is True and bool(row.get('actors')) and len(row.get('objects', [])) >= 2
        adjacent = not current or (0 < row['local_ms']-current[-1]['local_ms'] <= period_ms*1.1
                                  and _translation(current[-1]['objects'], row.get('objects', [])) is not None)
        if not valid or not adjacent:
            if current:
                runs.append(current)
            current = []
        if valid:
            current.append(row)
    if current:
        runs.append(current)
    intervals, decisions = [], []
    for run in runs:
        tracks, serial, displacement = {}, 0, (0., 0.)
        for index, row in enumerate(run):
            translation = _translation(run[index-1]['objects'], row['objects']) if index else (0., 0.)
            if translation is None:
                tracks = {}
                displacement = (0., 0.)
            else:
                displacement = (displacement[0]+translation[0], displacement[1]+translation[1])
            used = set()
            for actor in row['actors']:
                point = center(actor)
                options = [(math.dist(track['last'], (point[0]-displacement[0], point[1]-displacement[1])), name)
                           for name, track in tracks.items() if name not in used and track['last_index'] == index-1]
                distance, name = min(options) if options else (float('inf'), None)
                corrected = (point[0]-displacement[0], point[1]-displacement[1])
                if distance > .10:
                    serial += 1
                    name = serial
                    tracks[name] = {'points': [], 'last': corrected, 'last_index': index}
                track = tracks[name]
                track['points'].append(corrected)
                track.update(last=corrected, last_index=index)
                used.add(name)
        movement = []
        for track in tracks.values():
            points = track['points']
            if len(points) < minimum_observations:
                continue
            # Trim extremes for longer runs. A single detector outlier cannot
            # turn an otherwise stationary glove into a process episode.
            trim = max(1, len(points)//10) if len(points) >= 5 else 0
            xs, ys = sorted(p[0] for p in points), sorted(p[1] for p in points)
            spread = math.hypot(xs[-1-trim]-xs[trim], ys[-1-trim]-ys[trim])
            if len(points) < 5 and len({(round(x, 3), round(y, 3)) for x, y in points}) < 3:
                continue
            if spread+1e-9 >= motion_threshold:
                movement.append({'observation_count': len(points), 'camera_relative_spread_norm': spread})
        if movement and run[-1]['local_ms'] > run[0]['local_ms']:
            interval = (run[0]['local_ms'], run[-1]['local_ms'])
            intervals.append(interval)
            decisions.append({'basis': RECIPE, 'start_ms': interval[0], 'end_ms': interval[1],
                'sample_count': len(run), 'sample_native_pts': [r['source_frame']['source_pts'] for r in run],
                'motion_support': movement, 'scope': 'lab_work_video_candidate_not_physical_action',
                'physical_action_confirmed': False, 'event_acceptance_mutated': False})
    return intervals, decisions
