"""Published recorder-clock conversion shared by analysis and playback."""
from bisect import bisect_left
import math


def valid_points(clock):
    points = clock.get('points') or []
    if (len(points) >= 2
            and all(len(p) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                      and math.isfinite(v) for v in p) for p in points)
            and all(a[0] < b[0] and a[1] < b[1] for a, b in zip(points, points[1:], strict=False))):
        return points
    return []


def interpolate(points, value, axis):
    index = max(1, min(len(points)-1, bisect_left([p[axis] for p in points], value)))
    left, right = points[index-1:index+1]
    output = 1-axis
    return left[output] + (value-left[axis])/(right[axis]-left[axis])*(right[output]-left[output])


def capture_us(clock, local_ms):
    points = valid_points(clock)
    if clock.get('points') and not points:
        raise ValueError('Published recorder clock is malformed or non-monotonic')
    return round(interpolate(points, local_ms, 0)) if points else clock['origin_us'] + round(local_ms*1000)


def media_ms(clock, capture_us, origin_us):
    points = valid_points(clock) if clock.get('basis') == 'recorder_csv_interpolation' else []
    return ((interpolate(points, capture_us, 1), 'recorder_csv_interpolation') if points else
            ((capture_us-origin_us)/1000, 'capture_start_plus_media_time_estimate'))
