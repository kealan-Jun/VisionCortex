"""Pure interval algebra shared by scan planning and media decoding.

Endpoints keep the caller's time unit. Invalid or zero-length windows are
ignored; touching windows share a boundary and therefore form one interval.
"""

from collections.abc import Iterable


def merge_time_windows(
    windows: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(
        (float(start), float(end)) for start, end in windows if end > start
    ):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
