from visioncortex.analysis.experiments import (
    ExperimentAnalysisOperations,
    compose_experiments,
)
from visioncortex.pipeline import EvidencePipeline
from visioncortex.video_io import _merge_windows


def test_scan_and_decode_share_the_same_interval_contract():
    # Unordered, overlapping, touching and empty windows must route the same
    # coverage to progressive scanning and physical decoder sessions.
    windows = [(8, 10), (2, 4), (4, 6), (3, 5), (5, 5), (10, 9)]
    expected = [(2.0, 6.0), (8.0, 10.0)]
    assert EvidencePipeline._merge_time_windows(windows) == expected
    assert _merge_windows(windows) == expected


def test_composition_preserves_policy_inputs_and_independent_receipts():
    events, views, coarse, config = [], [], [], {"profile": "sealed-recorder"}
    raw, normalized, formal, groups, keys = (
        ["raw"],
        ["normalized"],
        ["formal"],
        ["group"],
        ["key"],
    )
    calls = []
    boundaries = [{"rule": "adapter-full-timeline"}]

    def build(e, v, c, *, coarse_windows, decision_receipts):
        assert (e, v, c, coarse_windows) == (events, views, config, coarse)
        decision_receipts.append({"rule": "boundary"})
        calls.append("build")
        return raw

    def normalize(segments, e, v, c, *, decision_receipts):
        assert segments is raw
        decision_receipts.append({"rule": "normalize"})
        calls.append("normalize")
        return normalized

    def admit(segments, e, v, candidates, c):
        assert segments is normalized and candidates is coarse
        calls.append("formal")
        return formal, [{"rule": "formal"}]

    def group(segments, e, v, c, *, decision_receipts, coarse_windows):
        assert segments is formal and coarse_windows is coarse
        decision_receipts.append({"rule": "continuity"})
        calls.append("group")
        return groups

    def select(g, segments, e, c, *, decision_receipts):
        assert g is groups and segments is formal
        decision_receipts.append({"rule": "selection"})
        calls.append("select")
        return keys

    result = compose_experiments(
        events,
        views,
        coarse,
        config,
        operations=ExperimentAnalysisOperations(build, normalize, admit, group, select),
        boundary_decisions=boundaries,
    )
    assert calls == ["build", "normalize", "formal", "group", "select"]
    assert result.selected_key_events is keys
    assert result.boundary_decisions is boundaries
    assert result.boundary_decisions == [
        {"rule": "adapter-full-timeline"},
        {"rule": "boundary"},
    ]
    assert result.normalization_decisions == [{"rule": "normalize"}]
    assert result.formal_decisions == [{"rule": "formal"}]
    assert result.continuity_decisions == [{"rule": "continuity"}]
    assert result.selection_decisions == [{"rule": "selection"}]


def test_progressive_preview_omits_receipt_collectors_without_forking_composition():
    calls = []

    def build(e, v, c, *, coarse_windows):
        calls.append("build")
        return []

    def normalize(s, e, v, c):
        calls.append("normalize")
        return s

    def admit(s, e, v, coarse, c):
        calls.append("formal")
        return s, [{"rule": "formal"}]

    def group(s, e, v, c, *, coarse_windows):
        calls.append("group")
        return []

    def select(g, s, e, c):
        calls.append("select")
        return []

    result = compose_experiments(
        [],
        [],
        [],
        {},
        operations=ExperimentAnalysisOperations(build, normalize, admit, group, select),
        collect_decisions=False,
    )
    assert calls == ["build", "normalize", "formal", "group", "select"]
    assert result.selection_decisions == result.continuity_decisions == []
    assert result.formal_decisions == [{"rule": "formal"}]
